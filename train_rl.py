import argparse
import os

import yaml
import torch
import torchvision.transforms as transforms
from torch.optim import Adam
import datetime
import wandb

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from rl_policy import EtaPolicyNetwork, train_rl_policy
from util.img_utils import total_variation_loss


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--gpu', type=int, default=0)

    # RL Hyperparameters
    parser.add_argument('--num_episodes', type=int, default=500, help="Number of RL episodes to run")
    parser.add_argument('--rl_lr', type=float, default=1e-3, help="Learning rate for the RL policy")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimensions for the MLP policy")

    # Wandb Configs
    parser.add_argument('--wandb_project', type=str, default="diffusion-rl-eta", help="Wandb project name")
    parser.add_argument('--wandb_name', type=str, default=None, help="Specific run name for wandb")

    parser.add_argument('--max_steps', type=int, default=50, help="Number of diffusion steps per RL episode")
    parser.add_argument('--step_penalty', type=float, default=0.01, help="Penalty applied per step taken")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.wandb_name if args.wandb_name else f"run_{timestamp}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)  # Logs all argparse arguments automatically!
    )

    # 1. Load Configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # SAFEGUARD: Ensure we are using our new RL conditioning method
    if task_config['conditioning']['method'] == 'adaptive_ps':
        raise ValueError("Cannot train RL using 'adaptive_ps'. Change your yaml config to 'rl_ps'.")

    # 2. Setup Diffusion Model & Operators (Identical to sample_condition.py)
    model = create_model(**model_config).to(device)
    model.eval()

    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning

    sampler = create_sampler(**diffusion_config)

    # 3. Setup Data
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Grab one real batch to use as our training environment
    ref_img = next(iter(loader)).to(device)
    y = operator.forward(ref_img)
    noisy_measurement = noiser(y)

    # 4. Setup RL Agent
    policy_net = EtaPolicyNetwork(hidden_dim=args.hidden_dim).to(device)
    optimizer = Adam(policy_net.parameters(), lr=args.rl_lr)

    # 5. Define Reward (Example: negative L1 distance to ground truth measurement)
    def reward_fn(generated_img, target_measurement):
        # 1. Data Consistency (How well does it match the measurement?)
        simulated_measurement = operator.forward(generated_img)
        consistency_loss = torch.nn.functional.mse_loss(simulated_measurement, target_measurement)

        # 2. Generated Quality (How smooth/natural is the image?)
        quality_loss = total_variation_loss(generated_img, weight=1.0)

        # Combine them (Negative because we want to MAXIMIZE the reward)
        # Tune the 0.1 weight based on what looks best
        total_loss = consistency_loss + (0.1 * quality_loss)
        return -total_loss

    data_kwargs = {
        'model': model,
        'x_start': torch.randn_like(ref_img, device=device).requires_grad_(),
        'measurement': noisy_measurement,
        'measurement_cond_fn': measurement_cond_fn,
        'record': False,
        'save_root': f'./results/rl_training_{timestamp}'
    }

    # 6. Train!
    print(f"Starting RL Training for {args.num_episodes} episodes (Max {args.max_steps} steps per ep)...")
    train_rl_policy(
        diffusion_model=sampler,
        policy_net=policy_net,
        optimizer=optimizer,
        num_episodes=args.num_episodes,
        reward_fn=reward_fn,
        data_kwargs=data_kwargs,
        max_steps = args.max_steps,
        step_penalty = args.step_penalty,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_dir = os.path.join("models", timestamp)
    os.makedirs(model_save_dir, exist_ok=True)

    save_path = os.path.join(model_save_dir, "optimized_eta_policy.pth")
    torch.save(policy_net.state_dict(), save_path)
    print(f"Training complete. Policy saved to: {save_path}")


if __name__ == '__main__':
    main()