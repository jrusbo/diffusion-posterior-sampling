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
from util.img_utils import total_variation_loss, calculate_psnr


from util.logger import get_logger
logger = get_logger()


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

    # Wandb Configs
    parser.add_argument('--wandb_project', type=str, default="diffusion-rl-eta", help="Wandb project name")
    parser.add_argument('--wandb_name', type=str, default=None, help="Specific run name for wandb")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    # 1. Load Configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # Extract RL config from YAML and ensure correct types
    rl_config = task_config.get('rl_config', {})
    num_episodes = int(rl_config.get('num_episodes', 500))
    rl_lr = float(rl_config.get('rl_lr', 1e-4))
    hidden_dim = int(rl_config.get('hidden_dim', 128))
    max_steps = int(rl_config.get('max_steps', 64))
    batch_size = int(rl_config.get('batch_size', 4))
    baseline_decay = float(rl_config.get('baseline_decay', 0.99))
    num_workers = int(rl_config.get('num_workers', 4))
    pin_memory = bool(rl_config.get('pin_memory', True))
    entropy_beta = float(rl_config.get('entropy_beta', 0.01))
    grad_clip = float(rl_config.get('grad_clip', 1.0))
    save_interval = int(rl_config.get('save_interval', 500))

    # Policy Hyperparameters
    eta_max = float(rl_config.get('eta_max', 5.0))
    policy_std = float(rl_config.get('policy_std', 0.3))
    state_norm_config = rl_config.get('state_norm', {})

    # Reward Weights
    reward_config = rl_config.get('reward_weights', {})
    w_psnr = float(reward_config.get('psnr', 1.0))
    w_consistency = float(reward_config.get('consistency', 500.0))
    w_tv = float(reward_config.get('tv', 5.0))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args.wandb_name if args.wandb_name else f"run_{timestamp}"

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=task_config
    )

    # Use run name for saving
    model_save_dir = os.path.join("models", run_name)
    os.makedirs(model_save_dir, exist_ok=True)
    save_path = os.path.join(model_save_dir, "optimized_eta_policy.pth")

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
    loader = get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, train=True, pin_memory=pin_memory)

    # 4. Setup RL Agent
    policy_net = EtaPolicyNetwork(
        hidden_dim=hidden_dim, 
        eta_max=eta_max, 
        std=policy_std,
        state_norm_config=state_norm_config
    ).to(device)
    optimizer = Adam(policy_net.parameters(), lr=rl_lr)

    # 5. Define Reward (Using PSNR and quality metrics)
    def reward_fn(generated_img, target_measurement, ref_img=None):
        if ref_img is None:
            raise ValueError("Training Reward Error: 'ref_img' (ground truth) is required for PSNR-based training.")

        with torch.no_grad():
            clean_measurement = operator.forward(ref_img)
        
        simulated_measurement = operator.forward(generated_img)
        # Per-sample MSE consistency
        consistency_loss = (simulated_measurement - clean_measurement).pow(2).mean(dim=list(range(1, simulated_measurement.ndim)))

        # 2. Generated Quality (How smooth/natural is the image?)
        # total_variation_loss is now per-sample.
        quality_loss = total_variation_loss(generated_img, weight=1.0)

        # 3. PSNR (per-sample)
        psnr_val = calculate_psnr(generated_img, ref_img)

        reward = w_psnr * psnr_val - w_consistency * consistency_loss - w_tv * quality_loss
        
        # Return dict of components for logging
        metrics = {
            "psnr": psnr_val,
            "consistency_mse": consistency_loss,
            "tv_loss": quality_loss
        }
        return reward, metrics

    # 6. Train!
    logger.info(f"Starting RL Training: {run_name}")        
    train_rl_policy(
        diffusion_model=sampler,
        model=model,
        policy_net=policy_net,
        optimizer=optimizer,
        loader=loader,
        operator=operator,
        noiser=noiser,
        measurement_cond_fn=measurement_cond_fn,
        num_episodes=num_episodes,
        reward_fn=reward_fn,
        max_steps=max_steps,
        device=device,
        baseline_decay=baseline_decay,
        entropy_beta=entropy_beta,
        grad_clip=grad_clip,
        save_path=save_path,
        save_interval=save_interval
    )


if __name__ == '__main__':
    main()
