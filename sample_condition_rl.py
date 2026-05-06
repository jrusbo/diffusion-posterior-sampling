import os
import argparse
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color
from rl_policy import EtaPolicyNetwork


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--policy_weights', type=str, required=True, help="Path to optimized_eta_policy.pth")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/rl_inference')
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')

    # 1. Load Configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    # 2. Setup Diffusion Model & Operators
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

    # 4. Load the Trained RL Policy
    policy_net = EtaPolicyNetwork(hidden_dim=64).to(device)
    policy_net.load_state_dict(torch.load(args.policy_weights, map_location=device))
    policy_net.eval()  # Crucial: set to evaluation mode

    # 5. Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # 6. Do Inference
    for i, ref_img in enumerate(loader):
        print(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)

        y = operator.forward(ref_img)
        y_n = noiser(y)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()

        # We don't need gradients for the policy or diffusion model during inference
        with torch.no_grad():
            sample = sampler.p_sample_loop(
                model=model,
                x_start=x_start,
                measurement=y_n,
                measurement_cond_fn=measurement_cond_fn,
                record=True,
                save_root=out_path,
                rl_mode=True,
                policy_net=policy_net
            )

        plt.imsave(os.path.join(out_path, 'input', fname), clear_color(y_n))
        plt.imsave(os.path.join(out_path, 'label', fname), clear_color(ref_img))
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_color(sample))


if __name__ == '__main__':
    main()