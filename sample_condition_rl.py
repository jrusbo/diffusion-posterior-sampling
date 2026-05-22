import os
import argparse
from datetime import datetime
from pathlib import Path
import re
import yaml
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset
from util.compute_progress_metrics import (
    aggregate_progress_metrics,
    collect_progress_metrics,
    save_progress_metrics,
)
from util.img_utils import clear_color, mask_generator
from util.logger import get_logger
from rl_policy import EtaPolicyNetwork


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_ref_and_fname(dataset, dataset_index: int, device: torch.device):
    if hasattr(dataset, 'fpaths'):
        source_name = Path(dataset.fpaths[dataset_index]).stem
    else:
        source_name = str(dataset_index).zfill(5)
    fname = f'{source_name}.png'
    ref_img = dataset[dataset_index].unsqueeze(0).to(device)
    return ref_img, fname


def find_existing_run_indices(save_dir: Path, task_name: str, max_runs: int) -> dict[int, Path]:
    existing_runs: dict[int, Path] = {}
    pattern = re.compile(rf"^{re.escape(task_name)}_run(\d+)$")

    for child in save_dir.iterdir():
        if not child.is_dir():
            continue

        match = pattern.match(child.name)
        if match is None:
            continue

        run_index = int(match.group(1))
        if run_index < 1 or run_index > max_runs:
            continue

        metrics_csv = child / 'progress_metrics.csv'
        if metrics_csv.exists():
            existing_runs[run_index] = metrics_csv

    return existing_runs


def ensure_run_metrics(run_root: Path, device: torch.device) -> Path:
    metrics_csv = run_root / 'progress_metrics.csv'
    eta_csv = run_root / 'eta_per_step.csv'

    if metrics_csv.exists() and eta_csv.exists():
        return metrics_csv

    progress_root = run_root / 'progress'
    label_root = run_root / 'label'
    if not progress_root.exists() or not label_root.exists():
        return metrics_csv

    progress_df = collect_progress_metrics(
        progress_root=progress_root,
        label_root=label_root,
        eta_csv=eta_csv,
        device=device,
    )
    save_progress_metrics(progress_df, metrics_csv)
    return metrics_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--diffusion_config', type=str, required=True)
    parser.add_argument('--task_config', type=str, required=True)
    parser.add_argument('--policy_weights', type=str, required=True, help="Path to optimized_eta_policy.pth")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results/rl_inference')
    parser.add_argument('--num_runs', type=int, default=1, help='Run sampling multiple times and aggregate metrics')
    args = parser.parse_args()

    # logger
    logger = get_logger()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)

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
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")

    sampler = create_sampler(**diffusion_config)

    # 3. Setup Data
    data_config = task_config['data']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = get_dataset(**data_config, transforms=transform)

    # 4. Load the Trained RL Policy
    rl_config = task_config.get('rl_config', {})
    policy_net = EtaPolicyNetwork(
        hidden_dim=int(rl_config.get('hidden_dim', 128)),
        eta_max=float(rl_config.get('eta_max', 5.0)),
        std=float(rl_config.get('policy_std', 0.3))
    ).to(device)
    policy_net.load_state_dict(torch.load(args.policy_weights, map_location=device))
    policy_net.eval()  # Crucial: set to evaluation mode

    task_name = measure_config['operator']['name']
    save_dir = Path(args.save_dir)
    task_root = save_dir / task_name
    task_root.mkdir(parents=True, exist_ok=True)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    num_runs = max(args.num_runs, 1)

    def build_run_root(run_index: int) -> Path:
        if num_runs <= 1:
            return task_root
        return save_dir / f'{task_name}_run{run_index:02d}'

    def run_single(run_root: str) -> str:
        os.makedirs(run_root, exist_ok=True)
        for img_dir in ['input', 'recon', 'progress', 'label']:
            os.makedirs(os.path.join(run_root, img_dir), exist_ok=True)

        if num_runs > 1:
            run_indices = torch.randperm(len(dataset)).tolist()
        else:
            run_indices = list(range(len(dataset)))

        for i, dataset_index in enumerate(run_indices):
            operator = get_operator(device=device, **measure_config['operator'])
            cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
            measurement_cond_fn = cond_method.conditioning
            ref_img, fname = get_ref_and_fname(dataset, dataset_index, device)
            logger.info(f"Inference for image {i} (dataset index {dataset_index}) -> {fname} in {os.path.basename(run_root)}")

            sample_progress_root = os.path.join(run_root, 'progress', Path(fname).stem)
            os.makedirs(sample_progress_root, exist_ok=True)

            if measure_config['operator']['name'] == 'inpainting':
                mask = mask_gen(ref_img)
                mask = mask[:, 0, :, :].unsqueeze(dim=0)

                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)

                def current_measurement_cond_fn(**kwargs):
                    return cond_method.conditioning(mask=mask, **kwargs)
            else:
                y = operator.forward(ref_img)
                y_n = noiser(y)
                current_measurement_cond_fn = measurement_cond_fn

            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sampler.p_sample_loop(
                model=model,
                x_start=x_start,
                measurement=y_n,
                measurement_cond_fn=current_measurement_cond_fn,
                record=True,
                save_root=sample_progress_root,
                rl_mode=True,
                policy_net=policy_net,
                ref_img=ref_img,
                conditioning_method=cond_method,
                operator=operator,
            )

            plt.imsave(os.path.join(run_root, 'input', fname), clear_color(y_n))
            plt.imsave(os.path.join(run_root, 'label', fname), clear_color(ref_img))
            plt.imsave(os.path.join(run_root, 'recon', fname), clear_color(sample))

        metrics_csv = os.path.join(run_root, 'progress_metrics.csv')
        progress_df = collect_progress_metrics(
            progress_root=Path(os.path.join(run_root, 'progress')),
            label_root=Path(os.path.join(run_root, 'label')),
            eta_csv=Path(os.path.join(run_root, 'eta_per_step.csv')),
            device=device,
        )
        save_progress_metrics(progress_df, Path(metrics_csv))
        return metrics_csv

    run_metrics_csvs = []
    if num_runs > 1:
        existing_run_metrics = find_existing_run_indices(save_dir, task_name, num_runs)
    else:
        existing_run_metrics = {}

    for run_index in range(1, num_runs + 1):
        run_root = build_run_root(run_index)
        if run_index in existing_run_metrics:
            ensure_run_metrics(run_root, device)
            logger.info(f"Reusing completed run {run_index}/{num_runs}: {run_root}")
            run_metrics_csvs.append(str(existing_run_metrics[run_index]))
            continue

        logger.info(f"Starting run {run_index}/{num_runs}: {run_root}")
        run_metrics_csvs.append(run_single(str(run_root)))

    if num_runs > 1:
        aggregate_csv = task_root / 'progress_metrics.csv'
        aggregate_df = aggregate_progress_metrics(run_metrics_csvs, output_csv=aggregate_csv)
        logger.info(f"Saved aggregated progress metrics to {aggregate_csv} ({len(aggregate_df)} rows)")


if __name__ == '__main__':
    main()