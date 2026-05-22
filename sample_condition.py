from functools import partial
import os
import argparse
from datetime import datetime
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--num_runs', type=int, default=1, help='Run sampling multiple times and aggregate metrics')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
   
    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    # Prepare Operator, noise, and conditioning
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 

    # SAFEGUARD: Ensure we are not using RL-specific conditioning in regular sampling
    if task_config['conditioning']['method'] == 'rl_ps':
        raise ValueError("Cannot use 'rl_ps' with sample_condition.py. Use sample_condition_rl.py instead.")

    task_name = measure_config['operator']['name']
    task_root = os.path.join(args.save_dir, task_name)
    os.makedirs(task_root, exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = get_dataset(**data_config, transforms=transform)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        
    num_runs = max(args.num_runs, 1)

    def build_run_root(run_index: int) -> str:
        if num_runs <= 1:
            return task_root
        return os.path.join(args.save_dir, f'{task_name}_run{run_index:02d}')

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

            current_measurement_cond_fn = measurement_cond_fn

            if measure_config['operator']['name'] == 'inpainting':
                mask = mask_gen(ref_img)
                mask = mask[:, 0, :, :].unsqueeze(dim=0)
                current_measurement_cond_fn = partial(cond_method.conditioning, mask=mask)

                y = operator.forward(ref_img, mask=mask)
                y_n = noiser(y)
            else:
                y = operator.forward(ref_img)
                y_n = noiser(y)

            x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
            sample = sampler.p_sample_loop(
                model=model,
                x_start=x_start,
                measurement=y_n,
                measurement_cond_fn=current_measurement_cond_fn,
                operator=operator,
                record=True,
                save_root=sample_progress_root,
                ref_img=ref_img,
                conditioning_method=cond_method,
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
    for run_index in range(1, num_runs + 1):
        run_root = build_run_root(run_index)
        logger.info(f"Starting run {run_index}/{num_runs}: {run_root}")
        run_metrics_csvs.append(run_single(run_root))

    if num_runs > 1:
        aggregate_csv = Path(task_root) / 'progress_metrics.csv'
        aggregate_df = aggregate_progress_metrics(run_metrics_csvs, output_csv=aggregate_csv)
        logger.info(f"Saved aggregated progress metrics to {aggregate_csv} ({len(aggregate_df)} rows)")

if __name__ == '__main__':
    main()
