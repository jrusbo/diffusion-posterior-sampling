from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import argparse

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--task', type=str, default='gaussian_deblur')
    parser.add_argument('--label_root', type=str, default=None)
    parser.add_argument('--recon_root', type=str, default=None)
    args = parser.parse_args()

    device = args.device
    task = args.task
    label_root = Path(args.label_root) if args.label_root else Path(f'./results/{task}/label')
    recon_root = Path(args.recon_root) if args.recon_root else Path(f'./results/{task}/recon')

    loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

    psnr_list = []
    ssim_list = []
    lpips_list = []

    n_images = len(list(label_root.glob('*.png')))

    assert n_images > 0, "empty file"

    for idx in tqdm(range(n_images)):
        fname = str(idx).zfill(5)

        label = plt.imread(label_root / f'{fname}.png')[:, :, :3]
        recon = plt.imread(recon_root / f'{fname}.png')[:, :, :3]

        h, w = label.shape[:2]

        # PSNR
        psnr = peak_signal_noise_ratio(label, recon)
        psnr_list.append(psnr)

        # SSIM
        ssim = structural_similarity(label, recon, channel_axis=-1, win_size=3, data_range=1.0)
        ssim_list.append(ssim)

        # LPIPS
        recon_t = torch.from_numpy(recon).permute(2, 0, 1).to(device)
        label_t = torch.from_numpy(label).permute(2, 0, 1).to(device)

        recon_t = recon_t.view(1, 3, h, w) * 2. - 1.
        label_t = label_t.view(1, 3, h, w) * 2. - 1.

        lpips_d = loss_fn_vgg(recon_t, label_t)
        lpips_list.append(lpips_d)

    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    lpips_avg = sum(lpips_list) / len(lpips_list)

    print(f'PSNR: {psnr_avg}')
    print(f'SSIM: {ssim_avg}')
    print(f'LPIPS: {lpips_avg}')


if __name__ == '__main__':
    main()