from pathlib import Path
import argparse
import csv
import re
import sys

import matplotlib.pyplot as plt
import lpips
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def normalize_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]

    if np.issubdtype(img.dtype, np.integer):
        max_val = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / float(max_val)
    elif np.issubdtype(img.dtype, np.floating):
        img = img.astype(np.float32)
        if img.max() > 1.1:
            img = img / 255.0

    img = np.clip(img, 0.0, 1.0)
    return img


def extract_numeric_step(path: Path) -> int:
    match = re.search(r'(\d+)', path.stem)
    if match:
        return int(match.group(1))
    return -1


def find_label_for_progress(progress_path: Path, label_root: Path, progress_root: Path):
    if not label_root.exists():
        return None

    if progress_path.parent != progress_root:
        sample_name = progress_path.parent.name
        for ext in IMAGE_EXTENSIONS:
            candidate = label_root / f"{sample_name}{ext}"
            if candidate.exists():
                return candidate
        nested_candidate = label_root / progress_path.parent.name / progress_path.name
        if nested_candidate.exists():
            return nested_candidate

    label_images = sorted(p for p in label_root.rglob('*') if is_image_file(p))
    if len(label_images) == 1:
        return label_images[0]

    numeric = extract_numeric_step(progress_path)
    if numeric >= 0:
        for ext in IMAGE_EXTENSIONS:
            candidate = label_root / f"{str(numeric).zfill(5)}{ext}"
            if candidate.exists():
                return candidate
            candidate = label_root / f"{numeric}{ext}"
            if candidate.exists():
                return candidate

    return None


def resolve_fixed_label(label_root: Path, label_name: str) -> Path:
    p = Path(label_name)

    # Case: user provided full filename (e.g., 00003.png)
    if p.suffix:
        candidate = label_root / p.name
        if candidate.exists():
            return candidate
    else:
        # Try with extensions
        for ext in IMAGE_EXTENSIONS:
            candidate = label_root / f"{label_name}{ext}"
            if candidate.exists():
                return candidate

    # Fallback: recursive search
    matches = list(label_root.rglob(f"{label_name}.*"))
    matches = [m for m in matches if is_image_file(m)]
    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(f"Fixed label not found: {label_name} under {label_root}")


def read_image(path: Path) -> np.ndarray:
    img = plt.imread(path)
    return normalize_image(img)


def compute_metrics(label: np.ndarray, recon: np.ndarray):
    if label.shape != recon.shape:
        raise ValueError(f"Shape mismatch: label={label.shape}, recon={recon.shape}")

    psnr = peak_signal_noise_ratio(label, recon, data_range=1.0)
    ssim = structural_similarity(label, recon, channel_axis=-1, win_size=3, data_range=1.0)

    recon_t = torch.from_numpy(recon).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    label_t = torch.from_numpy(label).permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    return psnr, ssim, recon_t, label_t


def collect_progress_metrics(
    progress_root: Path,
    label_root: Path,
    eta_csv: Path | None = None,
    label: str | None = None,
    device: str = 'cuda',
    verbose: bool = False,
) -> pd.DataFrame:
    fixed_label_path = resolve_fixed_label(label_root, label) if label is not None else None
    loss_fn = lpips.LPIPS(net='vgg').to(device)

    progress_images = sorted(
        [p for p in progress_root.rglob('*') if p.is_file() and is_image_file(p)],
        key=lambda p: (p.parent.name, extract_numeric_step(p), p.name)
    )

    if len(progress_images) == 0:
        raise FileNotFoundError(f'No progress images found in: {progress_root}')

    rows = []
    for progress_path in progress_images:
        label_path = fixed_label_path or find_label_for_progress(progress_path, label_root, progress_root)

        if label_path is None:
            if verbose:
                print(f'Skipping {progress_path}: no matching label found', file=sys.stderr)
            continue

        try:
            recon = read_image(progress_path)
            label_img = read_image(label_path)

            if recon.shape != label_img.shape:
                if verbose:
                    print(
                        f'Skipping {progress_path}: shape mismatch {recon.shape} vs {label_img.shape}',
                        file=sys.stderr,
                    )
                continue

            psnr, ssim, recon_t, label_t = compute_metrics(label_img, recon)
            recon_t = recon_t.to(device)
            label_t = label_t.to(device)
            lpips_value = loss_fn(recon_t, label_t).item()

            step = extract_numeric_step(progress_path)
            sample = progress_path.parent.name if progress_path.parent != progress_root else ''

            rows.append({
                'progress_image': str(progress_path.relative_to(progress_root)),
                'label_image': str(label_path.relative_to(label_root)),
                'step': step,
                'sample': sample,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips_value,
            })

            if verbose:
                print(f'[{progress_path.name}] step={step} psnr={psnr:.4f} ssim={ssim:.4f} lpips={lpips_value:.6f}')

        except Exception as exc:
            print(f'Error processing {progress_path}: {exc}', file=sys.stderr)

    if len(rows) == 0:
        raise RuntimeError('No metric rows were generated.')

    rows_df = pd.DataFrame(rows)
    eta_path = eta_csv if eta_csv is not None else progress_root.parent / 'eta_per_step.csv'
    eta_df = load_or_aggregate_eta_csv(eta_path, progress_root)
    if eta_df is not None:
        rows_df['step'] = pd.to_numeric(rows_df['step'], errors='coerce')
        rows_df = rows_df.merge(eta_df[['step', 'eta']], on='step', how='left')

    return rows_df


def save_progress_metrics(rows_df: pd.DataFrame, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(rows_df.columns))
        writer.writeheader()
        writer.writerows(rows_df.to_dict(orient='records'))

def load_or_aggregate_eta_csv(eta_path: Path, progress_root: Path) -> pd.DataFrame | None:
    eta_sources = [eta_path] if eta_path.exists() else []
    if not eta_sources:
        eta_sources = sorted(
            path for path in progress_root.rglob('eta_per_step.csv')
            if path.is_file()
        )

    if len(eta_sources) == 0:
        return None

    eta_frames = []
    for source_path in eta_sources:
        eta_df = pd.read_csv(source_path)
        required_eta = {'step', 'eta'}
        if not required_eta.issubset(set(eta_df.columns)):
            continue

        eta_df = eta_df.copy()
        eta_df['step'] = pd.to_numeric(eta_df['step'], errors='coerce')
        eta_df['eta'] = pd.to_numeric(eta_df['eta'], errors='coerce')
        eta_df = eta_df.dropna(subset=['step', 'eta'])
        if len(eta_df) > 0:
            eta_frames.append(eta_df[['step', 'eta']])

    if len(eta_frames) == 0:
        return None

    combined_eta = pd.concat(eta_frames, ignore_index=True)
    combined_eta = (
        combined_eta
        .groupby('step', as_index=False)
        .mean()
        .sort_values('step')
        .reset_index(drop=True)
    )

    if not eta_path.exists():
        eta_path.parent.mkdir(parents=True, exist_ok=True)
        combined_eta.to_csv(eta_path, index=False)

    return combined_eta

def aggregate_progress_metrics(csv_paths, output_csv: Path | None = None) -> pd.DataFrame:
    csv_paths = [Path(path) for path in csv_paths]
    if len(csv_paths) == 0:
        raise ValueError('No CSV paths provided for aggregation.')

    dataframes = [pd.read_csv(path, keep_default_na=False) for path in csv_paths]
    combined = pd.concat(dataframes, ignore_index=True)

    key_candidates = ['progress_image', 'label_image', 'sample', 'step']
    group_keys = [column for column in key_candidates if all(column in df.columns for df in dataframes)]
    if len(group_keys) == 0:
        raise ValueError('No common key columns found for aggregation.')

    for column in group_keys:
        combined[column] = combined[column].fillna('')

    numeric_columns = []
    for column in combined.columns:
        if column in group_keys:
            continue
        combined[column] = pd.to_numeric(combined[column], errors='coerce')
        if combined[column].notna().any():
            numeric_columns.append(column)

    grouped = combined.groupby(group_keys, as_index=False)
    aggregated_mean = grouped[numeric_columns].mean()
    aggregated_std = grouped[numeric_columns].std(ddof=0)

    aggregated = aggregated_mean.copy()
    for column in numeric_columns:
        aggregated[f'{column}_std'] = aggregated_std[column].fillna(0.0)

    if output_csv is not None:
        save_progress_metrics(aggregated, output_csv)

    return aggregated


def main():
    parser = argparse.ArgumentParser(description='Compute metrics on progress images and save results to CSV.')
    parser.add_argument('--task', type=str, default='gaussian_deblur', help='Task folder under results')
    parser.add_argument('--progress_root', type=str, default=None, help='Progress image root folder')
    parser.add_argument('--label_root', type=str, default=None, help='Label image root folder')
    parser.add_argument('--label', type=str, default=None, help='Fixed label image name or stem (e.g. 00003)')
    parser.add_argument('--eta_csv', type=str, default=None, help='Eta CSV path override')
    parser.add_argument('--output_csv', type=str, default=None, help='CSV output path')
    parser.add_argument('--device', type=str, default='cuda', help='Device for LPIPS computation')
    parser.add_argument('--verbose', action='store_true', help='Print per-image status')
    args = parser.parse_args()

    task = args.task
    progress_root = Path(args.progress_root) if args.progress_root else Path(f'./results/{task}/progress')
    label_root = Path(args.label_root) if args.label_root else Path(f'./results/{task}/label')
    output_csv = Path(args.output_csv) if args.output_csv else Path(f'./results/{task}/progress_metrics.csv')

    if not progress_root.exists():
        print(f'Error: progress_root does not exist: {progress_root}', file=sys.stderr)
        sys.exit(1)
    if not label_root.exists():
        print(f'Error: label_root does not exist: {label_root}', file=sys.stderr)
        sys.exit(1)

    # Resolve fixed label if provided
    fixed_label_path = None
    if args.label is not None:
        fixed_label_path = resolve_fixed_label(label_root, args.label)
        print(f'Using fixed label: {fixed_label_path}')

    eta_csv = Path(args.eta_csv) if args.eta_csv else progress_root.parent / 'eta_per_step.csv'

    try:
        rows_df = collect_progress_metrics(
            progress_root=progress_root,
            label_root=label_root,
            eta_csv=eta_csv,
            label=args.label,
            device=args.device,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    save_progress_metrics(rows_df, output_csv)

    print(f'Saved metrics for {len(rows_df)} progress images to: {output_csv}')


if __name__ == '__main__':
            eta_path = eta_csv if eta_csv is not None else progress_root.parent / 'eta_per_step.csv'
            eta_df = load_or_aggregate_eta_csv(eta_path, progress_root)
            if eta_df is not None:
                rows_df['step'] = pd.to_numeric(rows_df['step'], errors='coerce')
                rows_df = rows_df.merge(eta_df[['step', 'eta']], on='step', how='left')