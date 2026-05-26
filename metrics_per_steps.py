from pathlib import Path
import re
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
METRICS = ["psnr", "ssim", "lpips"]
X_SHIFT = 10.0


def find_step_folders(results_dir: Path):
    """Find folders named motion_blur_baseline_{N} and motion_blur_rl_{N}.
    Return dicts mapping N->Path for baseline and rl.
    """
    baseline_re = re.compile(r"motion_blur_baseline_(\d+)$")
    rl_re = re.compile(r"motion_blur_rl_(\d+)$")

    baseline = {}
    rl = {}

    for p in results_dir.glob("motion_blur_*"):
        if not p.is_dir():
            continue
        m = baseline_re.match(p.name)
        if m:
            baseline[int(m.group(1))] = p
            continue
        m = rl_re.match(p.name)
        if m:
            rl[int(m.group(1))] = p

    return baseline, rl


def load_and_aggregate(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if 'step' not in df.columns:
        raise ValueError(f"missing 'step' in {csv_path}")

    df = df.copy()
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    for m in METRICS + ['eta']:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors='coerce')
        std_col = f"{m}_std"
        if std_col in df.columns:
            df[std_col] = pd.to_numeric(df[std_col], errors='coerce')

    df = df.dropna(subset=['step'] + METRICS)

    if df['step'].duplicated().any():
        metric_cols = [c for c in METRICS + ['eta'] if c in df.columns]
        mean_df = df.groupby('step')[metric_cols].mean().reset_index()
        std_df = df.groupby('step')[metric_cols].std(ddof=0).reset_index()
        std_df = std_df.rename(columns={col: f"{col}_std" for col in metric_cols})
        df = mean_df.merge(std_df, on='step', how='left')

    df = df.sort_values('step').reset_index(drop=True)
    return df


def set_xlim(ax, xmin: float, xmax: float, pad_ratio: float = 0.03) -> None:
    span = xmax - xmin
    pad = span * pad_ratio if span > 0 else 1
    ax.set_xlim(xmin - pad, xmax + pad)


def plot_metric(ax, df, metric, label, color):
    std_col = f"{metric}_std"
    max_step = float(df['step'].max())
    x = (max_step - df['step'].to_numpy(dtype=float)) + X_SHIFT
    order = np.argsort(x)
    x = x[order]
    y = df[metric].to_numpy(dtype=float)
    y = y[order]
    ax.plot(x, y, label=label, color=color, linewidth=1.4, marker='o', markersize=3, alpha=0.5)
    if std_col in df.columns and df[std_col].notna().any():
        s = df[std_col].fillna(0.0).to_numpy(dtype=float)[order]
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.1)


def annotate_endpoint(ax, df: pd.DataFrame, metric: str, color: str) -> None:
    max_step = float(df["step"].max())
    row = df.loc[df["step"] == max_step]
    if row.empty:
        row = df.iloc[[-1]]

    x = X_SHIFT
    y = float(row[metric].iloc[0])
    ax.scatter([x], [y], color=color, s=18, alpha=0.5, zorder=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=None, help='results dir override')
    parser.add_argument('--save', type=str, default=None, help='save path for figure')
    parser.add_argument('--show', action='store_true', default=True)
    parser.add_argument('--no-show', action='store_false', dest='show')
    args = parser.parse_args()

    results_dir = Path(args.results) if args.results else RESULTS_DIR
    baseline_map, rl_map = find_step_folders(results_dir)

    if not baseline_map and not rl_map:
        raise RuntimeError('No motion_blur_baseline_* or motion_blur_rl_* folders found in results')

    all_steps_counts = sorted(set(list(baseline_map.keys()) + list(rl_map.keys())))

    # Collect dataframes
    data = {}
    for n in all_steps_counts:
        d = {}
        if n in baseline_map:
            p = baseline_map[n]
            csv = p / 'progress_metrics.csv'
            if not csv.exists():
                csv = p / 'motion_blur' / 'progress_metrics.csv'
            d['baseline'] = load_and_aggregate(csv)
        if n in rl_map:
            p = rl_map[n]
            csv = p / 'progress_metrics.csv'
            if not csv.exists():
                csv = p / 'motion_blur' / 'progress_metrics.csv'
            d['rl'] = load_and_aggregate(csv)
        data[n] = d

    # Determine global xmin/xmax across all available dfs
    xmin = 0.0
    xmax = max((float(df['step'].max()) for d in data.values() for df in d.values()))

    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 3 * len(METRICS)), sharex=True)

    colors = {'baseline': 'orange', 'rl': 'blue'}
    legend_used = {'baseline': False, 'rl': False}

    for ax, metric in zip(axes, METRICS):
        endpoint_annotated = {'baseline': False, 'rl': False}
        for n in all_steps_counts:
            d = data[n]
            if 'baseline' in d:
                label = 'baseline' if not legend_used['baseline'] else None
                plot_metric(ax, d['baseline'], metric, label, colors['baseline'])
                legend_used['baseline'] = True
                if n == max(baseline_map.keys()) and not endpoint_annotated['baseline']:
                    annotate_endpoint(ax, d['baseline'], metric, colors['baseline'])
                    endpoint_annotated['baseline'] = True
            if 'rl' in d:
                label = 'rl' if not legend_used['rl'] else None
                plot_metric(ax, d['rl'], metric, label, colors['rl'])
                legend_used['rl'] = True
                if n == max(rl_map.keys()) and not endpoint_annotated['rl']:
                    annotate_endpoint(ax, d['rl'], metric, colors['rl'])
                    endpoint_annotated['rl'] = True

        ax.set_title(metric.upper())
        ax.set_xlabel('steps taken')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        set_xlim(ax, xmin + X_SHIFT, xmax + X_SHIFT)

    axes[0].legend(loc='best')

    fig.tight_layout()

    save_path = Path(args.save) if args.save else results_dir / 'metrics_vs_steps.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
