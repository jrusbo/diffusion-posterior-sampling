from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["psnr", "ssim", "lpips", "eta"]
REQUIRED_METRICS = ["psnr", "ssim", "lpips"]


def resolve_csv(task: str, csv_path: str | None) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    return Path(f"./results/{task}/progress_metrics.csv")


def load_metrics(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = {"step", "psnr", "ssim", "lpips"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    if "eta" not in df.columns:
        df["eta"] = np.nan

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        std_col = f"{col}_std"
        if std_col in df.columns:
            df[std_col] = pd.to_numeric(df[std_col], errors="coerce")

    metric_cols = [col for col in METRICS if col in df.columns]
    df = df.dropna(subset=["step"] + REQUIRED_METRICS)

    raw_row_count = len(df)

    if df["step"].duplicated().any():
        grouped = df.groupby("step")
        mean_df = grouped[metric_cols].mean().reset_index()
        std_df = grouped[metric_cols].std(ddof=0).reset_index()
        std_df = std_df.rename(columns={col: f"{col}_std" for col in metric_cols})
        df = mean_df.merge(std_df, on="step", how="left")
        print(f"Averaged {csv_path.name} over repeated steps: {len(mean_df)} unique steps from {raw_row_count} raw rows")

    df = df.sort_values("step").reset_index(drop=True)
    return df


def compute_diff_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(df1, df2, on="step", how="inner", suffixes=("_1", "_2"))
    if merged.empty:
        raise ValueError("No shared steps found between the two runs.")

    diff_df = pd.DataFrame({"step": merged["step"]})
    for metric in METRICS:
        diff_df[metric] = merged[f"{metric}_1"] - merged[f"{metric}_2"]

    return diff_df.sort_values("step").reset_index(drop=True)

def set_xlim_inverted(ax, xmin: float, xmax: float, pad_ratio: float = 0.03) -> None:
    span = xmax - xmin
    pad = span * pad_ratio if span > 0 else 1
    ax.set_xlim(xmax + pad, xmin - pad)


def plot_metric_with_band(ax, df: pd.DataFrame, metric: str, label: str, color=None) -> None:
    std_col = f"{metric}_std"
    x_values = df["step"].to_numpy(dtype=float)
    y_values = df[metric].to_numpy(dtype=float)

    line = ax.plot(
        x_values,
        y_values,
        linewidth=1.4,
        marker="o",
        markersize=3,
        alpha=0.5,
        label=label,
        color=color,
        zorder=3,
    )[0]

    if std_col in df.columns and df[std_col].notna().any():
        std_values = df[std_col].fillna(0.0).to_numpy(dtype=float)
        lower = y_values - std_values
        upper = y_values + std_values
        band_color = line.get_color()
        ax.fill_between(
            x_values,
            lower,
            upper,
            alpha=0.1,
            color=band_color,
            edgecolor=band_color,
            linewidth=0.3,
            zorder=1,
        )
        ax.plot(x_values, lower, color=band_color, alpha=0.2, linewidth=0.8, zorder=2)
        ax.plot(x_values, upper, color=band_color, alpha=0.2, linewidth=0.8, zorder=2)


def main():
    parser = argparse.ArgumentParser(description="Visualize progress metrics from CSV.")
    parser.add_argument("--task", type=str, default="super_resolution", help="Task folder under results")
    parser.add_argument("--csv", type=str, default=None, help="CSV path override for the first run")
    parser.add_argument("--task2", type=str, default=None, help="Second task folder under results")
    parser.add_argument("--csv2", type=str, default=None, help="CSV path override for the second run")
    parser.add_argument("--name1", type=str, default=None, help="Legend name for the first run")
    parser.add_argument("--name2", type=str, default=None, help="Legend name for the second run")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    args = parser.parse_args()

    csv1 = resolve_csv(args.task, args.csv)
    df1 = load_metrics(csv1)
    label1 = args.name1 or (csv1.stem if args.csv else args.task)

    has_second = args.csv2 is not None or args.task2 is not None

    if has_second:
        if args.csv2 is not None:
            csv2 = Path(args.csv2)
        else:
            task2 = args.task2 if args.task2 is not None else args.task
            csv2 = Path(f"./results/{task2}/progress_metrics.csv")

        df2 = load_metrics(csv2)
        label2 = args.name2 or (csv2.stem if args.csv2 else (args.task2 or "run2"))
        diff_df = compute_diff_df(df1, df2)
        fig, axes = plt.subplots(2, len(METRICS), figsize=(5 * len(METRICS), 10), sharex="col")
        top_axes = axes[0]
        bottom_axes = axes[1]
    else:
        df2 = None
        diff_df = None
        label2 = None
        fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 5), sharex=True)
        top_axes = axes
        bottom_axes = None

    all_steps = [df1["step"].min(), df1["step"].max()]
    if has_second and df2 is not None:
        all_steps += [df2["step"].min(), df2["step"].max()]
    xmin, xmax = min(all_steps), max(all_steps)

    for ax, metric in zip(top_axes, METRICS):
        plot_metric_with_band(ax, df1, metric, label1)

        if has_second and df2 is not None:
            plot_metric_with_band(ax, df2, metric, label2)

        ax.set_title(f"{metric.upper()}")
        ax.set_xlabel("step")
        ax.set_ylabel(f"{metric}")
        ax.grid(True, alpha=0.3)
        set_xlim_inverted(ax, xmin, xmax)

    top_axes[0].legend()

    if has_second and diff_df is not None:
        for ax, metric in zip(bottom_axes, METRICS):
            ax.plot(
                diff_df["step"],
                diff_df[metric],
                linewidth=1,
                marker="o",
                markersize=3,
                alpha=0.55,
                label=f"{label1} - {label2}",
            )
            ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(f"Δ {metric.upper()}")
            ax.set_xlabel("step")
            ax.set_ylabel(f"{metric} difference")
            ax.grid(True, alpha=0.3)
            set_xlim_inverted(ax, xmin, xmax)

        bottom_axes[0].legend()
        fig.suptitle("Progress Metrics vs Step and Difference Between Runs", fontsize=14)
    else:
        fig.suptitle("Progress Metrics vs Step", fontsize=14)

    fig.tight_layout()

    save_path = csv1.parent / "metrics_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()