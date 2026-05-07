from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


METRICS = ["psnr", "ssim", "lpips"]


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

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["step"] + METRICS).sort_values("step")
    return df


def compute_diff_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(df1, df2, on="step", how="inner", suffixes=("_1", "_2"))
    if merged.empty:
        raise ValueError("No shared steps found between the two runs.")

    diff_df = pd.DataFrame({"step": merged["step"]})
    for metric in METRICS:
        diff_df[metric] = merged[f"{metric}_1"] - merged[f"{metric}_2"]

    return diff_df.sort_values("step")


def main():
    parser = argparse.ArgumentParser(description="Visualize progress metrics from CSV.")
    parser.add_argument("--task", type=str, default="super_resolution", help="Task folder under results")
    parser.add_argument("--csv", type=str, default=None, help="CSV path override for the first run")

    parser.add_argument("--task2", type=str, default=None, help="Second task folder under results")
    parser.add_argument("--csv2", type=str, default=None, help="CSV path override for the second run")

    parser.add_argument("--name1", type=str, default=None, help="Legend name for the first run")
    parser.add_argument("--name2", type=str, default=None, help="Legend name for the second run")

    parser.add_argument("--save_path", type=str, default=None, help="Path to save the full figure")
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

        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex="col")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)

    # Top row: original metrics
    top_axes = axes[0] if has_second else axes
    for ax, metric in zip(top_axes, METRICS):
        ax.plot(df1["step"], df1[metric], marker="o", linewidth=2, label=label1)

        if has_second:
            ax.plot(df2["step"], df2[metric], marker="o", linewidth=2, label=label2)

        ax.set_title(metric.upper())
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)

    top_axes[0].legend()

    # Bottom row: differences
    if has_second:
        for ax, metric in zip(axes[1], METRICS):
            ax.plot(
                diff_df["step"],
                diff_df[metric],
                marker="o",
                linewidth=2,
                label=f"{label1} - {label2}",
            )
            ax.axhline(0, linestyle="--", linewidth=1, alpha=0.7)
            ax.set_title(f"Δ {metric.upper()}")
            ax.set_xlabel("step")
            ax.set_ylabel(f"{metric} difference")
            ax.grid(True, alpha=0.3)

        axes[1, 0].legend()

        fig.suptitle("Progress Metrics vs Step and Difference Between Runs", fontsize=14)
    else:
        fig.suptitle("Progress Metrics vs Step", fontsize=14)

    fig.tight_layout()

    if args.save_path is not None:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if args.show or args.save_path is None:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()