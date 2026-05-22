from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Use script location as reference to find results directory
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
METRICS = ["psnr", "ssim", "lpips"]

# Only process these constant eta values
CONSTANT_ETAS = [0.01, 0.1, 0.3, 0.5, 1.0, 5.0, 10.0]


def parse_eta_from_folder(folder_name: str) -> float | None:
    """Extract eta value from motion_blur_* folder name. Return None if not a constant eta."""
    # Special case: baseline = eta 0.3
    if folder_name in ("motion_blur_baseline", "motion_blur_baseline_ddim"):
        return 0.3
    
    # Try simple format: motion_blur_X.XX
    match = re.match(r"motion_blur_([\d.]+)$", folder_name)
    if match:
        try:
            eta = float(match.group(1))
            # Only accept if it's in the constant eta list (with small tolerance for float comparison)
            for ce in CONSTANT_ETAS:
                if abs(eta - ce) < 1e-6:
                    return eta
        except ValueError:
            pass
    return None


def load_step0_metrics(csv_path: Path) -> dict:
    """Load step 0 metrics and std from CSV."""
    df = pd.read_csv(csv_path)

    required = {"step", "psnr", "ssim", "lpips"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for metric in METRICS:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        std_col = f"{metric}_std"
        if std_col in df.columns:
            df[std_col] = pd.to_numeric(df[std_col], errors="coerce")

    df = df.dropna(subset=["step"] + METRICS)

    step0 = df[df["step"] == 0]
    if step0.empty:
        raise ValueError(f"No step 0 found in {csv_path}")

    row = step0.iloc[0]
    result = {
        "psnr": float(row["psnr"]),
        "ssim": float(row["ssim"]),
        "lpips": float(row["lpips"]),
    }
    
    # Add std if available
    for metric in METRICS:
        std_col = f"{metric}_std"
        if std_col in row.index:
            result[f"{metric}_std"] = float(row[std_col]) if pd.notna(row[std_col]) else 0.0
        else:
            result[f"{metric}_std"] = 0.0
    
    return result


def main():
    rows = []

    for folder in sorted(RESULTS_DIR.glob("motion_blur_*")):
        if not folder.is_dir():
            continue

        # Try nested path first, then top-level
        csv_path = folder / "motion_blur" / "progress_metrics.csv"
        if not csv_path.exists():
            csv_path = folder / "progress_metrics.csv"
        if not csv_path.exists():
            continue

        eta = parse_eta_from_folder(folder.name)
        if eta is None:
            # Skip non-constant-eta folders
            continue

        metrics = load_step0_metrics(csv_path)

        rows.append(
            {
                "eta": eta,
                "psnr": metrics["psnr"],
                "psnr_std": metrics.get("psnr_std", 0.0),
                "ssim": metrics["ssim"],
                "ssim_std": metrics.get("ssim_std", 0.0),
                "lpips": metrics["lpips"],
                "lpips_std": metrics.get("lpips_std", 0.0),
                "folder": folder.name,
            }
        )

    if not rows:
        raise RuntimeError("No valid constant-eta motion_blur_* runs found.")

    df = pd.DataFrame(rows).sort_values("eta").reset_index(drop=True)
    
    # If there are multiple runs per eta (e.g., baseline and baseline_ddim both = 0.3),
    # aggregate them by taking mean of means and mean of stds
    eta_groups = df.groupby("eta", as_index=False)
    aggregated_rows = []
    for eta_val, group in eta_groups:
        aggregated_rows.append({
            "eta": eta_val,
            "psnr": group["psnr"].mean(),
            "psnr_std": group["psnr_std"].mean(),
            "ssim": group["ssim"].mean(),
            "ssim_std": group["ssim_std"].mean(),
            "lpips": group["lpips"].mean(),
            "lpips_std": group["lpips_std"].mean(),
        })
    
    df = pd.DataFrame(aggregated_rows).sort_values("eta").reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    baseline_eta = 0.3

    for ax, metric in zip(axes, METRICS):
        std_col = f"{metric}_std"
        
        # Main curve
        ax.plot(df["eta"], df[metric], marker="o", linewidth=1.5, label="Mean")

        # Add std band
        lower = df[metric] - df[std_col]
        upper = df[metric] + df[std_col]
        ax.fill_between(
            df["eta"],
            lower,
            upper,
            alpha=0.2,
            label="± Std"
        )

        # Set x-axis ticks to all eta values
        ax.set_xticks(df["eta"])
        ax.set_xticklabels([str(x) for x in df["eta"]], rotation=45)

        # Highlight baseline
        baseline_row = df[abs(df["eta"] - baseline_eta) < 1e-9]
        if not baseline_row.empty:
            x = baseline_row["eta"].iloc[0]
            y = baseline_row[metric].iloc[0]
            ax.scatter(
                x,
                y,
                s=150,
                zorder=5,
                color="red",
                marker="*",
                label=f"Baseline (η={baseline_eta})",
            )

        ax.set_xlabel("η (eta)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} vs η (step 0)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.tight_layout()

    save_path = RESULTS_DIR / "metrics_step0_vs_eta.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    print(f"Saved figure to: {save_path}")
    print("\nMetrics summary:")
    print(df[["eta", "psnr", "psnr_std", "ssim", "ssim_std", "lpips", "lpips_std"]].to_string(index=False))

    plt.show()


if __name__ == "__main__":
    main()