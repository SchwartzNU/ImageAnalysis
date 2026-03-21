import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = [
    ("Top_Surface_Ratio", "Top Surface Ratio"),
    ("Bot_Surface_Ratio", "Bottom Surface Ratio"),
    ("Stain_Top_Surface_To_Stain_Middle_Ratio", "Stain Top Surface / Stain Middle"),
]

ROUNDNESS_COL = "Segmented_Cell_Roundness"
DIAMETER_COL = "Segmented_Cell_Equivalent_Diameter_um"


def resolve_inputs(input_path):
    if os.path.isdir(input_path):
        folder = input_path
        folder_name = os.path.basename(os.path.normpath(folder))
        csv_path = os.path.join(folder, f"{folder_name}_processed.csv")
        output_dir = os.path.join(folder, "postprocess_plots")
    else:
        csv_path = input_path
        folder = os.path.dirname(os.path.abspath(csv_path))
        output_dir = os.path.join(folder, "postprocess_plots")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)
    return csv_path, output_dir


def shorten_label(file_name):
    label = os.path.splitext(os.path.basename(str(file_name)))[0]
    return label.replace("_DAPI_WGA_594_", "_").replace("_647_", "_")


def apply_filters(df, min_roundness=None, min_diameter_um=None, max_diameter_um=None):
    filtered_df = df.copy()

    if min_roundness is not None and ROUNDNESS_COL in filtered_df.columns:
        roundness_vals = pd.to_numeric(filtered_df[ROUNDNESS_COL], errors="coerce")
        filtered_df = filtered_df.loc[roundness_vals >= min_roundness].copy()

    if min_diameter_um is not None and DIAMETER_COL in filtered_df.columns:
        diameter_vals = pd.to_numeric(filtered_df[DIAMETER_COL], errors="coerce")
        filtered_df = filtered_df.loc[diameter_vals >= min_diameter_um].copy()

    if max_diameter_um is not None and DIAMETER_COL in filtered_df.columns:
        diameter_vals = pd.to_numeric(filtered_df[DIAMETER_COL], errors="coerce")
        filtered_df = filtered_df.loc[diameter_vals <= max_diameter_um].copy()

    return filtered_df


def plot_metric(df, metric_col, metric_label, output_path):
    plot_df = df[["file_name", metric_col]].copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])
    if plot_df.empty:
        return None

    file_names = list(dict.fromkeys(plot_df["file_name"].tolist()))
    x_positions = np.arange(len(file_names))
    file_to_x = {name: idx for idx, name in enumerate(file_names)}

    grouped = plot_df.groupby("file_name")[metric_col]
    medians = grouped.median().reindex(file_names)
    stds = grouped.std(ddof=1).fillna(0.0).reindex(file_names)
    counts = grouped.size().reindex(file_names)

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(file_names)), 6))

    rng = np.random.default_rng(20260320)
    for file_name in file_names:
        vals = plot_df.loc[plot_df["file_name"] == file_name, metric_col].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(
            np.full(len(vals), file_to_x[file_name]) + jitter,
            vals,
            s=18,
            alpha=0.5,
            color="#2F6B7A",
            edgecolors="none",
            zorder=2,
        )

    ax.bar(
        x_positions,
        medians.to_numpy(),
        yerr=stds.to_numpy(),
        width=0.55,
        color="#D9E6F2",
        edgecolor="#4C72B0",
        ecolor="#1F3552",
        capsize=6,
        linewidth=1.5,
        alpha=0.45,
        zorder=3,
    )

    ax.set_title(f"{metric_label} by File")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("file_name")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([shorten_label(name) for name in file_names], rotation=18, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

    for x, median_val, n in zip(x_positions, medians.to_numpy(), counts.to_numpy()):
        ax.text(x, median_val, f"n={int(n)}", ha="center", va="bottom", fontsize=9, color="#1F3552")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {
        "metric": metric_col,
        "output_path": output_path,
        "num_files": len(file_names),
        "num_points": int(len(plot_df)),
    }


def main():
    parser = argparse.ArgumentParser(description="Create per-file bar/scatter plots from processed analysis output.")
    parser.add_argument("input_path", help="Folder containing *_processed.csv or a direct path to the processed CSV.")
    parser.add_argument("--min-roundness", type=float, default=None, help="Keep only cells with roundness >= this value.")
    parser.add_argument("--min-diameter-um", type=float, default=None, help="Keep only cells with equivalent diameter >= this value.")
    parser.add_argument("--max-diameter-um", type=float, default=None, help="Keep only cells with equivalent diameter <= this value.")
    args = parser.parse_args()

    csv_path, output_dir = resolve_inputs(args.input_path)
    df = pd.read_csv(csv_path)
    filtered_df = apply_filters(
        df,
        min_roundness=args.min_roundness,
        min_diameter_um=args.min_diameter_um,
        max_diameter_um=args.max_diameter_um,
    )

    print(f"Loaded {len(df)} rows from: {csv_path}")
    print(f"Rows after filtering: {len(filtered_df)}")
    if args.min_roundness is not None:
        print(f"Applied roundness filter: {ROUNDNESS_COL} >= {args.min_roundness}")
    if args.min_diameter_um is not None:
        print(f"Applied minimum diameter filter: {DIAMETER_COL} >= {args.min_diameter_um}")
    if args.max_diameter_um is not None:
        print(f"Applied maximum diameter filter: {DIAMETER_COL} <= {args.max_diameter_um}")

    summaries = []
    for metric_col, metric_label in METRICS:
        if metric_col not in filtered_df.columns:
            print(f"Skipping missing column: {metric_col}")
            continue
        output_path = os.path.join(output_dir, f"{metric_col}.png")
        summary = plot_metric(filtered_df, metric_col, metric_label, output_path)
        if summary is not None:
            summaries.append(summary)
            print(f"Saved plot: {output_path}")

    if not summaries:
        raise RuntimeError("No plots were generated. Expected at least one target metric column.")


if __name__ == "__main__":
    main()
