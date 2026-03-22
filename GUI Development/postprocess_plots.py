import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

METRICS = [
    ("Top_Surface_Ratio", "Top Surface Ratio"),
    ("Stain_Top_Surface_To_Stain_Middle_Ratio", "Stain Top Surface / Stain Middle"),
]

ROUNDNESS_COL = "Segmented_Cell_Roundness"
DIAMETER_COL = "Segmented_Cell_Equivalent_Diameter_um"
GFP_COL = "eGFP_Value"
GFP_LABEL_COL = "GFP_Label"
GFP_GROUP_COLORS = {"GFP-Negative": "#4C78A8", "GFP-Positive": "#E45756"}

LEGACY_PLOT_FILENAMES = [
    "Stain_Mean_Intensity.png",
    "Bot_Surface_Ratio.png",
    "Bot_Surface_Ratio_by_soma_quartile.png",
    "Bot_Surface_Ratio_by_soma_size_class.png",
    "Top_Surface_Ratio_by_soma_quartile.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_by_soma_quartile.png",
    "Top_Surface_Ratio_by_soma_size_class.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_by_soma_size_class.png",
    "Top_Surface_Ratio_gfp_negative.png",
    "Top_Surface_Ratio_gfp_positive.png",
    "Top_Surface_Ratio_by_soma_size_class_gfp_negative.png",
    "Top_Surface_Ratio_by_soma_size_class_gfp_positive.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_gfp_negative.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_gfp_positive.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_by_soma_size_class_gfp_negative.png",
    "Stain_Top_Surface_To_Stain_Middle_Ratio_by_soma_size_class_gfp_positive.png",
]


def resolve_inputs(input_path):
    candidate_path = input_path
    if not os.path.isabs(candidate_path):
        local_candidate = os.path.join(PROJECT_ROOT, candidate_path)
        data_candidate = os.path.join(DEFAULT_DATA_DIR, candidate_path)
        if os.path.isdir(data_candidate) or os.path.isfile(data_candidate):
            candidate_path = data_candidate
        elif os.path.isdir(local_candidate) or os.path.isfile(local_candidate):
            candidate_path = local_candidate

    if os.path.isdir(candidate_path):
        folder = candidate_path
        folder_name = os.path.basename(os.path.normpath(folder))
        csv_path = os.path.join(folder, f"{folder_name}_processed.csv")
        output_dir = os.path.join(folder, "postprocess_plots")
    else:
        csv_path = candidate_path
        folder = os.path.dirname(os.path.abspath(csv_path))
        output_dir = os.path.join(folder, "postprocess_plots")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Processed CSV not found: {csv_path}")

    os.makedirs(output_dir, exist_ok=True)
    return csv_path, output_dir


def cleanup_legacy_plot_files(output_dir):
    removed = []
    skipped = []
    for name in LEGACY_PLOT_FILENAMES:
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                skipped.append(path)
    return removed, skipped


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


def prepare_metric_df(df, metric_col):
    cols = ["file_name", metric_col]
    if GFP_COL in df.columns:
        cols.append(GFP_COL)
    plot_df = df[cols].copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])
    return plot_df


def normalize_gfp_series(series):
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    def normalize_value(value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        return mapping.get(text, None)

    return series.apply(normalize_value)


def add_gfp_labels(plot_df):
    out = plot_df.copy()
    if GFP_COL not in out.columns:
        out[GFP_LABEL_COL] = None
        return out, []

    gfp_vals = normalize_gfp_series(out[GFP_COL])
    out[GFP_LABEL_COL] = gfp_vals.map({False: "GFP-Negative", True: "GFP-Positive"})
    groups_present = [label for label in ["GFP-Negative", "GFP-Positive"] if (out[GFP_LABEL_COL] == label).any()]
    return out, groups_present


def plot_metric(df, metric_col, metric_label, output_path):
    plot_df = prepare_metric_df(df, metric_col)
    if plot_df.empty:
        return None

    plot_df, gfp_groups = add_gfp_labels(plot_df)

    file_names = list(dict.fromkeys(plot_df["file_name"].tolist()))
    x_positions = np.arange(len(file_names))
    file_to_x = {name: idx for idx, name in enumerate(file_names)}

    fig, ax = plt.subplots(figsize=(max(8, 2.2 * len(file_names)), 6))

    rng = np.random.default_rng(20260320)
    if len(gfp_groups) >= 2:
        bar_width = 0.24
        offsets = {"GFP-Negative": -(bar_width / 2 + 0.02), "GFP-Positive": (bar_width / 2 + 0.02)}
        counts_by_group = {group: [] for group in gfp_groups}
        for group in gfp_groups:
            group_df = plot_df.loc[plot_df[GFP_LABEL_COL] == group]
            grouped = group_df.groupby("file_name")[metric_col]
            medians = grouped.median().reindex(file_names)
            stds = grouped.std(ddof=1).fillna(0.0).reindex(file_names)
            counts = grouped.size().reindex(file_names).fillna(0).astype(int)
            counts_by_group[group] = counts.to_numpy()

            for file_name in file_names:
                vals = group_df.loc[group_df["file_name"] == file_name, metric_col].to_numpy()
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-bar_width * 0.22, bar_width * 0.22, size=len(vals))
                ax.scatter(
                    np.full(len(vals), file_to_x[file_name] + offsets[group]) + jitter,
                    vals,
                    s=16,
                    alpha=0.5,
                    color=GFP_GROUP_COLORS[group],
                    edgecolors="none",
                    zorder=2,
                )

            mask = medians.notna().to_numpy()
            centers = x_positions + offsets[group]
            ax.bar(
                centers[mask],
                medians.to_numpy()[mask],
                yerr=stds.to_numpy()[mask],
                width=bar_width,
                color=GFP_GROUP_COLORS[group],
                edgecolor=GFP_GROUP_COLORS[group],
                ecolor=GFP_GROUP_COLORS[group],
                capsize=5,
                linewidth=1.4,
                alpha=0.32,
                zorder=3,
                label=group,
            )

            for center, median_val, n in zip(centers, medians.to_numpy(), counts.to_numpy()):
                if pd.isna(median_val) or n <= 0:
                    continue
                ax.text(center, median_val, f"n={int(n)}", ha="center", va="bottom", fontsize=8, color="#22313F")

        ax.legend(frameon=False, title="GFP")
        ax.set_title(f"{metric_label} by File and GFP Status")
    else:
        grouped = plot_df.groupby("file_name")[metric_col]
        medians = grouped.median().reindex(file_names)
        stds = grouped.std(ddof=1).fillna(0.0).reindex(file_names)
        counts = grouped.size().reindex(file_names)

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

        for x, median_val, n in zip(x_positions, medians.to_numpy(), counts.to_numpy()):
            ax.text(x, median_val, f"n={int(n)}", ha="center", va="bottom", fontsize=9, color="#1F3552")

        ax.set_title(f"{metric_label} by File")

    ax.set_ylabel(metric_label)
    ax.set_xlabel("file_name")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([shorten_label(name) for name in file_names], rotation=18, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return {
        "metric": metric_col,
        "output_path": output_path,
        "num_files": len(file_names),
        "num_points": int(len(plot_df)),
    }


def generate_plots(filtered_df, output_dir):
    summaries = []
    for metric_col, metric_label in METRICS:
        if metric_col not in filtered_df.columns:
            continue

        output_path = os.path.join(output_dir, f"{metric_col}.png")
        summary = plot_metric(filtered_df, metric_col, metric_label, output_path)
        if summary is not None:
            summaries.append(summary)
            print(f"Saved plot: {output_path}")
    return summaries


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

    removed, skipped = cleanup_legacy_plot_files(output_dir)
    for path in removed:
        print(f"Removed legacy plot: {path}")
    for path in skipped:
        print(f"Skipped locked legacy plot: {path}")

    summaries = generate_plots(filtered_df, output_dir)

    if not summaries:
        raise RuntimeError("No plots were generated. Expected at least one target metric column.")


if __name__ == "__main__":
    main()
