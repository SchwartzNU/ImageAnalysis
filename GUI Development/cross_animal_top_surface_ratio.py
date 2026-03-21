import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_METRIC = "Top_Surface_Ratio"
TARGET_STAINS = ["GLUT1", "GLUT3"]
FOLDER_PREFIXES = ("40", "42")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "Data"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "Results"


def discover_animal_folders(root):
    return sorted(
        path for path in root.iterdir()
        if path.is_dir()
        and path.name.startswith(FOLDER_PREFIXES)
        and path.name[2:].isdigit()
    )


def load_metadata_map(folder):
    metadata_path = folder / f"{folder.name}_metadata.csv"
    if not metadata_path.exists():
        return None

    metadata_df = pd.read_csv(metadata_path, usecols=["filename", "treatment", "stain"]).copy()
    metadata_df = metadata_df.rename(
        columns={
            "filename": "file_name",
            "treatment": "Metadata_Treatment",
            "stain": "Metadata_Stain",
        }
    )
    return metadata_df


def load_combined_data(root):
    rows = []
    for folder in discover_animal_folders(root):
        csv_path = folder / f"{folder.name}_processed.csv"
        if not csv_path.exists():
            continue

        df = pd.read_csv(
            csv_path,
            usecols=["file_name", "Stain", "Experimental_Condition", TARGET_METRIC],
        ).copy()
        metadata_df = load_metadata_map(folder)
        if metadata_df is not None:
            df = df.merge(metadata_df, how="left", on="file_name")
            df["Stain"] = df["Metadata_Stain"].fillna(df["Stain"])
            df["Experimental_Condition"] = df["Metadata_Treatment"].fillna(df["Experimental_Condition"])

        df["animal_id"] = folder.name
        df["Light_Dark_Condition"] = df["Experimental_Condition"].map({"dark": "dark", "open": "light"})
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No processed CSV files were found in 40xx/42xx animal folders.")

    combined = pd.concat(rows, ignore_index=True)
    combined[TARGET_METRIC] = pd.to_numeric(combined[TARGET_METRIC], errors="coerce")
    combined = combined.dropna(subset=[TARGET_METRIC, "Stain"])
    return combined


def summarize_by_animal(combined_df, condition_column):
    summary = (
        combined_df.dropna(subset=[condition_column])
        .groupby(["animal_id", "Stain", condition_column], as_index=False)
        .agg(
            Animal_Top_Surface_Ratio_Median=(TARGET_METRIC, "median"),
            Animal_Top_Surface_Ratio_Mean=(TARGET_METRIC, "mean"),
            Cell_Count=(TARGET_METRIC, "size"),
            File_Count=("file_name", "nunique"),
        )
    )
    return summary


def plot_condition_comparison(summary_df, stain, condition_column, condition_order, title, output_path):
    stain_df = summary_df.loc[
        (summary_df["Stain"] == stain) & (summary_df[condition_column].isin(condition_order))
    ].copy()
    if stain_df.empty:
        return None

    colors = ["#4C78A8", "#E45756"]
    x_positions = np.arange(len(condition_order))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    rng = np.random.default_rng(20260321)

    for idx, condition in enumerate(condition_order):
        cond_df = stain_df.loc[stain_df[condition_column] == condition]
        if cond_df.empty:
            continue

        values = cond_df["Animal_Top_Surface_Ratio_Median"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), x_positions[idx]) + jitter,
            values,
            s=40,
            alpha=0.75,
            color=colors[idx],
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )

        median_val = float(np.median(values))
        std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ax.bar(
            [x_positions[idx]],
            [median_val],
            yerr=[std_val],
            width=0.5,
            color=colors[idx],
            edgecolor=colors[idx],
            ecolor=colors[idx],
            capsize=5,
            alpha=0.30,
            linewidth=1.4,
            zorder=2,
        )
        ax.text(
            x_positions[idx],
            median_val,
            f"n={len(values)}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#22313F",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_order)
    ax.set_ylabel("Animal Median Top Surface Ratio")
    ax.set_title(f"{title}: {stain}")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def plot_two_panel_comparison(summary_df, condition_column, condition_order, title_prefix, output_path):
    colors = ["#4C78A8", "#E45756"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    rng = np.random.default_rng(20260321)
    any_data = False

    for ax, stain in zip(axes, TARGET_STAINS):
        stain_df = summary_df.loc[
            (summary_df["Stain"] == stain) & (summary_df[condition_column].isin(condition_order))
        ].copy()
        x_positions = np.arange(len(condition_order))

        # Connect matched animal summaries across the two conditions when both exist.
        paired = stain_df.pivot_table(
            index="animal_id",
            columns=condition_column,
            values="Animal_Top_Surface_Ratio_Median",
            aggfunc="first",
        )
        if all(cond in paired.columns for cond in condition_order):
            paired = paired.dropna(subset=condition_order)
            for _, pair_row in paired.iterrows():
                ax.plot(
                    x_positions,
                    [float(pair_row[condition_order[0]]), float(pair_row[condition_order[1]])],
                    color="#7F8C8D",
                    alpha=0.45,
                    linewidth=1.2,
                    zorder=1,
                )

        for idx, condition in enumerate(condition_order):
            cond_df = stain_df.loc[stain_df[condition_column] == condition]
            if cond_df.empty:
                continue
            any_data = True
            values = cond_df["Animal_Top_Surface_Ratio_Median"].to_numpy(dtype=float)
            jitter = rng.uniform(-0.08, 0.08, size=len(values))
            ax.scatter(
                np.full(len(values), x_positions[idx]) + jitter,
                values,
                s=40,
                alpha=0.75,
                color=colors[idx],
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )

            median_val = float(np.median(values))
            std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ax.bar(
                [x_positions[idx]],
                [median_val],
                yerr=[std_val],
                width=0.5,
                color=colors[idx],
                edgecolor=colors[idx],
                ecolor=colors[idx],
                capsize=5,
                alpha=0.30,
                linewidth=1.4,
                zorder=2,
            )
            ax.text(
                x_positions[idx],
                median_val,
                f"n={len(values)}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#22313F",
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(condition_order)
        ax.set_title(stain)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    if not any_data:
        plt.close(fig)
        return None

    axes[0].set_ylabel("Animal Median Top Surface Ratio")
    fig.suptitle(title_prefix)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare Top_Surface_Ratio across animals by condition.")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(DEFAULT_DATA_DIR),
        help="Directory containing numbered animal folders (default: ./Data).",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if (root / "Data").exists() and root.is_dir():
        root = root / "Data"

    results_dir = DEFAULT_RESULTS_DIR if root == DEFAULT_DATA_DIR else root.parent / "Results"
    output_dir = results_dir / "cross_animal_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_df = load_combined_data(root)

    open_sutured_summary = summarize_by_animal(combined_df, "Experimental_Condition")
    open_sutured_summary = open_sutured_summary.loc[
        open_sutured_summary["Experimental_Condition"].isin(["open", "sutured"])
    ].copy()
    open_sutured_summary["Comparison_Type"] = "open_vs_sutured"
    open_sutured_summary = open_sutured_summary.rename(
        columns={"Experimental_Condition": "Comparison_Condition"}
    )

    dark_light_summary = summarize_by_animal(combined_df, "Light_Dark_Condition")
    dark_light_summary = dark_light_summary.loc[
        dark_light_summary["Light_Dark_Condition"].isin(["dark", "light"])
    ].copy()
    dark_light_summary["Comparison_Type"] = "dark_vs_light"
    dark_light_summary = dark_light_summary.rename(
        columns={"Light_Dark_Condition": "Comparison_Condition"}
    )

    summary_df = pd.concat([open_sutured_summary, dark_light_summary], ignore_index=True)
    summary_csv = output_dir / "top_surface_ratio_animal_level_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV: {summary_csv}")

    open_sutured_plot = plot_two_panel_comparison(
        open_sutured_summary.rename(columns={"Comparison_Condition": "Comparison_Condition"}),
        "Comparison_Condition",
        ["sutured", "open"],
        "Top Surface Ratio Across Animals: Sutured vs Open",
        output_dir / "top_surface_ratio_open_vs_sutured.png",
    )
    if open_sutured_plot:
        print(f"Saved plot: {open_sutured_plot}")

    dark_light_plot = plot_two_panel_comparison(
        dark_light_summary.rename(columns={"Comparison_Condition": "Comparison_Condition"}),
        "Comparison_Condition",
        ["dark", "light"],
        "Top Surface Ratio Across Animals: Dark vs Light",
        output_dir / "top_surface_ratio_dark_vs_light.png",
    )
    if dark_light_plot:
        print(f"Saved plot: {dark_light_plot}")


if __name__ == "__main__":
    main()
