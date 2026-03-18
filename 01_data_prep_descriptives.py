#%%
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")


#%%

DATA_DIR = Path(r"C:\Users\thakk\Downloads\Emotions survey")
FILES = {
    "Pre":    DATA_DIR / "Complete_PreMatchFaceReader_PROCESSED_Feb23_20260223_121519.xlsx",
    "Mid":    DATA_DIR / "Complete_FaceReader_Midmatch_Feb19.xlsx",
    "Result": DATA_DIR / "Complete_ResultMoment_FaceReader_Data_Feb23.xlsx",
    "Post":   DATA_DIR / "Complete_FaceReader_PostMatch_Feb19.xlsx",
}

SHEETS = {
    "Pre":    "Pooled_By_Participant",
    "Mid":    "Combined_Pooled",
    "Result": "Pooled_By_Participant",
    "Post":   "Pooled_By_Participant",
}

# Competitions where athletes are visually impaired or blind
BLIND_COMPETITIONS = {
    "Paralympic",
    "IBSA Judo Grand-prix Portugal 2023",
    "IBSA Men's blind football world games 2023",
    "IBSA Men's blind football world championship 2025",
    "IBSA Women's blind football world championship 2025",
    "IBSA Women's blind football world championship Birmingham 2023",
    "Blind Football Grand Prix Tokyo 2019",
}

# Emotion outcomes used throughout the analysis
EMOTION_COLS    = ["Happy", "Sad", "Angry", "Surprised", "Scared", "Disgusted"]
AFFECTIVE_COLS  = ["Valence", "Arousal"]
ALL_OUTCOMES    = EMOTION_COLS + AFFECTIVE_COLS
EXTRA_COLS      = ["Neutral"] #not modelled

# Timepoint order for all plots and tables
TP_ORDER = ["Pre", "Mid", "Result", "Post"]

# Colour palette (colourblind-friendly)
PALETTE = {
    "High PD – Blind":    "#D55E00",
    "High PD – Sighted":  "#E69F00",
    "Low PD – Blind":     "#0072B2",
    "Low PD – Sighted":   "#56B4E9",
}
OUT_DIR = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\descriptive")
#%%

# =============================================================================
# SECTION1- lod abd merge data

def load_and_merge() -> pd.DataFrame:
   
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    keep_cols = (
        ["Name", "Sr_No", "Nationality", "PD", "Gender",
         "Competition", "Result", "TimePoint_Label"]
        + ALL_OUTCOMES + EXTRA_COLS
    )

    frames = []
    for tp, fpath in FILES.items():
        if not fpath.exists():
            print(f"  ⚠  File not found: {fpath.name} — skipping {tp}")
            continue
        df = pd.read_excel(fpath, sheet_name=SHEETS[tp])
        df["TimePoint_Label"] = tp
        present = [c for c in keep_cols if c in df.columns]
        frames.append(df[present])
        print(f"  ✓  Loaded {tp:8s}  ({len(df):>4} rows, {df['Name'].nunique()} athletes)")

    data = pd.concat(frames, ignore_index=True)

    #Classify Sighted vs Blind
    data["Sighted_Blind"] = np.where(
        data["Competition"].isin(BLIND_COMPETITIONS), "Blind", "Sighted"
    )

    # Standardise text columns 
    data["Result"] = data["Result"].astype(str).str.strip().str.capitalize()
    data["PD"]     = data["PD"].astype(str).str.strip().str.capitalize()

    #Drop rows with missing key fields
    data = data.dropna(subset=["Name", "PD", "Result", "Sighted_Blind"])
    data = data[data["PD"].isin(["High", "Low"])]
    data = data[data["Result"].isin(["Win", "Loss"])]

    # Ordered categorical for TimePoint
    data["TimePoint_Label"] = pd.Categorical(
        data["TimePoint_Label"], categories=TP_ORDER, ordered=True
    )

    # Group label for plotting
    data["Group"] = data["PD"] + " PD – " + data["Sighted_Blind"]

    # Integer codes (used in Step 2 modelling)
    data["Athlete_Code"] = pd.Categorical(data["Name"]).codes
    data["PD_Code"]      = (data["PD"] == "High").astype(int)
    data["SB_Code"]      = (data["Sighted_Blind"] == "Blind").astype(int)
    data["Result_Code"]  = (data["Result"] == "Win").astype(int)

    print(f"\n{'─'*60}")
    print(f"  Total observations : {len(data):,}")
    print(f"  Unique athletes    : {data['Name'].nunique():,}")
    print(f"\n  Breakdown by group:")
    breakdown = (
        data.groupby(["Result", "Sighted_Blind", "PD", "TimePoint_Label"])["Name"]
        .count()
        .rename("N")
        .reset_index()
    )
    print(breakdown.to_string(index=False))
    print("="*60 + "\n")

    return data

#%%
# SECTION 2 — Descriptive statistics

def descriptive_tables(data: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full summary table
    full_summary = (
        data.groupby(["Result", "Sighted_Blind", "PD", "TimePoint_Label"])[ALL_OUTCOMES]
        .agg(["mean", "std", "count"])
        .round(4)
    )
    full_summary.to_excel(out_dir / "Table_Full_Descriptives.xlsx")

    #Clean mean-only table
    for result_label in ["Win", "Loss"]:
        subset = data[data["Result"] == result_label]
        clean = (
            subset.groupby(["Sighted_Blind", "PD", "TimePoint_Label"])[ALL_OUTCOMES]
            .mean()
            .round(4)
        )
        study = "Winners" if result_label == "Win" else "Losers"
        clean.to_excel(out_dir / f"Table_Means_{study}.xlsx")

    # Sample sizes
    ns = (
        data.groupby(["Result", "Sighted_Blind", "PD", "TimePoint_Label"])["Name"]
        .count()
        .rename("N")
        .reset_index()
    )
    ns.to_excel(out_dir / "Table_Sample_Sizes.xlsx", index=False)

    print(f"  ✓  Descriptive tables saved to {out_dir.name}/")

#%%
# SECTION 3 — TRAJECTORY PLOTS

def trajectory_plots(data: pd.DataFrame, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    primary_outcomes = ["Valence", "Arousal", "Happy", "Sad", "Angry"]

    for outcome in primary_outcomes:
        if outcome not in data.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

        for ax, result_label, title in zip(
            axes, ["Win", "Loss"], ["Winners", "Losers"]
        ):
            subset = data[data["Result"] == result_label].dropna(subset=[outcome])

            grp_stats = (
                subset.groupby(["Group", "TimePoint_Label"])[outcome]
                .agg(["mean", "sem"])
                .reset_index()
            )

            for grp, color in PALETTE.items():
                d = grp_stats[grp_stats["Group"] == grp].copy()
                if d.empty:
                    continue
                tp_labels = d["TimePoint_Label"].astype(str).tolist()
                means     = d["mean"].values
                sems      = d["sem"].values

                ax.plot(tp_labels, means, marker="o", color=color,
                        label=grp, linewidth=2.5, markersize=8, zorder=3)
                ax.fill_between(tp_labels,
                                means - sems, means + sems,
                                alpha=0.15, color=color, zorder=2)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Match Situation", fontsize=11)
            ax.set_ylabel(outcome, fontsize=11)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.tick_params(axis="x", labelsize=10)

            if outcome == "Valence":
                ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.6)

            if ax == axes[1]:
                ax.legend(fontsize=9, loc="best", framealpha=0.85,
                          title="Group", title_fontsize=9)

        fig.suptitle(
            f"{outcome}  —  Mean ± SEM across Match Situations\n"
            f"by Power Distance × Vision Status",
            fontsize=13, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        fpath = out_dir / f"Trajectory_{outcome}.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  ✓  Trajectory plots saved to {out_dir.name}/")

#%%
# SECTION 4 — CORRELATION HEATMAP

def correlation_heatmap(data: pd.DataFrame, out_dir: Path):
    """
    Generate a heatmap of correlations between all emotion outcome variables.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    for ax, result_label, title in zip(
        axes, ["Win", "Loss"], ["Winners", "Losers"]
    ):
        subset = data[data["Result"] == result_label][ALL_OUTCOMES].dropna()
        corr   = subset.corr()
        mask   = np.triu(np.ones_like(corr, dtype=bool))

        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, annot_kws={"size": 9},
            vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title(f"{title}", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(
        "Emotion Variable Correlations (All Timepoints Combined)",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "Correlation_Heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Correlation heatmap saved")

#%%
# SECTION 5 — BOX PLOTS PER GROUP × TIMEPOINT

def boxplots(data: pd.DataFrame, out_dir: Path):

    out_dir.mkdir(parents=True, exist_ok=True)

    for outcome in ["Valence", "Happy"]:
        if outcome not in data.columns:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        for ax, result_label, title in zip(
            axes, ["Win", "Loss"], ["Winners", "Losers"]
        ):
            subset = data[data["Result"] == result_label].dropna(subset=[outcome])
            grp_order = list(PALETTE.keys())

            sns.boxplot(
                data=subset, x="TimePoint_Label", y=outcome,
                hue="Group", hue_order=grp_order, palette=PALETTE,
                ax=ax, width=0.65, fliersize=2, linewidth=1,
                order=TP_ORDER
            )

            if outcome == "Valence":
                ax.axhline(0, color="grey", linestyle="--",
                           linewidth=1, alpha=0.6, zorder=0)

            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xlabel("Match Situation", fontsize=11)
            ax.set_ylabel(outcome, fontsize=11)
            ax.grid(axis="y", alpha=0.3, linestyle="--")
            ax.legend(fontsize=8, loc="upper left", framealpha=0.8,
                      title="Group", title_fontsize=8)

        fig.suptitle(
            f"{outcome} Distribution by Group × Match Situation",
            fontsize=13, fontweight="bold", y=1.01
        )
        plt.tight_layout()
        plt.savefig(out_dir / f"Boxplot_{outcome}.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"  ✓  Box plots saved")

#%%
# SECTION 6 — SAMPLE SIZE SUMMARY PLOT

def sample_size_plot(data: pd.DataFrame, out_dir: Path):
    """
    Bar chart showing N per group × timepoint — useful for reporting
    and for understanding the missing data situation with Mid-match.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ns = (
        data.groupby(["Result", "Group", "TimePoint_Label"])["Name"]
        .count()
        .rename("N")
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, result_label, title in zip(
        axes, ["Win", "Loss"], ["Winners", "Losers"]
    ):
        subset = ns[ns["Result"] == result_label]
        grp_order = list(PALETTE.keys())
        color_list = [PALETTE[g] for g in grp_order]

        sns.barplot(
            data=subset, x="TimePoint_Label", y="N",
            hue="Group", hue_order=grp_order, palette=PALETTE,
            ax=ax, order=TP_ORDER
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Match Situation", fontsize=11)
        ax.set_ylabel("Number of Athletes", fontsize=11)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.legend(fontsize=8, framealpha=0.8, title="Group", title_fontsize=8)

    fig.suptitle(
        "Sample Sizes per Group × Match Situation",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(out_dir / "Sample_Sizes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Sample size plot saved")

#%%
# MAIN

def main():
    print("\n" + "="*60)
    print("STEP 1 — DATA PREPARATION & DESCRIPTIVE ANALYSIS")
    print("="*60)

    # Load & merge
    data = load_and_merge()

    # Save merged dataset 
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\Merged_AllTimepoints.csv")
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(merged_path, index=False)
    print(f"  ✓  Merged dataset saved → outputs/Merged_AllTimepoints.csv\n")

    # Descriptive tables 
    print("─── Descriptive Tables ───")
    descriptive_tables(data, OUT_DIR)

    # Plots 
    print("\n─── Trajectory Plots ───")
    trajectory_plots(data, OUT_DIR)

    print("\n─── Correlation Heatmap ───")
    correlation_heatmap(data, OUT_DIR)

    print("\n─── Box Plots ───")
    boxplots(data, OUT_DIR)

    print("\n─── Sample Size Plot ───")
    sample_size_plot(data, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"✅  STEP 1 COMPLETE")
    print(f"    All outputs saved in: outputs/descriptive/")
    print(f"    Next: run  python 02_phase1_crosssectional.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

# %%
