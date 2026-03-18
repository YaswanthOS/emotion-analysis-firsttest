"""Cross-Sectional Multivariate Multilevel Bayesian Models

What this script does:
  For each timepoint (Pre, Result, Post) and each study (Winners, Losers):
    - Fits a multivariate multilevel Bayesian regression model
    - All 8 emotion outcomes modelled simultaneously in one model
    - Extracts the 4 key group contrasts from study design
    - Saves posterior summaries, contrast tables, and diagnostic plots

Requirements:
    pip install pymc arviz pandas numpy matplotlib seaborn openpyxl

HOW TO RUN IN VS CODE:
    - Run cells one by one with Shift+Enter
    - OR press Ctrl+F5 to run the whole script at once
    - For a quick test: run the TEST RUN cell at the bottom first
    - For the full analysis: run the FULL RUN cell at the bottom
"""

#%%
# IMPORTS

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

print("✓ Imports done")

#%%
# CONFIGURATION — edit the paths here for your machine

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

#Where the merged CSV from Step 1 was saved
# If you ran Step 1 already, this file exists and Step 2 will use it directly
MERGED_CSV_PATH = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\Merged_AllTimepoints.csv")

#Where to save all model outputs 
OUT_DIR = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\phase1")

#Competitions where athletes are visually impaired or blind
BLIND_COMPETITIONS = {
    "Paralympic",
    "IBSA Judo Grand-prix Portugal 2023",
    "IBSA Men's blind football world games 2023",
    "IBSA Men's blind football world championship 2025",
    "IBSA Women's blind football world championship 2025",
    "IBSA Women's blind football world championship Birmingham 2023",
    "Blind Football Grand Prix Tokyo 2019",
}

#Emotion outcomes modelled simultaneously
MODEL_OUTCOMES = [
    "Happy", "Sad", "Angry", "Surprised",
    "Scared", "Disgusted", "Valence", "Arousal"
]
N_OUTCOMES = len(MODEL_OUTCOMES)

print("✓ Configuration set")
print(f"  Data folder  : {DATA_DIR}")
print(f"  Merged CSV   : {MERGED_CSV_PATH}")
print(f"  Output folder: {OUT_DIR}")


#%%
# SECTION 1 — DATA LOADING

def load_data() -> pd.DataFrame:
    """
    Load data for modelling.

    First tries to load the merged CSV saved by Step 1 (fast).
    If that file doesn't exist, loads and merges the Excel files directly.
    """
    if MERGED_CSV_PATH.exists():
        print(f"  ✓ Found merged CSV — loading from file (fast)")
        data = pd.read_csv(MERGED_CSV_PATH)
        data["TimePoint_Label"] = pd.Categorical(
            data["TimePoint_Label"],
            categories=["Pre", "Mid", "Result", "Post"],
            ordered=True
        )
        print(f"  ✓ Loaded {len(data):,} rows, {data['Name'].nunique()} athletes")
        return data

    print("  Merged CSV not found — loading from Excel files")
    keep_cols = (
        ["Name", "Sr_No", "Nationality", "PD", "Gender",
         "Competition", "Result", "TimePoint_Label"]
        + MODEL_OUTCOMES
    )
    frames = []
    for tp, fpath in FILES.items():
        if not fpath.exists():
            print(f"  ⚠ File not found: {fpath.name} — skipping {tp}")
            continue
        df = pd.read_excel(fpath, sheet_name=SHEETS[tp])
        df["TimePoint_Label"] = tp
        present = [c for c in keep_cols if c in df.columns]
        frames.append(df[present])
        print(f"  ✓ Loaded {tp}")

    data = pd.concat(frames, ignore_index=True)

    # Classify Sighted vs Blind
    data["Sighted_Blind"] = np.where(
        data["Competition"].isin(BLIND_COMPETITIONS), "Blind", "Sighted"
    )

    # Standardise text columns
    data["Result"] = data["Result"].astype(str).str.strip().str.capitalize()
    data["PD"]     = data["PD"].astype(str).str.strip().str.capitalize()

    # Remove rows with missing key fields
    data = data.dropna(subset=["Name", "PD", "Result"])
    data = data[
        data["PD"].isin(["High", "Low"]) &
        data["Result"].isin(["Win", "Loss"])
    ]

    data["TimePoint_Label"] = pd.Categorical(
        data["TimePoint_Label"],
        categories=["Pre", "Mid", "Result", "Post"],
        ordered=True
    )

    print(f"  ✓ Loaded {len(data):,} rows, {data['Name'].nunique()} athletes")
    return data


def prepare_subset(
    data: pd.DataFrame,
    timepoint: str,
    study: str
):
    """
    Filter the full dataset down to one timepoint and one study.

    Parameters
    ----------
    data      : the full merged dataset
    timepoint : "Pre", "Mid", "Result", or "Post"
    study     : "winners" or "losers"

    Returns
    -------
    df       : filtered DataFrame ready for modelling
    athletes : array of unique athlete names in this subset
    """
    result_label = "Win" if study == "winners" else "Loss"

    df = data[
        (data["TimePoint_Label"] == timepoint) &
        (data["Result"] == result_label)
    ].copy()

    # Drop rows with any missing emotion data
    df = df.dropna(subset=MODEL_OUTCOMES)

    # Create fresh athlete index (0, 1, 2, ...) within this subset
    athletes = df["Name"].unique()
    ath_map  = {a: i for i, a in enumerate(athletes)}
    df["Ath_idx"] = df["Name"].map(ath_map)

    # Numeric codes for predictors
    df["PD_Code"] = (df["PD"] == "High").astype(int)      # 0=Low, 1=High
    df["SB_Code"] = (df["Sighted_Blind"] == "Blind").astype(int)  # 0=Sighted, 1=Blind

    print(f"\n  Subset ready: {timepoint} | {study.upper()}")
    print(f"  ├── Athletes  : {len(athletes)}")
    print(f"  ├── Rows      : {len(df)}")
    print(f"  └── N per group:")
    print(df.groupby(["Sighted_Blind", "PD"])["Name"]
            .count().rename("N").to_string())

    return df, athletes


#%%
# SECTION 2 — BAYESIAN MODEL

def fit_model(
    df: pd.DataFrame,
    athletes: np.ndarray,
    n_samples: int = 2000,
    n_tune: int = 1000,
):
    """
    Fit the multivariate multilevel Bayesian regression model.

    THE MODEL IN PLAIN ENGLISH:
    ───────────────────────────
    For each athlete, at each timepoint, we observe 8 emotion scores
    simultaneously. We want to know:
      - Do High PD athletes score differently than Low PD athletes?
      - Do Blind athletes score differently than Sighted athletes?
      - Does the PD effect differ between Blind and Sighted athletes?

    The model has two levels:
      Level 1 (athlete): each athlete gets their own personal baseline
                         emotion level (random intercept). This accounts
                         for the fact that some people just naturally have
                         more expressive faces than others.
      Level 2 (group):   on top of that baseline, we estimate the group
                         effects (PD, Sighted/Blind, their interaction).

    All 8 outcomes are estimated simultaneously, and the model also
    learns how the emotions correlate with each other (via the LKJ prior
    on the covariance matrix Σ).

    Parameters
    ----------
    df         : prepared subset dataframe from prepare_subset()
    athletes   : array of unique athlete names
    n_samples  : MCMC draws per chain (use 500 for testing, 2000 for final)
    n_tune     : MCMC tuning steps (use 300 for testing, 1000 for final)

    Returns
    -------
    trace : ArviZ InferenceData — contains all posterior samples
    model : PyMC model object
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "\n\nPyMC is not installed.\n"
            "Run this in your terminal:\n"
            "    pip install pymc arviz\n"
            "Then restart VS Code and try again.\n"
        )

    n_athletes = len(athletes)

    # Prepare arrays for the model
    Y       = df[MODEL_OUTCOMES].values.astype(float)  # shape: (N, 8)
    ath_idx = df["Ath_idx"].values                      # which athlete is each row
    pd_code = df["PD_Code"].values.astype(float)        # 0=Low PD, 1=High PD
    sb_code = df["SB_Code"].values.astype(float)        # 0=Sighted, 1=Blind
    pd_sb   = pd_code * sb_code                         # interaction term

    # coords give human-readable labels to model dimensions
    coords = {
    "athlete":     athletes,
    "outcome":     MODEL_OUTCOMES,
    "observation": np.arange(len(df)),  
}

    with pm.Model(coords=coords) as model:

        # ── LEVEL 2: Hyperpriors ──────────────────────────────────────────────
        # These describe the distribution of athlete baselines across the
        # whole sample — what is the average baseline, and how much do
        # athletes vary around it?
        mu_alpha    = pm.Normal(
            "mu_alpha",
            mu=0.0, sigma=0.5,
            dims="outcome"
        )
        sigma_alpha = pm.HalfNormal(
            "sigma_alpha",
            sigma=0.3,
            dims="outcome"
        )

        # ── LEVEL 1: Random intercept per athlete ─────────────────────────────
        # Each athlete gets their own personal offset from the group average.
        # This is what makes the model "multilevel" — we're modelling
        # variation both within groups and between individual athletes.
        #
        # We use the "non-centred" parameterisation (offset × sigma)
        # which helps the sampler work more efficiently.
        alpha_offset = pm.Normal(
            "alpha_offset",
            mu=0, sigma=1,
            dims=("athlete", "outcome")
        )
        alpha = pm.Deterministic(
            "alpha",
            mu_alpha + alpha_offset * sigma_alpha,
            dims=("athlete", "outcome")
        )

        # Fixed effects (group-level predictors)
        # One coefficient per outcome (8 values each)
        # Prior: Normal(0, 0.5) — weakly informative, says we expect
        # effects to be smallish but lets the data override that
        beta_PD    = pm.Normal("beta_PD",    mu=0, sigma=0.5, dims="outcome")
        beta_SB    = pm.Normal("beta_SB",    mu=0, sigma=0.5, dims="outcome")
        beta_PD_SB = pm.Normal("beta_PD_SB", mu=0, sigma=0.3, dims="outcome")

        # ── Residual covariance matrix (the multivariate part) ────────────────
        # This estimates how the 8 emotions co-vary with each other
        # after accounting for group effects.
        # LKJ prior with eta=2 gives mild regularisation toward
        # an identity matrix (i.e., slightly sceptical of extreme correlations)
        sd_dist = pm.HalfNormal.dist(sigma=0.5, shape=N_OUTCOMES)
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol_cov",
            n=N_OUTCOMES,
            eta=2.0,
            sd_dist=sd_dist,
            compute_corr=True
        )

        # Linear predictor
        # For each observation, predict all 8 outcomes simultaneously.
        # alpha[ath_idx] looks up each athlete's personal baseline.
        # The [:, None] makes pd_code a column vector so it broadcasts
        # correctly across all 8 outcomes.
        mu = (
            alpha[ath_idx]
            + beta_PD    * pd_code[:, None]
            + beta_SB    * sb_code[:, None]
            + beta_PD_SB * pd_sb[:, None]
        )

        #Likelihood
        # "Given our predictions mu and the correlation structure,
        #  how likely is the data we actually observed?"
        # MvNormal = Multivariate Normal — all 8 outcomes jointly
        pm.MvNormal(
            "obs",
            mu=mu,
            chol=chol,
            observed=Y,
            dims=("observation", "outcome")   # ← renamed
        )

        # MCMC Sampling 
        # target_accept=0.9 makes the sampler take smaller, more careful steps
        # This is recommended for multilevel models
        trace = pm.sample(
            n_samples,
            tune=n_tune,
            target_accept=0.9,
            return_inferencedata=True,
            progressbar=True,
            random_seed=42,
            idata_kwargs={"log_likelihood": True},
        )

    return trace, model


#%%
# SECTION 3 — POSTERIOR SUMMARY

def posterior_summary(trace, tag: str, out_dir: Path) -> pd.DataFrame:
    """
    Extract a readable summary table from the model results.

    For each predictor × outcome combination, reports:
      mean     — most likely value of the effect
      sd       — uncertainty in the estimate
      hdi_3%   — lower bound of 94% credible interval
      hdi_97%  — upper bound of 94% credible interval
      r_hat    — convergence check (should be very close to 1.0)
      P(>0)    — probability the effect is positive
      P(<0)    — probability the effect is negative

    HOW TO READ THE RESULTS:
      If hdi_3% and hdi_97% are both the same sign (both positive or
      both negative), the effect is credibly non-zero — the Bayesian
      equivalent of "statistically significant".

      r_hat > 1.05 means the model may not have converged — treat those
      estimates with caution and consider running with more tuning steps.
    """
    import arviz as az

    var_names = ["beta_PD", "beta_SB", "beta_PD_SB"]

    summary = az.summary(
        trace,
        var_names=var_names,
        hdi_prob=0.94,
        round_to=4,
    )

    # Add probability of direction for each outcome
    for var in var_names:
        # samples shape: (chains, draws, n_outcomes)
        samples = trace.posterior[var].values
        for i, outcome in enumerate(MODEL_OUTCOMES):
            row_label = f"{var}[{outcome}]"
            if row_label in summary.index:
                s = samples[:, :, i].flatten()
                summary.loc[row_label, "P(>0)"] = round((s > 0).mean(), 3)
                summary.loc[row_label, "P(<0)"] = round((s < 0).mean(), 3)

    # Save to CSV
    summary.to_csv(out_dir / f"posterior_summary_{tag}.csv")

    # Print to console
    print(f"\n  ── Posterior Summary: {tag} ──")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat",
                   "P(>0)", "P(<0)"]].to_string())

    return summary


#%%
# SECTION 4 — CONTRAST ANALYSIS

def compute_contrasts(trace, tag: str, out_dir: Path) -> pd.DataFrame:
    """
    Compute the 4 key group contrasts from Saumya's study design.

    WHY WE DO THIS:
    The model only has 3 parameters (beta_PD, beta_SB, beta_PD_SB),
    but Saumya wants 4 specific comparisons. These are derived by
    combining the parameters mathematically:

      Contrast 1: High vs Low PD | Sighted athletes only
        → When Blind=0, the interaction term drops out
        → Effect = just beta_PD

      Contrast 2: High vs Low PD | Blind athletes only
        → When Blind=1, the interaction adds on top
        → Effect = beta_PD + beta_PD_SB

      Contrast 3: Blind vs Sighted | Low PD athletes only
        → When PD=0, the interaction term drops out
        → Effect = just beta_SB

      Contrast 4: Blind vs Sighted | High PD athletes only
        → When PD=1, the interaction adds on top
        → Effect = beta_SB + beta_PD_SB

    We compute each contrast on the FULL posterior distribution
    (not just the mean), so we get proper uncertainty estimates.
    """
    import arviz as az

    post = trace.posterior
    rows = []

    for i, outcome in enumerate(MODEL_OUTCOMES):

        # Get full posterior samples for this outcome
        # shape after .flatten(): (n_chains × n_samples,)
        b_PD    = post["beta_PD"].values[:, :, i].flatten()
        b_SB    = post["beta_SB"].values[:, :, i].flatten()
        b_PD_SB = post["beta_PD_SB"].values[:, :, i].flatten()

        # Define the 4 contrasts
        contrasts = {
            "High vs Low PD | Sighted":   b_PD,
            "High vs Low PD | Blind":     b_PD + b_PD_SB,
            "Blind vs Sighted | Low PD":  b_SB,
            "Blind vs Sighted | High PD": b_SB + b_PD_SB,
        }

        for contrast_name, samples in contrasts.items():
            hdi   = az.hdi(samples, hdi_prob=0.94)
            p_pos = (samples > 0).mean()

            # HDI_excl_0: "Yes" means the interval doesn't cross zero
            # → credible non-zero effect
            hdis_excl = "Yes" if (hdi[0] > 0 or hdi[1] < 0) else "No"

            rows.append({
                "Outcome":    outcome,
                "Contrast":   contrast_name,
                "Mean":       round(float(samples.mean()), 4),
                "SD":         round(float(samples.std()), 4),
                "HDI_low":    round(float(hdi[0]), 4),
                "HDI_high":   round(float(hdi[1]), 4),
                "P(>0)":      round(float(p_pos), 3),
                "Direction":  "Positive" if p_pos > 0.5 else "Negative",
                "HDI_excl_0": hdis_excl,
            })

    contrast_df = pd.DataFrame(rows)
    contrast_df.to_csv(out_dir / f"contrasts_{tag}.csv", index=False)

    # Print summary to console — ✓ marks credible effects
    print(f"\n  ── Contrasts: {tag} ──")
    print(f"  {'Outcome':<12} {'Contrast':<35} {'Mean':>7} "
          f"{'HDI_low':>9} {'HDI_high':>9} {'P(>0)':>7} {'*':>5}")
    print("  " + "─" * 85)
    for _, row in contrast_df.iterrows():
        star = "✓" if row["HDI_excl_0"] == "Yes" else ""
        print(
            f"  {row['Outcome']:<12} {row['Contrast']:<35} "
            f"{row['Mean']:>7.4f} {row['HDI_low']:>9.4f} "
            f"{row['HDI_high']:>9.4f} {row['P(>0)']:>7.3f} {star:>5}"
        )

    return contrast_df


#%%
# SECTION 5 — VISUALISATIONS

def plot_forest(trace, tag: str, out_dir: Path, timepoint: str, study: str):
    """
    Forest plot of posterior effects for all 3 predictors × 8 outcomes.

    HOW TO READ IT:
      Each row is one emotion outcome.
      The dot is the posterior mean, the horizontal bar is the 94% HDI.
      If the bar does NOT cross the red dashed zero line → credible effect.
      Orange coloured rows = credible effects.
      Grey rows = HDI crosses zero, effect is uncertain.
    """
    import arviz as az

    fig, axes = plt.subplots(1, 3, figsize=(15, 7))

    var_map = {
        "beta_PD":    ("High vs Low PD",         axes[0]),
        "beta_SB":    ("Blind vs Sighted",        axes[1]),
        "beta_PD_SB": ("PD × Vision Interaction", axes[2]),
    }

    for var, (title, ax) in var_map.items():
        # shape: (chains, draws, n_outcomes)
        samples = trace.posterior[var].values
        means   = samples.reshape(-1, N_OUTCOMES).mean(axis=0)
        hdis    = np.array([
            az.hdi(samples[:, :, i].flatten(), hdi_prob=0.94)
            for i in range(N_OUTCOMES)
        ])

        y_pos  = np.arange(N_OUTCOMES)
        colors = [
            "#D55E00" if (hdis[i, 0] > 0 or hdis[i, 1] < 0) else "#999999"
            for i in range(N_OUTCOMES)
        ]

        # Draw bars, error bars, and dots
        ax.barh(y_pos, means, color=colors, alpha=0.6, height=0.55)
        ax.errorbar(
            means, y_pos,
            xerr=[means - hdis[:, 0], hdis[:, 1] - means],
            fmt="none", color="black", capsize=4, linewidth=1.5
        )
        ax.scatter(means, y_pos, color=colors, zorder=5, s=60)

        # Zero line
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

        # Shade rows with credible effects
        for i in range(N_OUTCOMES):
            if hdis[i, 0] > 0 or hdis[i, 1] < 0:
                ax.axhspan(i - 0.4, i + 0.4, alpha=0.06, color="#D55E00")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODEL_OUTCOMES, fontsize=10)
        ax.set_xlabel("Effect Size (posterior mean)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

    study_label = "Winners" if study == "winners" else "Losers"
    fig.suptitle(
        f"Posterior Effects — {timepoint} | {study_label}\n"
        f"Orange = 94% HDI excludes zero  |  Grey = uncertain",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"forest_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Forest plot saved → {fpath.name}")


def plot_correlation_matrix(trace, tag: str, out_dir: Path,
                             timepoint: str, study: str):
    """
    Heatmap of the posterior mean residual correlation matrix.

    This shows how the 8 emotions co-vary with each other AFTER
    accounting for group differences (PD, Vision status).
    This is the model-estimated version — more reliable than the
    raw correlation heatmap from Step 1.
    """
    # Extract posterior correlation matrix
    # shape: (chains, draws, n_outcomes, n_outcomes)
    corr_samples = trace.posterior["chol_cov_corr"].values
    corr_mean    = corr_samples.reshape(-1, N_OUTCOMES, N_OUTCOMES).mean(axis=0)
    corr_df      = pd.DataFrame(
        corr_mean, index=MODEL_OUTCOMES, columns=MODEL_OUTCOMES
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_mean, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, ax=ax,
        linewidths=0.5, annot_kws={"size": 10},
        vmin=-1, vmax=1,
        cbar_kws={"label": "Posterior Mean Correlation", "shrink": 0.8}
    )
    study_label = "Winners" if study == "winners" else "Losers"
    ax.set_title(
        f"Residual Emotion Correlations (Model-Estimated)\n"
        f"{timepoint} | {study_label}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"correlation_matrix_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Correlation matrix saved → {fpath.name}")


def plot_trace(trace, tag: str, out_dir: Path):
    """
    MCMC convergence diagnostic plots.

    HOW TO READ THEM:
      Left panel: the sampled values over time for each chain
        → Good: all chains are overlapping and look like static noise
          (this is what people mean by 'fuzzy caterpillars')
        → Bad: chains drifting upward/downward, or separated from each other

      Right panel: kernel density estimate of the posterior distribution
        → Good: all chains have the same smooth bell-shaped curve
        → Bad: chains have different peaks or very different widths

    If the plots look bad, try running with more tuning steps (--tune 2000).
    """
    import arviz as az

    az.plot_trace(
        trace,
        var_names=["beta_PD", "beta_SB", "beta_PD_SB"],
        compact=True,
        figsize=(12, 8)
    )
    plt.suptitle(
        f"MCMC Trace Plots — {tag}\n"
        f"Good convergence = chains mix together like fuzzy caterpillars",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fpath = out_dir / f"trace_{tag}.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Trace plot saved → {fpath.name}")


def plot_contrasts_heatmap(contrast_df: pd.DataFrame, tag: str,
                            out_dir: Path, timepoint: str, study: str):
    """
    Summary heatmap of all contrast results.

    Rows = emotion outcomes
    Columns = the 4 group comparisons
    Cell colour = direction and size of effect
    * = the HDI excludes zero (credible effect)

    This is the most compact way to see all results at once.
    """
    pivot      = contrast_df.pivot(
        index="Outcome", columns="Contrast", values="Mean"
    )
    credible   = contrast_df.pivot(
        index="Outcome", columns="Contrast", values="HDI_excl_0"
    )

    # Build annotation: value + asterisk for credible effects
    annot_text = pivot.copy().astype(str)
    for r in credible.index:
        for c in credible.columns:
            val = f"{pivot.loc[r, c]:.3f}"
            if credible.loc[r, c] == "Yes":
                val += " *"
            annot_text.loc[r, c] = val

    fig, ax = plt.subplots(figsize=(13, 6))
    sns.heatmap(
        pivot, annot=annot_text, fmt="",
        cmap="RdBu_r", center=0, ax=ax,
        linewidths=0.5, annot_kws={"size": 9},
        cbar_kws={"label": "Posterior Mean Effect", "shrink": 0.8}
    )
    study_label = "Winners" if study == "winners" else "Losers"
    ax.set_title(
        f"Group Contrasts — {timepoint} | {study_label}\n"
        f"* = 94% HDI excludes zero (credible effect)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=15, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    plt.tight_layout()
    fpath = out_dir / f"contrasts_heatmap_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Contrast heatmap saved → {fpath.name}")


#%%
# SECTION 6 — RUN ONE COMPLETE MODEL

def run_one_model(
    data: pd.DataFrame,
    timepoint: str,
    study: str,
    n_samples: int = 2000,
    n_tune: int = 1000,
    out_dir: Path = OUT_DIR,
):
    """
    Run the full pipeline for one timepoint × one study:
      1. Filter data to the right subset
      2. Fit the Bayesian model
      3. Check convergence (R-hat)
      4. Generate posterior summary table
      5. Compute the 4 group contrasts
      6. Save all plots
      7. Save the full posterior trace to disk

    Parameters
    ----------
    timepoint : "Pre", "Mid", "Result", or "Post"
    study     : "winners" or "losers"
    n_samples : MCMC draws per chain
    n_tune    : MCMC tuning steps
    out_dir   : where to save outputs
    """
    import arviz as az

    tag    = f"{timepoint}_{study}"
    tp_dir = out_dir / timepoint.lower()
    tp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  RUNNING MODEL: {timepoint.upper()} | {study.upper()}")
    print(f"  samples={n_samples}, tune={n_tune}")
    print(f"{'='*60}")

    # Step 1: prepare data subset
    df, athletes = prepare_subset(data, timepoint, study)

    if len(df) < 20:
        print(f"  ⚠ Only {len(df)} observations — too few to fit model. Skipping.")
        return None

    # Step 2: fit model
    trace, model = fit_model(df, athletes, n_samples=n_samples, n_tune=n_tune)

    # Step 3: check convergence
    rhat = az.rhat(trace, var_names=["beta_PD", "beta_SB", "beta_PD_SB"])
    max_rhat = float(max(
        rhat["beta_PD"].values.max(),
        rhat["beta_SB"].values.max(),
        rhat["beta_PD_SB"].values.max()
    ))
    if max_rhat > 1.05:
        print(f"\n  ⚠ WARNING: max R-hat = {max_rhat:.3f}")
        print(f"    The model may not have converged fully.")
        print(f"    Consider re-running with a higher --tune value (e.g. 2000)")
    else:
        print(f"\n  ✓ Convergence OK  (max R-hat = {max_rhat:.3f})")

    # Step 4-7: outputs
    posterior_summary(trace, tag, tp_dir)
    contrast_df = compute_contrasts(trace, tag, tp_dir)
    plot_forest(trace, tag, tp_dir, timepoint, study)
    plot_correlation_matrix(trace, tag, tp_dir, timepoint, study)
    plot_trace(trace, tag, tp_dir)
    plot_contrasts_heatmap(contrast_df, tag, tp_dir, timepoint, study)

    # Save the full trace (can be reloaded later with az.from_netcdf)
    trace_path = tp_dir / f"trace_{tag}.nc"
    trace.to_netcdf(str(trace_path))
    print(f"  ✓  Full trace saved → {trace_path.name}")
    print(f"\n  Done: {tag}")

    return trace


#%%
# QUICK TEST RUN — run this cell first to check everything works
# This runs just ONE model (Pre | Winners) with very few samples.
# Takes about 5-10 minutes.
# Check the outputs folder afterwards — if files appear, everything is working.
# Only then proceed to the FULL RUN cell below.

def test_run():
    print("\n" + "="*60)
    print("  QUICK TEST RUN")
    print("  Timepoint : Pre")
    print("  Study     : winners")
    print("  Samples   : 200  (very low — just checking the model runs)")
    print("  Tune      : 200")
    print("="*60)

    data = load_data()

    trace = run_one_model(
        data,
        timepoint  = "Pre",
        study      = "winners",
        n_samples  = 200,   # very low — just for testing
        n_tune     = 200,
        out_dir    = OUT_DIR,
    )

    if trace is not None:
        print("\n" + "="*60)
        print("✅ TEST PASSED — model ran successfully")
        print(f"   Check outputs at: {OUT_DIR / 'pre'}")
        print("   If everything looks right, run the FULL RUN cell below")
        print("="*60)
    else:
        print("\n⚠ TEST FAILED — check the error messages above")

test_run()


#%%
# FULL RUN — run this cell after the test passes
# Runs all 6 models: Pre / Result / Post  ×  Winners / Losers
# Total time: roughly 2-4 hours.
# You can also run just one model at a time by calling
# run_one_model() directly with specific timepoint and study arguments.

def full_run(
    timepoints = ["Pre", "Result", "Post"],   # change this list to run specific timepoints
    studies    = ["winners", "losers"],        # change this to run specific studies
    n_samples  = 2000,                         # increase to 4000 for final paper results
    n_tune     = 1000,                         # increase to 2000 if R-hat warnings appear
):
    print("\n" + "="*60)
    print("  FULL PHASE 1 ANALYSIS")
    print(f"  Timepoints : {timepoints}")
    print(f"  Studies    : {studies}")
    print(f"  Samples    : {n_samples} per chain")
    print(f"  Tune steps : {n_tune}")
    print(f"  Output dir : {OUT_DIR}")
    print("="*60)

    data    = load_data()
    results = {}

    for timepoint in timepoints:
        for study in studies:
            trace = run_one_model(
                data,
                timepoint  = timepoint,
                study      = study,
                n_samples  = n_samples,
                n_tune     = n_tune,
                out_dir    = OUT_DIR,
            )
            results[f"{timepoint}_{study}"] = trace

    print(f"\n{'='*60}")
    print(f"✅  ALL MODELS COMPLETE")
    print(f"    Results saved in: {OUT_DIR}")
    print(f"\n    For each model you will find:")
    print(f"    ├── posterior_summary_[tp]_[study].csv")
    print(f"    ├── contrasts_[tp]_[study].csv")
    print(f"    ├── forest_[tp]_[study].png")
    print(f"    ├── contrasts_heatmap_[tp]_[study].png")
    print(f"    ├── correlation_matrix_[tp]_[study].png")
    print(f"    ├── trace_[tp]_[study].png")
    print(f"    └── trace_[tp]_[study].nc")
    print(f"{'='*60}\n")

    return results


# full_run()


#%%
# SINGLE MODEL RUN — useful if you want to run just one specific model

# TIMEPOINT = "Result"    # options: "Pre", "Mid", "Result", "Post"
# STUDY     = "winners"   # options: "winners", "losers"
# SAMPLES   = 2000
# TUNE      = 1000

# data  = load_data()
# trace = run_one_model(
#     data,
#     timepoint = TIMEPOINT,
#     study     = STUDY,
#     n_samples = SAMPLES,
#     n_tune    = TUNE,
#     out_dir   = OUT_DIR,
# )