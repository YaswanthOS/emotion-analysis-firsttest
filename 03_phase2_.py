"""
STEP 3 — Phase 2: Combined Longitudinal Multivariate Multilevel Bayesian Model

WHAT THIS SCRIPT DOES:
  Phase 2 takes all three timepoints (Pre, Result, Post) and puts them into
  ONE single model, rather than running separate models per timepoint as in
  Phase 1. TimePoint becomes a variable inside the model, and its interactions
  with PD and Vision give us the richest scientific findings.

PHASE 2 vs PHASE 1:
  - The random intercept per athlete is estimated much more reliably because
    the model now sees each athlete at multiple timepoints. It can cleanly
    separate stable individual expressiveness from cultural PD effects.
  - We can ask trajectory questions: does the emotional change from
    Pre to Result differ between High and Low PD athletes?
  - The 3-way interaction (PD x Vision x TimePoint) directly tests whether
    Blind and Sighted athletes from different PD cultures diverge specifically
    at the emotionally intense Result moment.

MODEL STRUCTURE:
  Pre-match is the reference category. All TimePoint effects are measured
  as changes FROM Pre-match. So:
    beta_PD           = PD effect at Pre-match
    beta_PD_TP_result = additional PD effect at Result (on top of Pre)
    beta_PD_TP_post   = additional PD effect at Post (on top of Pre)

  The 4 group contrasts are computed at each timepoint by combining the
  relevant parameters.

Requirements:
    pip install pymc arviz pandas numpy matplotlib seaborn openpyxl

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

print("Imports done")


#%%
# CONFIGURATION — edit paths for your machine
# Where the merged CSV from Step 1 was saved
MERGED_CSV_PATH = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\Merged_AllTimepoints.csv")

# Where to save all Phase 2 outputs
OUT_DIR = Path(r"C:\Users\thakk\Downloads\Emotions survey\outputs\phase2")

# Mid-match is excluded by default
# Set INCLUDE_MID = True to add it as a sensitivity check.
INCLUDE_MID = False

# Emotion outcomes modelled simultaneously
MODEL_OUTCOMES = [
    "Happy", "Sad", "Angry", "Surprised",
    "Scared", "Disgusted", "Valence", "Arousal"
]
N_OUTCOMES = len(MODEL_OUTCOMES)

# TimePoint ordering — Pre is always the reference category
TIMEPOINTS = ["Pre", "Result", "Post"]
if INCLUDE_MID:
    TIMEPOINTS = ["Pre", "Mid", "Result", "Post"]

print("Configuration set")
print(f"  Merged CSV   : {MERGED_CSV_PATH}")
print(f"  Output folder: {OUT_DIR}")
print(f"  Timepoints   : {TIMEPOINTS}  (Pre = reference category)")


#%%
# SECTION 1 — DATA LOADING AND PREPARATION
def load_data() -> pd.DataFrame:
    """
    Load the merged dataset from Step 1 and filter to the required timepoints.
    """
    if not MERGED_CSV_PATH.exists():
        raise FileNotFoundError(
            f"\nMerged CSV not found at:\n  {MERGED_CSV_PATH}\n"
            f"Please run 01_data_prep_descriptives.py first to generate it."
        )

    print(f"  Loading merged dataset...")
    data = pd.read_csv(MERGED_CSV_PATH)
    data["TimePoint_Label"] = pd.Categorical(
        data["TimePoint_Label"],
        categories=["Pre", "Mid", "Result", "Post"],
        ordered=True
    )
    data = data[data["TimePoint_Label"].isin(TIMEPOINTS)].copy()
    print(f"  Loaded {len(data):,} rows, {data['Name'].nunique()} athletes")
    print(f"  Rows per timepoint: {data['TimePoint_Label'].value_counts().to_dict()}")
    return data


def prepare_longitudinal_subset(data: pd.DataFrame, study: str):
    """
    Prepare data for the combined longitudinal model.

    KEY DIFFERENCE FROM PHASE 1:
    Phase 1 kept only one timepoint per model — each athlete appeared once.
    Here we keep ALL timepoints together — the same athlete can appear
    up to 3 times. The model tracks each athlete across time.

    TimePoint is encoded as indicator (dummy) variables:
      Pre    = reference category (no indicator needed, it is the baseline)
      Result = is_result (1 if this row is Result moment, else 0)
      Post   = is_post   (1 if this row is Post moment, else 0)

    All effects at Result and Post are measured as CHANGES FROM Pre.
    """
    result_label = "Win" if study == "winners" else "Loss"
    df = data[data["Result"] == result_label].copy()
    df = df.dropna(subset=MODEL_OUTCOMES)

    # TimePoint indicator variables
    df["is_result"] = (df["TimePoint_Label"] == "Result").astype(float)
    df["is_post"]   = (df["TimePoint_Label"] == "Post").astype(float)
    if INCLUDE_MID:
        df["is_mid"] = (df["TimePoint_Label"] == "Mid").astype(float)

    # Athlete index (fresh 0,1,2,... within this subset)
    athletes = df["Name"].unique()
    ath_map  = {a: i for i, a in enumerate(athletes)}
    df["Ath_idx"] = df["Name"].map(ath_map)

    # Predictor codes
    df["PD_Code"] = (df["PD"] == "High").astype(float)
    df["SB_Code"] = (df["Sighted_Blind"] == "Blind").astype(float)

    # Summary
    print(f"\n  Longitudinal subset: {study.upper()}")
    print(f"  Total rows      : {len(df)}")
    print(f"  Athletes        : {len(athletes)}")
    tp_counts = df.groupby("Name")["TimePoint_Label"].nunique()
    for n in [1, 2, 3]:
        print(f"  Athletes @ {n} TP  : {(tp_counts == n).sum()}")
    print(f"\n  N per group x timepoint:")
    print(df.groupby(["Sighted_Blind", "PD", "TimePoint_Label"])["Name"]
            .count().rename("N").to_string())
    return df, athletes


#%%
# SECTION 2 — BAYESIAN MODEL
def fit_longitudinal_model(
    df: pd.DataFrame,
    athletes: np.ndarray,
    n_samples: int = 2000,
    n_tune: int = 1000,
):
    """
    Fit the Phase 2 combined longitudinal multivariate multilevel model.

    MODEL IN PLAIN LANGUAGE:
    For each athlete observation and each of the 8 emotion outcomes:

        predicted_emotion = personal_baseline_of_this_athlete
                          + how_emotions_change_at_Result (vs Pre)
                          + how_emotions_change_at_Post   (vs Pre)
                          + effect_of_being_High_PD
                          + effect_of_being_Blind
                          + does_PD_effect_CHANGE_at_Result?
                          + does_PD_effect_CHANGE_at_Post?
                          + does_Vision_effect_CHANGE_at_Result?
                          + does_Vision_effect_CHANGE_at_Post?
                          + PD x Vision interaction at Pre
                          + does_the_PD_x_Vision_interaction_CHANGE_at_Result?
                          + does_the_PD_x_Vision_interaction_CHANGE_at_Post?

    WHY THE RANDOM INTERCEPT IS STRONGER HERE THAN IN PHASE 1:
    In Phase 1 each athlete appeared only once so the model could not
    separate natural expressiveness from cultural expressiveness.
    Now each athlete appears at up to 3 timepoints. The personal baseline
    is the part that stays CONSTANT across time for each person.
    PD and Vision are group-level effects that differ BETWEEN people.
    The model can now cleanly separate these two sources of variation.
    """
    try:
        import pymc as pm
    except ImportError:
        raise ImportError(
            "\n\nPyMC is not installed.\n"
            "Run in terminal:  pip install pymc arviz\n"
        )

    n_athletes = len(athletes)

    # Prepare numpy arrays
    Y         = df[MODEL_OUTCOMES].values.astype(float)
    ath_idx   = df["Ath_idx"].values
    pd_code   = df["PD_Code"].values
    sb_code   = df["SB_Code"].values
    is_result = df["is_result"].values
    is_post   = df["is_post"].values

    # Pre-compute all interaction terms for clarity
    pd_result  = pd_code * is_result          # PD x Result
    pd_post    = pd_code * is_post            # PD x Post
    sb_result  = sb_code * is_result          # Vision x Result
    sb_post    = sb_code * is_post            # Vision x Post
    pd_sb      = pd_code * sb_code            # PD x Vision
    pd_sb_res  = pd_code * sb_code * is_result  # 3-way x Result
    pd_sb_post = pd_code * sb_code * is_post    # 3-way x Post

    coords = {
        "athlete":     athletes,
        "outcome":     MODEL_OUTCOMES,
        "observation": np.arange(len(df)),
    }

    with pm.Model(coords=coords) as model:

        # Hyperpriors
        mu_alpha    = pm.Normal("mu_alpha",    mu=0.0, sigma=0.5, dims="outcome")
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.3, dims="outcome")

        # Random intercept per athlete 
        # Much more reliable than Phase 1 because each athlete now has
        # multiple observations. The model learns their true personal
        # baseline from watching them across timepoints.
        alpha_offset = pm.Normal(
            "alpha_offset", mu=0, sigma=1, dims=("athlete", "outcome")
        )
        alpha = pm.Deterministic(
            "alpha",
            mu_alpha + alpha_offset * sigma_alpha,
            dims=("athlete", "outcome")
        )

        # TimePoint main effects 
        # How much do emotions change from Pre to Result / Post on average?
        beta_TP_result = pm.Normal("beta_TP_result", mu=0, sigma=0.5, dims="outcome")
        beta_TP_post   = pm.Normal("beta_TP_post",   mu=0, sigma=0.5, dims="outcome")

        #  Group main effects at Pre-match 
        # Same interpretation as Phase 1 beta_PD, beta_SB, beta_PD_SB
        # but now estimated using the full longitudinal data
        beta_PD    = pm.Normal("beta_PD",    mu=0, sigma=0.5, dims="outcome")
        beta_SB    = pm.Normal("beta_SB",    mu=0, sigma=0.5, dims="outcome")
        beta_PD_SB = pm.Normal("beta_PD_SB", mu=0, sigma=0.3, dims="outcome")

        # PD x TimePoint interactions 
        # Does the PD suppression effect grow stronger or weaker at Result?
        # Negative beta_PD_TP_result means PD suppression INCREASES at Result.
        # Positive means PD suppression DECREASES at Result.
        beta_PD_TP_result = pm.Normal("beta_PD_TP_result", mu=0, sigma=0.3, dims="outcome")
        beta_PD_TP_post   = pm.Normal("beta_PD_TP_post",   mu=0, sigma=0.3, dims="outcome")

        # Vision x TimePoint interactions
        # Does the Blind/Sighted difference emerge or fade across the match?
        beta_SB_TP_result = pm.Normal("beta_SB_TP_result", mu=0, sigma=0.3, dims="outcome")
        beta_SB_TP_post   = pm.Normal("beta_SB_TP_post",   mu=0, sigma=0.3, dims="outcome")

        #3-way interactions: PD x Vision x TimePoint
        # The most interesting parameters — does the PD/Vision interaction
        # pattern change specifically at the emotionally intense moments?
        # A credible effect here would be the key Phase 2 finding.
        beta_PD_SB_TP_result = pm.Normal("beta_PD_SB_TP_result", mu=0, sigma=0.2, dims="outcome")
        beta_PD_SB_TP_post   = pm.Normal("beta_PD_SB_TP_post",   mu=0, sigma=0.2, dims="outcome")

        # ── Residual covariance matrix (LKJ prior) ────────────────────────────
        sd_dist = pm.HalfNormal.dist(sigma=0.5, shape=N_OUTCOMES)
        chol, corr, stds = pm.LKJCholeskyCov(
            "chol_cov", n=N_OUTCOMES, eta=2.0,
            sd_dist=sd_dist, compute_corr=True
        )

        # ── Linear predictor 
        # The [:, None] broadcasts 1D predictors to (N, 8) to match Y shape
        mu = (
            alpha[ath_idx]
            # TimePoint effects
            + beta_TP_result * is_result[:, None]
            + beta_TP_post   * is_post[:, None]
            # Group effects at Pre (baseline)
            + beta_PD        * pd_code[:, None]
            + beta_SB        * sb_code[:, None]
            + beta_PD_SB     * pd_sb[:, None]
            # PD x TimePoint
            + beta_PD_TP_result * pd_result[:, None]
            + beta_PD_TP_post   * pd_post[:, None]
            # Vision x TimePoint
            + beta_SB_TP_result * sb_result[:, None]
            + beta_SB_TP_post   * sb_post[:, None]
            # 3-way: PD x Vision x TimePoint
            + beta_PD_SB_TP_result * pd_sb_res[:, None]
            + beta_PD_SB_TP_post   * pd_sb_post[:, None]
        )

        # ── Likelihood ────────────────────────────────────────────────────────
        pm.MvNormal(
            "obs", mu=mu, chol=chol, observed=Y,
            dims=("observation", "outcome")
        )

        # ── Sample 
        trace = pm.sample(
            n_samples, tune=n_tune,
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
    Extract and save the posterior summary for all model parameters.

    PARAMETERS TO FOCUS ON:
      beta_PD, beta_SB, beta_PD_SB
        Group effects at Pre-match. Comparable to Phase 1 results.

      beta_TP_result, beta_TP_post
        How much do emotions change from Pre to Result/Post on average?
        These should be large and credible — emotions change a lot at Result.

      beta_PD_TP_result, beta_PD_TP_post
        Does the PD effect CHANGE from Pre to Result/Post?
        This is the key longitudinal finding for PD.

      beta_SB_TP_result, beta_SB_TP_post
        Does the Vision effect CHANGE from Pre to Result/Post?

      beta_PD_SB_TP_result, beta_PD_SB_TP_post
        The 3-way interaction. If credible at Result, this is the headline
        finding — PD and Vision interact specifically at the most emotional moment.
    """
    import arviz as az

    var_names = [
        "beta_PD", "beta_SB", "beta_PD_SB",
        "beta_TP_result", "beta_TP_post",
        "beta_PD_TP_result", "beta_PD_TP_post",
        "beta_SB_TP_result", "beta_SB_TP_post",
        "beta_PD_SB_TP_result", "beta_PD_SB_TP_post",
    ]

    present_vars = [v for v in var_names if v in trace.posterior]
    summary = az.summary(trace, var_names=present_vars, hdi_prob=0.94, round_to=4)

    for var in present_vars:
        samples = trace.posterior[var].values
        for i, outcome in enumerate(MODEL_OUTCOMES):
            row_label = f"{var}[{outcome}]"
            if row_label in summary.index:
                s = samples[:, :, i].flatten()
                summary.loc[row_label, "P(>0)"] = round((s > 0).mean(), 3)
                summary.loc[row_label, "P(<0)"] = round((s < 0).mean(), 3)

    summary.to_csv(out_dir / f"posterior_summary_{tag}.csv")
    print(f"\n  Posterior Summary saved.")
    print(summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat",
                   "P(>0)", "P(<0)"]].to_string())
    return summary


#%%
# SECTION 4 — CONTRAST ANALYSIS

def compute_longitudinal_contrasts(trace, tag: str, out_dir: Path) -> pd.DataFrame:
    """
    Compute the 4 group contrasts at each of the 3 timepoints.

    This is the core Phase 2 result table. Same 4 contrasts as Phase 1,
    but now computed at Pre, Result, AND Post so you can see how they
    evolve across the match.

    HOW THE MATH WORKS:

    At Pre (reference — all interaction terms are zero):
      High vs Low PD | Sighted    = beta_PD
      High vs Low PD | Blind      = beta_PD + beta_PD_SB
      Blind vs Sighted | Low PD   = beta_SB
      Blind vs Sighted | High PD  = beta_SB + beta_PD_SB

    At Result (add the TP interaction terms):
      High vs Low PD | Sighted    = beta_PD + beta_PD_TP_result
      High vs Low PD | Blind      = beta_PD + beta_PD_SB
                                    + beta_PD_TP_result + beta_PD_SB_TP_result
      Blind vs Sighted | Low PD   = beta_SB + beta_SB_TP_result
      Blind vs Sighted | High PD  = beta_SB + beta_PD_SB
                                    + beta_SB_TP_result + beta_PD_SB_TP_result

    At Post: same pattern but with _post parameters
    """
    import arviz as az

    post = trace.posterior
    rows = []

    # Define what parameters to add at each timepoint
    timepoint_params = {
        "Pre": {
            "pd_tp": None, "sb_tp": None, "pd_sb_tp": None
        },
        "Result": {
            "pd_tp": "beta_PD_TP_result",
            "sb_tp": "beta_SB_TP_result",
            "pd_sb_tp": "beta_PD_SB_TP_result"
        },
        "Post": {
            "pd_tp": "beta_PD_TP_post",
            "sb_tp": "beta_SB_TP_post",
            "pd_sb_tp": "beta_PD_SB_TP_post"
        },
    }

    for tp_name, params in timepoint_params.items():
        for i, outcome in enumerate(MODEL_OUTCOMES):

            # Base parameters at Pre-match
            b_PD    = post["beta_PD"].values[:, :, i].flatten()
            b_SB    = post["beta_SB"].values[:, :, i].flatten()
            b_PD_SB = post["beta_PD_SB"].values[:, :, i].flatten()

            # TimePoint adjustments (zeros at Pre)
            zeros = np.zeros(len(b_PD))
            b_PD_TP    = post[params["pd_tp"]].values[:, :, i].flatten() if params["pd_tp"] else zeros
            b_SB_TP    = post[params["sb_tp"]].values[:, :, i].flatten() if params["sb_tp"] else zeros
            b_PD_SB_TP = post[params["pd_sb_tp"]].values[:, :, i].flatten() if params["pd_sb_tp"] else zeros

            contrasts = {
                "High vs Low PD | Sighted":   b_PD + b_PD_TP,
                "High vs Low PD | Blind":     b_PD + b_PD_SB + b_PD_TP + b_PD_SB_TP,
                "Blind vs Sighted | Low PD":  b_SB + b_SB_TP,
                "Blind vs Sighted | High PD": b_SB + b_PD_SB + b_SB_TP + b_PD_SB_TP,
            }

            for contrast_name, samples in contrasts.items():
                hdi   = az.hdi(samples, hdi_prob=0.94)
                p_pos = (samples > 0).mean()
                rows.append({
                    "TimePoint":  tp_name,
                    "Outcome":    outcome,
                    "Contrast":   contrast_name,
                    "Mean":       round(float(samples.mean()), 4),
                    "SD":         round(float(samples.std()), 4),
                    "HDI_low":    round(float(hdi[0]), 4),
                    "HDI_high":   round(float(hdi[1]), 4),
                    "P(>0)":      round(float(p_pos), 3),
                    "Direction":  "Positive" if p_pos > 0.5 else "Negative",
                    "HDI_excl_0": "Yes" if (hdi[0] > 0 or hdi[1] < 0) else "No",
                })

    contrast_df = pd.DataFrame(rows)
    contrast_df.to_csv(out_dir / f"contrasts_{tag}.csv", index=False)

    print(f"\n  Contrasts saved.")
    print(f"\n  {'TP':<8} {'Outcome':<12} {'Contrast':<35} "
          f"{'Mean':>7} {'HDI_low':>9} {'HDI_high':>9} {'P(>0)':>7} {'*':>4}")
    print("  " + "-" * 92)
    for _, row in contrast_df.iterrows():
        star = "v" if row["HDI_excl_0"] == "Yes" else ""
        print(
            f"  {row['TimePoint']:<8} {row['Outcome']:<12} {row['Contrast']:<35} "
            f"{row['Mean']:>7.4f} {row['HDI_low']:>9.4f} "
            f"{row['HDI_high']:>9.4f} {row['P(>0)']:>7.3f} {star:>4}"
        )

    return contrast_df


#%%
# SECTION 5 — VISUALISATIONS

def plot_trajectory_posteriors(trace, tag: str, out_dir: Path, study: str):
    """
    Model-estimated emotion trajectories across timepoints for each group.

    This is the key Phase 2 figure. Unlike the descriptive trajectory plots
    from Step 1 (which just showed raw means), these trajectories come from
    the model posterior and include proper Bayesian uncertainty bands.

    They show the full emotional arc Pre -> Result -> Post for each group
    with 94% credible intervals.
    """
    import arviz as az

    study_label = "Winners" if study == "winners" else "Losers"
    tp_labels   = ["Pre", "Result", "Post"]

    groups = {
        "High PD - Blind":    (1, 1),
        "High PD - Sighted":  (1, 0),
        "Low PD - Blind":     (0, 1),
        "Low PD - Sighted":   (0, 0),
    }
    palette = {
        "High PD - Blind":    "#D55E00",
        "High PD - Sighted":  "#E69F00",
        "Low PD - Blind":     "#0072B2",
        "Low PD - Sighted":   "#56B4E9",
    }

    outcomes_to_plot = ["Valence", "Arousal", "Happy", "Sad"]
    fig, axes = plt.subplots(len(outcomes_to_plot), 1,
                             figsize=(10, 5 * len(outcomes_to_plot)))

    post = trace.posterior

    for ax, outcome in zip(axes, outcomes_to_plot):
        i = MODEL_OUTCOMES.index(outcome)

        for grp_name, (pd_val, sb_val) in groups.items():

            b_PD    = post["beta_PD"].values[:, :, i].flatten()
            b_SB    = post["beta_SB"].values[:, :, i].flatten()
            b_PD_SB = post["beta_PD_SB"].values[:, :, i].flatten()
            mu_alph = post["mu_alpha"].values[:, :, i].flatten()

            means_per_tp = []
            hdis_per_tp  = []

            for tp in tp_labels:
                if tp == "Pre":
                    pred = (mu_alph
                            + b_PD * pd_val
                            + b_SB * sb_val
                            + b_PD_SB * pd_val * sb_val)
                elif tp == "Result":
                    b_TP    = post["beta_TP_result"].values[:, :, i].flatten()
                    b_PD_TP = post["beta_PD_TP_result"].values[:, :, i].flatten()
                    b_SB_TP = post["beta_SB_TP_result"].values[:, :, i].flatten()
                    b_3way  = post["beta_PD_SB_TP_result"].values[:, :, i].flatten()
                    pred = (mu_alph + b_TP
                            + b_PD * pd_val + b_SB * sb_val + b_PD_SB * pd_val * sb_val
                            + b_PD_TP * pd_val + b_SB_TP * sb_val
                            + b_3way * pd_val * sb_val)
                else:  # Post
                    b_TP    = post["beta_TP_post"].values[:, :, i].flatten()
                    b_PD_TP = post["beta_PD_TP_post"].values[:, :, i].flatten()
                    b_SB_TP = post["beta_SB_TP_post"].values[:, :, i].flatten()
                    b_3way  = post["beta_PD_SB_TP_post"].values[:, :, i].flatten()
                    pred = (mu_alph + b_TP
                            + b_PD * pd_val + b_SB * sb_val + b_PD_SB * pd_val * sb_val
                            + b_PD_TP * pd_val + b_SB_TP * sb_val
                            + b_3way * pd_val * sb_val)

                hdi = az.hdi(pred, hdi_prob=0.94)
                means_per_tp.append(float(pred.mean()))
                hdis_per_tp.append(hdi)

            color  = palette[grp_name]
            tp_x   = np.arange(len(tp_labels))
            means  = np.array(means_per_tp)
            lowers = np.array([h[0] for h in hdis_per_tp])
            uppers = np.array([h[1] for h in hdis_per_tp])

            ax.plot(tp_x, means, marker="o", color=color,
                    label=grp_name, linewidth=2.5, markersize=8, zorder=3)
            ax.fill_between(tp_x, lowers, uppers,
                            alpha=0.15, color=color, zorder=2)

        ax.set_xticks(np.arange(len(tp_labels)))
        ax.set_xticklabels(tp_labels, fontsize=11)
        ax.set_ylabel(outcome, fontsize=11)
        ax.set_title(outcome, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        if outcome == "Valence":
            ax.axhline(0, color="grey", linestyle=":", linewidth=1, alpha=0.6)
        ax.legend(fontsize=9, loc="best", framealpha=0.85,
                  title="Group", title_fontsize=9)

    fig.suptitle(
        f"Model-Estimated Emotion Trajectories - {study_label}\n"
        f"Posterior mean +/- 94% credible interval",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"trajectories_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Trajectory plot saved -> {fpath.name}")


def plot_forest_longitudinal(trace, tag: str, out_dir: Path, study: str):
    """
    Forest plot for all Phase 2 parameters.

    Arranged in a 3x3 grid showing:
      Row 1: Group effects at Pre (same as Phase 1)
      Row 2: TimePoint main effects + PD x TP interactions
      Row 3: Vision x TP interactions + 3-way interactions

    The Phase 2-specific findings are in rows 2 and 3 — these show whether
    group differences GROW or SHRINK as the match progresses.
    """
    import arviz as az

    study_label = "Winners" if study == "winners" else "Losers"

    param_groups = {
        "PD effect at Pre":          "beta_PD",
        "Vision effect at Pre":      "beta_SB",
        "PD x Vision at Pre":        "beta_PD_SB",
        "Pre->Result change":        "beta_TP_result",
        "Pre->Post change":          "beta_TP_post",
        "PD x Result interaction":   "beta_PD_TP_result",
        "PD x Post interaction":     "beta_PD_TP_post",
        "Vision x Result interact.": "beta_SB_TP_result",
        "3-way x Result":            "beta_PD_SB_TP_result",
    }

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes_flat = axes.flatten()

    for ax, (param_label, var) in zip(axes_flat, param_groups.items()):
        if var not in trace.posterior:
            ax.set_visible(False)
            continue

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

        ax.barh(y_pos, means, color=colors, alpha=0.6, height=0.55)
        ax.errorbar(means, y_pos,
                    xerr=[means - hdis[:, 0], hdis[:, 1] - means],
                    fmt="none", color="black", capsize=3, linewidth=1.2)
        ax.scatter(means, y_pos, color=colors, zorder=5, s=40)
        ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(MODEL_OUTCOMES, fontsize=8)
        ax.set_title(param_label, fontsize=9, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        for j in range(N_OUTCOMES):
            if hdis[j, 0] > 0 or hdis[j, 1] < 0:
                ax.axhspan(j - 0.4, j + 0.4, alpha=0.05, color="#D55E00")

    fig.suptitle(
        f"Phase 2 Posterior Effects - {study_label}\n"
        f"Orange = 94% HDI excludes zero  |  'interaction' = change from Pre",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"forest_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Forest plot saved -> {fpath.name}")


def plot_contrasts_across_time(contrast_df: pd.DataFrame, tag: str,
                                out_dir: Path, study: str):
    """
    Line plots showing how each of the 4 contrasts evolves over time.

    This is the Phase 2 signature figure. It shows directly whether group
    differences grow or shrink from Pre to Result to Post.

    Layout: rows = emotion outcomes, columns = the 4 contrasts
    * marks = credible effects at that timepoint
    """
    study_label  = "Winners" if study == "winners" else "Losers"
    tp_order     = ["Pre", "Result", "Post"]
    contrast_list = contrast_df["Contrast"].unique()
    outcomes_to_plot = ["Valence", "Happy", "Sad", "Arousal"]

    contrast_colors = {
        "High vs Low PD | Sighted":   "#E69F00",
        "High vs Low PD | Blind":     "#D55E00",
        "Blind vs Sighted | Low PD":  "#56B4E9",
        "Blind vs Sighted | High PD": "#0072B2",
    }

    fig, axes = plt.subplots(
        len(outcomes_to_plot), len(contrast_list),
        figsize=(5 * len(contrast_list), 4 * len(outcomes_to_plot)),
        sharey="row"
    )

    for row_i, outcome in enumerate(outcomes_to_plot):
        for col_i, contrast in enumerate(contrast_list):
            ax = axes[row_i, col_i]
            sub = contrast_df[
                (contrast_df["Outcome"]  == outcome) &
                (contrast_df["Contrast"] == contrast)
            ].copy()
            sub["TimePoint"] = pd.Categorical(
                sub["TimePoint"], categories=tp_order, ordered=True
            )
            sub = sub.sort_values("TimePoint")

            color = contrast_colors.get(contrast, "#666666")
            tp_x  = np.arange(len(sub))

            ax.plot(tp_x, sub["Mean"].values, marker="o", color=color,
                    linewidth=2.5, markersize=8)
            ax.fill_between(tp_x,
                            sub["HDI_low"].values, sub["HDI_high"].values,
                            alpha=0.15, color=color)

            # Star for credible effects
            for j, (_, row) in enumerate(sub.iterrows()):
                if row["HDI_excl_0"] == "Yes":
                    y_star = row["HDI_high"] + abs(row["HDI_high"]) * 0.08
                    ax.annotate("*", (j, y_star), ha="center",
                                fontsize=14, color=color, fontweight="bold")

            ax.axhline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_xticks(tp_x)
            ax.set_xticklabels(sub["TimePoint"].tolist(), fontsize=9)
            ax.grid(axis="y", alpha=0.3, linestyle="--")

            if row_i == 0:
                ax.set_title(contrast.replace(" | ", "\n| "),
                             fontsize=9, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(outcome, fontsize=10)

    fig.suptitle(
        f"Group Contrast Trajectories - {study_label}\n"
        f"* = 94% HDI excludes zero at that timepoint",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"contrast_trajectories_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Contrast trajectories plot saved -> {fpath.name}")


def plot_contrasts_heatmap(contrast_df: pd.DataFrame, tag: str,
                            out_dir: Path, study: str):
    """
    Summary heatmap of contrasts — one panel per timepoint.
    * marks credible effects.
    """
    study_label  = "Winners" if study == "winners" else "Losers"
    tp_list      = ["Pre", "Result", "Post"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, tp in zip(axes, tp_list):
        sub = contrast_df[contrast_df["TimePoint"] == tp]
        if sub.empty:
            ax.set_visible(False)
            continue

        pivot    = sub.pivot(index="Outcome", columns="Contrast", values="Mean")
        credible = sub.pivot(index="Outcome", columns="Contrast", values="HDI_excl_0")

        annot_text = pivot.copy().astype(str)
        for r in credible.index:
            for c in credible.columns:
                val = f"{pivot.loc[r, c]:.3f}"
                if credible.loc[r, c] == "Yes":
                    val += " *"
                annot_text.loc[r, c] = val

        vmax = contrast_df["Mean"].abs().max()
        sns.heatmap(
            pivot, annot=annot_text, fmt="",
            cmap="RdBu_r", center=0, ax=ax,
            linewidths=0.5, annot_kws={"size": 8},
            vmin=-vmax, vmax=vmax,
            cbar_kws={"label": "Posterior Mean Effect", "shrink": 0.8}
        )
        ax.set_title(tp, fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(
        f"Group Contrasts by TimePoint - {study_label}\n"
        f"* = 94% HDI excludes zero",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    fpath = out_dir / f"contrasts_heatmap_{tag}.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Contrasts heatmap saved -> {fpath.name}")


def plot_trace(trace, tag: str, out_dir: Path):
    """MCMC convergence check — chains should look like fuzzy caterpillars."""
    import arviz as az

    az.plot_trace(
        trace,
        var_names=["beta_PD", "beta_SB", "beta_PD_TP_result", "beta_PD_TP_post"],
        compact=True, figsize=(12, 10)
    )
    plt.suptitle(
        f"MCMC Trace Plots - {tag}\n"
        f"Good = chains overlap and look like fuzzy caterpillars",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    fpath = out_dir / f"trace_{tag}.png"
    plt.savefig(fpath, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Trace plot saved -> {fpath.name}")


#%%
# SECTION 6 — RUN ONE COMPLETE MODEL

def run_phase2_model(
    study: str,
    n_samples: int = 2000,
    n_tune: int = 1000,
    out_dir: Path = OUT_DIR,
):
    """
    Run the full Phase 2 pipeline for one study (winners or losers).

    Steps:
      1. Load and prepare longitudinal data (all timepoints together)
      2. Fit the combined Bayesian model
      3. Check convergence (R-hat)
      4. Posterior summary
      5. Contrast analysis at each timepoint
      6. All plots
      7. Save full trace to .nc file
    """
    import arviz as az

    tag = f"longitudinal_{study}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  PHASE 2 MODEL: {study.upper()}")
    print(f"  samples={n_samples}, tune={n_tune}")
    print(f"  Timepoints: {TIMEPOINTS}  (Pre = reference)")
    print(f"{'='*60}")

    # Step 1: Load and prepare
    data = load_data()
    df, athletes = prepare_longitudinal_subset(data, study)

    if len(df) < 50:
        print(f"  Too few observations ({len(df)}) — skipping.")
        return None

    # Step 2: Fit
    trace, model = fit_longitudinal_model(
        df, athletes, n_samples=n_samples, n_tune=n_tune
    )

    # Step 3: Convergence check
    check_vars = ["beta_PD", "beta_SB", "beta_PD_TP_result", "beta_PD_TP_post"]
    rhat = az.rhat(trace, var_names=check_vars)
    max_rhat = float(max(rhat[v].values.max() for v in check_vars))
    if max_rhat > 1.05:
        print(f"\n  WARNING: max R-hat = {max_rhat:.3f}")
        print(f"    Consider re-running with higher tune steps (e.g. n_tune=2000)")
    else:
        print(f"\n  Convergence OK  (max R-hat = {max_rhat:.3f})")

    # Steps 4-7: Outputs
    posterior_summary(trace, tag, out_dir)
    contrast_df = compute_longitudinal_contrasts(trace, tag, out_dir)
    plot_trajectory_posteriors(trace, tag, out_dir, study)
    plot_forest_longitudinal(trace, tag, out_dir, study)
    plot_contrasts_across_time(contrast_df, tag, out_dir, study)
    plot_contrasts_heatmap(contrast_df, tag, out_dir, study)
    plot_trace(trace, tag, out_dir)

    trace_path = out_dir / f"trace_{tag}.nc"
    trace.to_netcdf(str(trace_path))
    print(f"  Full trace saved -> {trace_path.name}")
    print(f"\n  Done: {tag}")
    return trace


#%%
# QUICK TEST RUN — run this cell first to check everything works
# Runs Winners only with 200 samples.
# Takes about 10-20 minutes.
# Check that output files appear — if yes, proceed to full run.

def test_run():
    print("\n" + "="*60)
    print("  PHASE 2 - QUICK TEST RUN")
    print("  Study     : winners")
    print("  Samples   : 200  (just checking model runs)")
    print("  Tune      : 200")
    print("="*60)

    trace = run_phase2_model(
        study     = "winners",
        n_samples = 200,
        n_tune    = 200,
        out_dir   = OUT_DIR / "test",
    )

    if trace is not None:
        print("\n" + "="*60)
        print("TEST PASSED - model ran successfully")
        print(f"   Check outputs at: {OUT_DIR / 'test'}")
        print("   If everything looks right, run the FULL RUN cell below")
        print("="*60)
    else:
        print("\nTEST FAILED - check the error messages above")


test_run()


#%%
# FULL RUN — run after test passes
# Runs both Winners and Losers with 2000 samples each.
# Phase 2 is slightly slower than Phase 1 — more data, more parameters.
# Budget roughly 1-2 hours total.

def full_run(
    studies   = ["winners", "losers"],
    n_samples = 2000,
    n_tune    = 1000,
):
    print("\n" + "="*60)
    print("  PHASE 2 - FULL LONGITUDINAL ANALYSIS")
    print(f"  Studies    : {studies}")
    print(f"  Timepoints : {TIMEPOINTS}")
    print(f"  Samples    : {n_samples} per chain")
    print(f"  Output dir : {OUT_DIR}")
    print("="*60)

    results = {}
    for study in studies:
        trace = run_phase2_model(
            study=study, n_samples=n_samples, n_tune=n_tune, out_dir=OUT_DIR
        )
        results[study] = trace

    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE")
    print(f"Results saved in: {OUT_DIR}")
    print(f"\nFor each study:")
    print(f"  posterior_summary_longitudinal_[study].csv")
    print(f"  contrasts_longitudinal_[study].csv")
    print(f"  trajectories_longitudinal_[study].png")
    print(f"  forest_longitudinal_[study].png")
    print(f"  contrast_trajectories_longitudinal_[study].png")
    print(f"  contrasts_heatmap_longitudinal_[study].png")
    print(f"  trace_longitudinal_[study].png")
    print(f"  trace_longitudinal_[study].nc")
    print(f"{'='*60}\n")
    return results


full_run()


#%%
# =============================================================================
# RUN ONE STUDY ONLY
# =============================================================================

# STUDY    = "winners"   # or "losers"
# SAMPLES  = 2000
# TUNE     = 1000

# trace = run_phase2_model(
#     study     = STUDY,
#     n_samples = SAMPLES,
#     n_tune    = TUNE,
#     out_dir   = OUT_DIR,
# )
