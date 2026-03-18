# Athlete Facial Emotion Analysis
### Multivariate Multilevel Bayesian Regression
**Facial Emotional Expressions in Olympic and Paralympic/IBSA Athletes**

---

## What This Study Is About

This repository contains the full analysis pipeline for a study examining how athletes' facial expressions differ across different moments of a competition, and whether those patterns are shaped by two key factors:

**Power Distance (PD)** — a cultural dimension from Hofstede's framework, where countries are classified as High PD (score > 50) or Low PD (score < 50). High PD cultures tend to have stronger norms around emotional suppression in formal and competitive settings.

**Vision Status** — athletes are classified as Sighted (Olympic / World Championship competitions) or Blind/Visually Impaired (Paralympic / IBSA competitions).

Facial emotion data was automatically extracted from competition video footage using **FaceReader** (Noldus Information Technology) at four match situations:

| TimePoint | Description |
|---|---|
| Pre | Before the contest |
| Mid | During the contest |
| Result | The instant the outcome is revealed |
| Post | After the contest settles |

---

## Study Design

### Study 1 — Winners
| # | Comparison |
|---|---|
| 1 | Only Sighted winners: Low PD vs. High PD (at each match situation) |
| 2 | Only Blind winners: Low PD vs. High PD (at each match situation) |
| 3 | Only Low PD winners: Sighted vs. Blind (at each match situation) |
| 4 | Only High PD winners: Sighted vs. Blind (at each match situation) |

### Study 2 — Losers
| # | Comparison |
|---|---|
| 1 | Only Sighted losers: Low PD vs. High PD (at each match situation) |
| 2 | Only Blind losers: Low PD vs. High PD (at each match situation) |
| 3 | Only Low PD losers: Sighted vs. Blind (at each match situation) |
| 4 | Only High PD losers: Sighted vs. Blind (at each match situation) |

---

## Repository Structure

```
emotion-analysis-firsttest/
│
├── README.md                        ← You are here
├── requirements.txt                 ← Python dependencies
│
├── 01_data_prep_descriptives.py     ← Step 1: merge data, descriptive stats, all plots
├── 02_phase1_crosssectional.py      ← Step 2: Bayesian models (one per timepoint)
│
└── outputs/
    ├── Merged_AllTimepoints.csv          ← Full merged dataset (all 4 timepoints)
    ├── descriptive/                      ← Tables and plots from Step 1
    │   ├── Table_Full_Descriptives.xlsx
    │   ├── Table_Means_Winners.xlsx
    │   ├── Table_Means_Losers.xlsx
    │   ├── Table_Sample_Sizes.xlsx
    │   ├── Trajectory_Valence.png
    │   ├── Trajectory_Arousal.png
    │   ├── Trajectory_Happy.png
    │   ├── Trajectory_Sad.png
    │   ├── Trajectory_Angry.png
    │   ├── Boxplot_Valence.png
    │   ├── Boxplot_Happy.png
    │   ├── Correlation_Heatmap.png
    │   └── Sample_Sizes.png
    └── phase1/
        └── pre/                          ← Test run results (Pre | Winners, 200 samples)
            ├── posterior_summary_Pre_winners.csv
            ├── contrasts_Pre_winners.csv
            ├── forest_Pre_winners.png
            ├── contrasts_heatmap_Pre_winners.png
            ├── correlation_matrix_Pre_winners.png
            ├── trace_Pre_winners.png
            └── trace_Pre_winners.nc
```

> **Note:** The raw Excel data files are not included in this repository as they contain unpublished research data. Place the four FaceReader Excel files in a `data/` folder before running the scripts.

---

## Analysis Pipeline

### Phase 1 — Cross-Sectional Models (this repository)

A **multivariate multilevel Bayesian regression model** is fitted separately for each timepoint (Pre, Result, Post). Within each timepoint, two models are run — one for Winners only, one for Losers only. The mid-match timepoint is excluded from the primary analysis due to substantially more missing data compared to other timepoints, but can be included as a sensitivity check.

All 8 emotion outcomes are modelled **simultaneously in one model**:
`Happy, Sad, Angry, Surprised, Scared, Disgusted, Valence, Arousal`

From each model, four contrasts are extracted — one for each of Saumya's study comparisons.

### Phase 2 — Combined Longitudinal Model (future work)

One single multivariate multilevel Bayesian model with Pre + Result + Post together, with TimePoint as a within-subject factor. This will answer the question of whether emotional trajectories across the match differ between groups, and will provide cleaner separation of individual-level and group-level effects.

---

## The Model

For each timepoint and study subset, the model is:

```
[Happy, Sad, Angry, Surprised, Scared, Disgusted, Valence, Arousal]
  ~ MultivariateNormal(μ, Σ)

μ = α_athlete        ← random intercept per athlete (multilevel structure)
  + β_PD  × PD       ← High vs Low PD effect
  + β_SB  × Blind    ← Blind vs Sighted effect
  + β_PD_SB × PD × Blind   ← interaction

α_athlete ~ Normal(μ_α, σ_α)     ← athlete-level random intercept
Σ ~ LKJCholeskyCov(η=2)          ← residual covariance across all 8 outcomes
```

**Why multilevel?** The same athlete can appear at multiple timepoints. The random intercept per athlete accounts for the fact that some people naturally have more expressive faces than others, separating this individual-level variation from the group-level cultural effects we're interested in.

**Why multivariate?** Emotions are correlated — Happy and Valence move together, Sad and Happy move in opposite directions. Modelling all 8 outcomes simultaneously captures these correlations and produces more statistically efficient estimates than running 8 separate models.

### Priors
```
β_PD, β_SB  ~ Normal(0, 0.5)    — weakly informative
β_PD_SB     ~ Normal(0, 0.3)    — slightly more regularised for the interaction
σ_α         ~ HalfNormal(0.3)
Σ           ~ LKJCholeskyCov(η=2)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place data files
Create a `data/` folder in the same directory as the scripts and add the four Excel files:
```
data/
├── Complete_PreMatchFaceReader_PROCESSED_Feb23_20260223_121519.xlsx
├── Complete_FaceReader_Midmatch_Feb19.xlsx
├── Complete_ResultMoment_FaceReader_Data_Feb23.xlsx
└── Complete_FaceReader_PostMatch_Feb19.xlsx
```

### 3. Update file paths
Open both scripts and update the `DATA_DIR` and `OUT_DIR` variables at the top to match where your files are saved.

### 4. Run Step 1 — Data preparation and descriptives
```bash
python 01_data_prep_descriptives.py
```
This generates all descriptive statistics and trajectory plots. **Does not require PyMC — runs immediately.**

### 5. Run Step 2 — Phase 1 Bayesian models

First run a quick test to verify everything works (5–10 minutes):
```python
# In the script, uncomment this line in the TEST RUN cell:
test_run()
```

Then run the full analysis (2–4 hours):
```python
# In the script, uncomment this line in the FULL RUN cell:
full_run()
```

---

## Outputs

Each model run produces the following files:

| File | Description |
|---|---|
| `posterior_summary_[tp]_[study].csv` | Effect size, SD, 94% HDI, R-hat, P(>0) for every parameter × outcome |
| `contrasts_[tp]_[study].csv` | The 4 group comparisons with posterior means and credible intervals |
| `forest_[tp]_[study].png` | Visual of all effects — orange bars have HDI excluding zero |
| `contrasts_heatmap_[tp]_[study].png` | Colour-coded grid of all contrasts × outcomes, * = credible effect |
| `correlation_matrix_[tp]_[study].png` | Model-estimated residual emotion correlations |
| `trace_[tp]_[study].png` | MCMC convergence diagnostics (look for fuzzy caterpillars) |
| `trace_[tp]_[study].nc` | Full posterior trace saved to disk (reloadable with ArviZ) |

---

## Interpreting Results

**Posterior mean** — the most likely value of the effect

**94% HDI** — we are 94% confident the true effect lies within this interval. If the interval excludes zero, the effect is credibly non-zero.

**P(>0)** — probability the effect is positive. Values near 1.0 or 0.0 indicate a clear directional effect. Values near 0.5 mean no clear direction.

**R-hat** — convergence check. Should be very close to 1.0. Values above 1.05 indicate the model may not have converged and results should be treated with caution.

### Example reading
```
beta_PD[Valence]:  mean = -0.134,  HDI [-0.200, -0.067],  P(>0) = 0.000
```
→ High PD athletes show credibly lower Valence than Low PD athletes.
The HDI is entirely negative and P(>0) = 0, meaning there is essentially zero probability the true effect is positive. Strong evidence for emotional suppression in High PD athletes.

---

## Preliminary Findings (Test Run — Pre-match, Winners, 200 samples)

These results are from the test run and should be interpreted cautiously due to the low sample count. The full run with 2000 samples will produce more reliable estimates.

**Power Distance is the dominant pre-match factor.** High PD winners show credibly lower Happy (−0.065) and Valence (−0.134), and credibly higher Sad (+0.043) and Angry (+0.035) compared to Low PD winners before the match even begins. This is consistent with Hofstede's cultural display rule framework — athletes from high power distance cultures arrive at competition with more suppressed positive facial expression.

**Vision status matters less Pre-match.** The Blind vs Sighted comparison shows only two credible effects Pre-match: Blind athletes show slightly higher Arousal (+0.033) and lower Surprised (−0.012). Most other comparisons do not reach credibility, suggesting that the Blind/Sighted difference is more likely to emerge at the emotionally intense Result moment.

**No PD × Vision interaction Pre-match.** The interaction term is flat and centred on zero for all outcomes, meaning PD and Vision operate independently before the match.

---

## Dependencies

```
pymc >= 5.0
arviz >= 0.17
pandas >= 1.5
numpy >= 1.23
matplotlib >= 3.6
seaborn >= 0.12
openpyxl >= 3.0
scipy >= 1.9
```

---

For questions about the analysis code, open an issue in this repository.
