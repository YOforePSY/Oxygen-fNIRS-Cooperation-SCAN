"""
ROI-specificity analysis.

This script runs the same moderation model across multiple ROI-level
cooperation-related IBS change variables and summarizes the interaction term.

Expected input:
    data/processed_main_analysis_data.csv
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed_main_analysis_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

roi_vars = [
    "Coop_mPFC_Change_z",
    "Coop_L_DLPFC_Change_z",
    "Coop_R_DLPFC_Change_z",
    "Coop_L_vlPFC_Change_z",
    "Coop_R_vlPFC_Change_z",
    "Coop_L_STC_Change_z",
    "Coop_R_STC_Change_z",
    "Coop_L_SFC_Change_z",
    "Coop_R_SFC_Change_z",
]

roi_vars = [v for v in roi_vars if v in df.columns]

if "Trust_mean_z" not in df.columns:
    raise ValueError("Trust_mean_z is required.")

outcomes = ["success_gain_z", "efficiency_gain_z"]
rows = []

for outcome in outcomes:
    if outcome not in df.columns:
        continue

    for roi in roi_vars:
        formula = f"{outcome} ~ {roi} * Trust_mean_z + age_mean_z + sex_code + order_group"
        model = smf.ols(formula, data=df).fit()

        interaction_term = f"{roi}:Trust_mean_z"
        if interaction_term not in model.params.index:
            # statsmodels may reverse the term order depending on formula parsing
            interaction_term = f"Trust_mean_z:{roi}"

        rows.append({
            "outcome": outcome,
            "roi": roi,
            "interaction_term": interaction_term,
            "beta_interaction": model.params.get(interaction_term, float("nan")),
            "se": model.bse.get(interaction_term, float("nan")),
            "t": model.tvalues.get(interaction_term, float("nan")),
            "p": model.pvalues.get(interaction_term, float("nan")),
            "ci_low": model.conf_int().loc[interaction_term, 0],
            "ci_high": model.conf_int().loc[interaction_term, 1],
            "adj_r_squared": model.rsquared_adj,
        })

results = pd.DataFrame(rows)

# FDR correction within each outcome
results["fdr_q"] = float("nan")
for outcome in results["outcome"].unique():
    mask = results["outcome"] == outcome
    pvals = results.loc[mask, "p"].values
    _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
    results.loc[mask, "fdr_q"] = qvals

results.to_csv(OUTPUT_DIR / "roi_specificity_interactions.csv", index=False)

print("ROI-specificity models completed.")
print(results)
