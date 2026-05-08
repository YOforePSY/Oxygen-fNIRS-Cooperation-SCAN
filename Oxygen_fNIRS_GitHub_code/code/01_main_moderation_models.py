"""
Main moderation models for the HBOT-fNIRS cooperation study.

This script estimates whether interpersonal trust moderates the association
between cooperation-related mPFC inter-brain synchrony change and
hyperbaric-oxygen-related cooperation gains.

Expected input:
    data/processed_main_analysis_data.csv

Expected key variables:
    dyad_id
    Trust_mean_z
    Coop_mPFC_Change_z
    success_gain_z
    efficiency_gain_z
    age_mean_z
    sex_code
    order_group
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed_main_analysis_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = [
    "Trust_mean_z",
    "Coop_mPFC_Change_z",
    "success_gain_z",
    "efficiency_gain_z",
    "age_mean_z",
    "sex_code",
    "order_group",
]
missing = [v for v in required if v not in df.columns]
if missing:
    raise ValueError(f"Missing required variables: {missing}")

# Interaction term
df["mPFC_x_Trust"] = df["Coop_mPFC_Change_z"] * df["Trust_mean_z"]

models = {
    "success_gain": "success_gain_z ~ Coop_mPFC_Change_z * Trust_mean_z + age_mean_z + sex_code + order_group",
    "efficiency_gain": "efficiency_gain_z ~ Coop_mPFC_Change_z * Trust_mean_z + age_mean_z + sex_code + order_group",
}

rows = []
for name, formula in models.items():
    model = smf.ols(formula, data=df).fit()
    result_path = OUTPUT_DIR / f"model_{name}.txt"
    result_path.write_text(model.summary().as_text(), encoding="utf-8")

    for term in model.params.index:
        rows.append({
            "model": name,
            "term": term,
            "beta": model.params[term],
            "se": model.bse[term],
            "t": model.tvalues[term],
            "p": model.pvalues[term],
            "ci_low": model.conf_int().loc[term, 0],
            "ci_high": model.conf_int().loc[term, 1],
            "r_squared": model.rsquared,
            "adj_r_squared": model.rsquared_adj,
        })

summary = pd.DataFrame(rows)
summary.to_csv(OUTPUT_DIR / "main_moderation_model_summary.csv", index=False)

print("Main moderation models completed.")
print(summary)
