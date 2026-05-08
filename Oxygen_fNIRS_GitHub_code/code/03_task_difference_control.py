"""
Task-difference control analysis.

This script tests whether a task-difference IBS change variable explains the
same cooperation-gain outcomes. It is intended as a control analysis to show
that the main brain-behavior association is specific to cooperation-related IBS.
"""

from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed_main_analysis_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = ["TaskDiff_mPFC_Change_z", "Trust_mean_z", "age_mean_z", "sex_code", "order_group"]
missing = [v for v in required if v not in df.columns]
if missing:
    raise ValueError(f"Missing required variables: {missing}")

outcomes = ["success_gain_z", "efficiency_gain_z"]
rows = []

for outcome in outcomes:
    if outcome not in df.columns:
        continue

    formula = f"{outcome} ~ TaskDiff_mPFC_Change_z * Trust_mean_z + age_mean_z + sex_code + order_group"
    model = smf.ols(formula, data=df).fit()
    interaction_term = "TaskDiff_mPFC_Change_z:Trust_mean_z"
    if interaction_term not in model.params.index:
        interaction_term = "Trust_mean_z:TaskDiff_mPFC_Change_z"

    rows.append({
        "outcome": outcome,
        "predictor": "TaskDiff_mPFC_Change_z",
        "beta_interaction": model.params[interaction_term],
        "se": model.bse[interaction_term],
        "t": model.tvalues[interaction_term],
        "p": model.pvalues[interaction_term],
        "ci_low": model.conf_int().loc[interaction_term, 0],
        "ci_high": model.conf_int().loc[interaction_term, 1],
        "adj_r_squared": model.rsquared_adj,
    })

pd.DataFrame(rows).to_csv(OUTPUT_DIR / "task_difference_control_summary.csv", index=False)

print("Task-difference control analysis completed.")
print(pd.DataFrame(rows))
