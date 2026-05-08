"""
Pseudo-dyad validation template.

This script provides a simplified permutation-style template. If the repository
contains only dyad-level processed data, true pseudo-dyad reconstruction from
raw participant-level time series cannot be performed. Therefore, this script
demonstrates a conservative label-permutation validation of the mPFC IBS
predictor relative to behavioral outcomes.

For full pseudo-dyad IBS construction, participant-level fNIRS time series are
required and should not be publicly uploaded unless ethics approval explicitly
permits public sharing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed_main_analysis_data.csv"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 20260509
N_PERMUTATIONS = 5000

rng = np.random.default_rng(RANDOM_SEED)
df = pd.read_csv(DATA_FILE)

required = ["Coop_mPFC_Change_z", "Trust_mean_z", "efficiency_gain_z", "age_mean_z", "sex_code", "order_group"]
missing = [v for v in required if v not in df.columns]
if missing:
    raise ValueError(f"Missing required variables: {missing}")

def get_interaction_beta(data):
    model = smf.ols(
        "efficiency_gain_z ~ Coop_mPFC_Change_z * Trust_mean_z + age_mean_z + sex_code + order_group",
        data=data
    ).fit()
    term = "Coop_mPFC_Change_z:Trust_mean_z"
    if term not in model.params.index:
        term = "Trust_mean_z:Coop_mPFC_Change_z"
    return model.params[term]

observed_beta = get_interaction_beta(df)

null_betas = []
for _ in range(N_PERMUTATIONS):
    permuted = df.copy()
    permuted["Coop_mPFC_Change_z"] = rng.permutation(permuted["Coop_mPFC_Change_z"].values)
    null_betas.append(get_interaction_beta(permuted))

null_betas = np.array(null_betas)
empirical_p = (np.sum(np.abs(null_betas) >= abs(observed_beta)) + 1) / (N_PERMUTATIONS + 1)

summary = pd.DataFrame({
    "observed_beta": [observed_beta],
    "n_permutations": [N_PERMUTATIONS],
    "empirical_p_two_tailed": [empirical_p],
    "null_beta_mean": [null_betas.mean()],
    "null_beta_sd": [null_betas.std(ddof=1)],
})

summary.to_csv(OUTPUT_DIR / "pseudo_dyad_label_permutation_summary.csv", index=False)
pd.DataFrame({"null_beta": null_betas}).to_csv(OUTPUT_DIR / "pseudo_dyad_null_distribution.csv", index=False)

print("Pseudo-dyad / permutation validation completed.")
print(summary)
