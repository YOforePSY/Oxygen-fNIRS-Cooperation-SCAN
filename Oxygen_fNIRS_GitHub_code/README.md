# HBOT-fNIRS cooperation analysis code

This repository contains analysis code for the manuscript:

**Hyperbaric oxygen intervention enhances cooperative gains in high-trust people at high altitude: the role of medial prefrontal inter-brain synchrony**

## Repository contents

- `code/01_main_moderation_models.py`: main moderation models for cooperation gains
- `code/02_roi_specificity_models.py`: ROI-level specificity analyses
- `code/03_task_difference_control.py`: task-difference control analysis
- `code/04_pseudo_dyad_validation.py`: pseudo-dyad validation template
- `code/05_generate_figures.py`: example figure-generation script
- `data/`: place the de-identified processed data CSV here
- `outputs/`: generated statistical tables and figures

## Data

The repository is designed for de-identified processed dyad-level data. Raw fNIRS hyperscanning time-series data, trial-level behavioral data, and item-level questionnaire data should not be placed in this repository.

Expected input file:

```text
data/processed_main_analysis_data.csv
```

At minimum, the file should contain the following variables:

```text
dyad_id
Trust_mean
Trust_mean_z
Coop_mPFC_Change_z
success_gain_z
efficiency_gain_z
age_mean_z
sex_code
order_group
```

Additional ROI variables can be included for ROI-specificity analyses.

## Notes

The scripts are intended to document the statistical workflow used in the manuscript. Minor modifications may be needed if variable names differ from the final analysis dataset.
