# NBA Player Classification

This project uses an MLP to classify NBA players into performance tiers based on peak NBA WS percentiles.

## Structure
- `data/`: Contains raw and processed data.
- `notebooks/`: Jupyter notebooks for EDA and feature engineering.
- `src/`: Source code for the model, training, and utilities.
- `reports/`: Analysis results, figures, and documentation.

## Usage
1. Run training: `python src/train.py`.
2. Results and confusion matrix will be logged to W&B if configured.
