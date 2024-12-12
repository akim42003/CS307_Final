"""
Inference script for NBA player classification.
Usage: python inference.py path/to/input_data.csv
"""

import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        hidden_size1 = int((2/3) * input_size + 1)
        hidden_size2 = hidden_size1 // 2
        hidden_size3 = hidden_size2 // 2

        self.network = nn.Sequential(
            nn.LazyLinear(hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.Sigmoid(),
            nn.LazyLinear(hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.Sigmoid(),
            nn.LazyLinear(hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.Sigmoid(),
            nn.Linear(hidden_size3, 3)  # 3 classes: A, B, C
        )

    def forward(self, x):
        return self.network(x)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the path to the input CSV file.")
        sys.exit(1)

    input_csv = sys.argv[1]

    # Load input data
    df = pd.read_csv(input_csv)

    # Drop target column if present
    target_col = 'WS'
    if target_col in df.columns:
        df = df.drop(columns=[target_col])

    # Drop columns
    drop_cols = ['TS%', 'eFG%', 'Season', 'Draft Year', 'Team', 'Draft Team', 'Draft College']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Extract player names if available, otherwise fallback
    if 'Player' in df.columns:
        player_names = df['Player'].values
        df = df.drop(columns=['Player'])  # Drop now since model doesn't need it
    else:
        # If no Player column, just label by row
        player_names = [f"Row {i}" for i in range(len(df))]

    # Fill NAs
    if '3P%' in df.columns:
        df['3P%'] = df['3P%'].fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df, dtype=int)

    # Load the final column order from training
    try:
        final_columns = np.load('columns.npy', allow_pickle=True).tolist()
    except FileNotFoundError:
        print("columns.npy not found. Please provide the final column ordering from training.")
        sys.exit(1)

    # Identify missing columns and add them all at once
    missing_cols = [col for col in final_columns if col not in df.columns]
    if missing_cols:
        df_missing = pd.DataFrame(0, index=df.index, columns=missing_cols)
        df = pd.concat([df, df_missing], axis=1)

    # Drop any extra columns not in final_columns and reorder
    df = df[final_columns]

    # Load scaler parameters
    try:
        scaler_data = np.load('scaler.npz')
        mean_ = scaler_data['mean']
        scale_ = scaler_data['scale']
        df = (df - mean_) / scale_
    except FileNotFoundError:
        print("scaler.npz not found. Please provide the scaler parameters from training.")
        sys.exit(1)

    # Convert df to tensor
    X_infer = torch.tensor(df.values, dtype=torch.float32)

    # Load the trained model
    model = MLP(input_size=df.shape[1])

    # Initialize lazy layers with a dummy batch of size 2
    dummy_input = torch.randn(2, df.shape[1])
    _ = model(dummy_input)

    try:
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    except FileNotFoundError:
        print("best_model.pth not found. Please provide the trained model checkpoint.")
        sys.exit(1)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        outputs = model(X_infer)
        _, preds = torch.max(outputs, 1)

   # Define the performance tiers
    class_names = ['Star', 'Role Player', 'Bench Player']
    pred_classes = [class_names[p] for p in preds.numpy()]

    max_name_length = max(len(name) for name in player_names)
    header_format = f"{{:<{max_name_length}}}  {{}}"

    # Print header
    print("-" * (max_name_length + 12))
    print(header_format.format("Player Name", "Prediction"))
    print("-" * (max_name_length + 12))

    # Define a format string for predictions
    row_format = f"{{:<{max_name_length}}}  {{}}"

    # Print predictions with aligned output
    for i, c in enumerate(pred_classes):
        print(row_format.format(player_names[i], c))