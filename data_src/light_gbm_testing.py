import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Read in data
player_data_df = pd.read_csv('data/combined_draft_class_with_ws_FINAL.csv')
target_col = 'WS'

# Separate target column from parameter data
X = player_data_df.drop(columns=[target_col])
y = player_data_df[target_col]

# Drop columns
X = X.drop(columns=['Player','TS%', 'eFG%', 'Season', 'Draft Year'])

# Normalize 3PA, 3P, PTS with Z-Score
X['3PA'] = (X['3PA'] - X['3PA'].mean()) / X['3PA'].std()
X['3P'] = (X['3P'] - X['3P'].mean()) / X['3P'].std()
X['PTS'] = (X['PTS'] - X['PTS'].mean()) / X['PTS'].std()

# Fill NAs with appropriate values
X['3P%'] = X['3P%'].fillna(0)

# One hot encoding
# X = pd.get_dummies(X, dtype=int)

# Drop all columns that are not ints
X = X.select_dtypes(include=['int64'])

# Split into train, val, and test (70/20/10)
X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_and_val, y_train_and_val, test_size=2/9, random_state=42)


# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters for LightGBM regressor
params = {
    'objective': 'regression',
    'metric': 'rmse',  # Root Mean Square Error
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8
}

# Train the model
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
)

# Plot feature importance
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importance()
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important features at the top
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Predicting WS')
plt.show()