import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Read in data
player_data_df = pd.read_csv('./combined_draft_class_with_ws_FINAL.csv')
target_col = 'WS'

# Separate target column from parameter data
X = player_data_df.drop(columns=[target_col])
y = player_data_df[target_col]

# Drop columns that are not needed
X = X.drop(columns=['Player','TS%', 'eFG%', 'Season', 'Draft Year'])

# Fill NAs (for example in '3P%')
X['3P%'] = X['3P%'].fillna(0)

# One-hot encode categorical features
X = pd.get_dummies(X, dtype=int)

# Compute quantiles for classification
quantile_90 = y.quantile(0.9)
quantile_65 = y.quantile(0.65)
quantile_30 = y.quantile(0.3)

def ws_to_grade(ws):
    if ws >= quantile_90:
        return 0  # A
    elif ws >= quantile_65:
        return 1  # B
    elif ws >= quantile_30:
        return 2  # C
    else:
        return 3  # D

y_class = y.apply(ws_to_grade)

# Split data into train/val/test
X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
    X, y_class, test_size=0.1, random_state=42, stratify=y_class
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_and_val, y_train_and_val, test_size=2/9, random_state=42, stratify=y_train_and_val
)

# 1. Check class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=y_class, order=['A','B','C','D'], palette='Set2')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 2. Examine basic statistical info of numeric features
print("Statistical Summary of Features:")
display(X.describe())

# Identify numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns

# 2a. Plot histograms of numeric features
X[numeric_cols].hist(figsize=(15,10), bins=20, edgecolor='black')
plt.suptitle("Histograms of Numeric Features", fontsize=16)
plt.show()

# 2b. Boxplots of numeric features to identify outliers
plt.figure(figsize=(15,8))
sns.boxplot(data=X[numeric_cols], orient='h', palette='Set3')
plt.title("Boxplots of Numeric Features")
plt.xlabel("Value")
plt.show()

# 3. Feature correlation matrix
corr = X[numeric_cols].corr()

plt.figure(figsize=(12,10))
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

# Print top correlations
corr_pairs = corr.unstack().sort_values(kind="quicksort", ascending=False)
print("Top correlated feature pairs:\n", corr_pairs[corr_pairs < 1].head(10))

# 4. Feature distributions by class (violin/box plots)
# For a smaller subset of features (e.g., first 5 numeric features) to illustrate
selected_features = numeric_cols[:5]

# Convert target classes into a dataframe for plotting
df_full = X.copy()
df_full['Class'] = y_class

# 4a. Violin plots by class for selected features
for feat in selected_features:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='Class', y=feat, data=df_full, order=['A','B','C','D'], palette='Set2')
    plt.title(f"Distribution of {feat} by Class")
    plt.show()

# 4b. Boxplots by class for selected features
for feat in selected_features:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='Class', y=feat, data=df_full, order=['A','B','C','D'], palette='Set3')
    sns.stripplot(x='Class', y=feat, data=df_full, order=['A','B','C','D'], color='black', size=3, jitter=True, alpha=0.7)
    plt.title(f"{feat} by Class (Box + Strip)")
    plt.show()

# 5. Optional: PCA for dimensionality reduction visualization
# This can help us see if classes are separable in a lower-dimensional space
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X[numeric_cols].fillna(0))  # fillna if necessary

plt.figure(figsize=(6,4))
for label in ['A', 'B', 'C', 'D']:
    plt.scatter(X_pca[y_class == label, 0], X_pca[y_class == label, 1], label=label, alpha=0.7)
plt.legend()
plt.title("PCA Projection (First 2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
