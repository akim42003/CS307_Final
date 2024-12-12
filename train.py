"""
Authors: Alex Kim & Connor Whynott
Description: MLP to classify players into performance tiers (Star, Bench, Role) based on peak NBA WS percentiles.
"""

################### Imports ###################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
import os

################## Initialize Weights & Biases ###################

wandb.init(project="nba_player_classification", config={
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64
})

config = wandb.config

################## Data ###################

# Read in data
player_data_df = pd.read_csv('data/combined_draft_class_with_ws_FINAL.csv')
target_col = 'WS'

# Separate target column from parameter data
X = player_data_df.drop(columns=[target_col])
y = player_data_df[target_col]

# Drop columns not needed
X = X.drop(columns=['Player','TS%', 'eFG%', 'Season', 'Draft Year', 'Team', 'Draft Team', 'Draft College'])

# Fill NAs
X['3P%'] = X['3P%'].fillna(0)

# One-hot encode categorical features
X = pd.get_dummies(X, dtype=int)

# Compute quantiles for classification
quantile_80 = y.quantile(0.80)
quantile_40 = y.quantile(0.40)

def ws_to_grade(ws):
    if ws >= quantile_80:
        return 0  # Star Player
    elif ws >= quantile_40:
        return 1  # Role Player
    else:
        return 2  # Bench Player

y_class = y.apply(ws_to_grade)

################## Feature Engineering Based on Correlation ###################

if 'Pick' in X.columns:
    X['Pick'] = X['Pick'].replace(0, 1)
    X['InvPick'] = 1.0 / X['Pick']
    X = X.drop(columns=['Pick'])

# Save the final column ordering so inference can match these columns
np.save('columns.npy', X.columns.values)

################## Split Data ###################

# Split into train, validation, and test sets
X_train_and_val, X_test, y_train_and_val, y_test = train_test_split(
    X, y_class, test_size=0.1, random_state=42, stratify=y_class
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_and_val, y_train_and_val, test_size=2/9, random_state=42, stratify=y_train_and_val
)

# Standardize data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save the scaler parameters
np.savez('scaler.npz', mean=scaler.mean_, scale=scaler.scale_)

# Convert to pytorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

################## Model ###################

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
            nn.Linear(hidden_size3, 3)  # 3 classes
        )
    def forward(self, x):
        return self.network(x)

################### Lists to Store Metrics ###################

train_losses = []
val_losses = []
val_accuracies = []

################## Training ###################

input_size = X_train.shape[1]

class_counts = np.bincount(y_train.values)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

model = MLP(input_size)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-3)

best_val_accuracy = 0.0

for epoch in range(config.epochs):
    model.train()
    running_train_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # [batch_size, 3]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)  # [batch_size, 3]
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()
            total_val += labels.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_accuracy": val_accuracy
    })

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{config.epochs}], '
              f'Training Loss: {avg_train_loss:.4f}, '
              f'Validation Loss: {avg_val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.4f}')

    # Save the best model so far
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_model.pth")

################## Testing ###################

model.eval()

test_loss = 0.0
correct_test = 0
total_test = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct_test += (predicted == labels).sum().item()
        total_test += labels.size(0)
        
        # Store predictions and true labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss /= len(test_loader)
test_accuracy = correct_test / total_test

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}\n')

# Log final metrics and confusion matrix to W&B
wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

cm = confusion_matrix(all_labels, all_preds)
class_names = ['A', 'B', 'C']  # A: Star Player, B: Role Player, C: Bench Player

# Log confusion matrix to W&B as a plot
fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title('Confusion Matrix (Test Set)')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
wandb.log({"confusion_matrix": wandb.Image(fig)})
plt.close(fig)

# Finish the W&B run
wandb.finish()
