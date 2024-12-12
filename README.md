# NBA Player Classification

This project uses an MLP to classify NBA players into performance tiers based on peak NBA WS percentiles.

## Contents
- data_src contains training and testing csv files and code relating to data processing and preliminary data analysis.
- best_model.pth contains the model weights, and scaler.npz and columns.npy are related to the standard scaler.
- train.py and inference.py are used to train and run the model.
- contribution.md details what each author did for this project.

## Executing Inference

To execute the program, run the following line

python inference.py [*path to data*]

## Dependencies

The following libraries must be installed for inference:
- torch
- numpy
- pandas

The following files must be in the directory:
- best_model.pth
- scaler.npz
- columns.npy

## WandB

The weights and biases outcome report for the model weights in this repository can be found at this link:
- https://wandb.ai/amkim-hamilton-college/nba_player_classification/runs/ze2ndpaj?nw=nwuseramkim
