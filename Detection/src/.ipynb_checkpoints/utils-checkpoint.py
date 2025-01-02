import os
import datetime
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
import numpy as np
from tqdm import tqdm

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_plot(epochs, train_losses, val_losses, auc_values, results_path, plot_name):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_losses, label='Train Loss', color='tab:blue')
    ax1.plot(epochs, val_losses, label='Val Loss', color='tab:green')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('AUC', color=color)
    ax2.plot(epochs, auc_values, label='AUC', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Training and Validation Losses & AUC')
    plt.savefig(os.path.join(results_path, plot_name))
    plt.close()

def get_paths(data_path, checkpoint_name):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d")

    results_path = os.path.join(data_path, "results")
    daily_results_path = os.path.join(results_path, f"model_results_{timestamp}/")
    if not os.path.exists(daily_results_path):
        os.makedirs(daily_results_path)

    checkpoint_path = os.path.join(daily_results_path, checkpoint_name)
    if not os.path.exists(daily_results_path):
        return daily_results_path, None
    else:
        return daily_results_path, checkpoint_path