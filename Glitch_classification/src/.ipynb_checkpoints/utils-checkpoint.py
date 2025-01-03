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
