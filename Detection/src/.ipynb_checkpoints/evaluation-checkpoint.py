import argparse
import datasets
from datasets import load_from_disk
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime
import gc

from sklearn.metrics import f1_score, classification_report, roc_curve, roc_auc_score
from sklearn.utils import resample

from transformers import WhisperModel
from peft import PeftModel

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import two_channel_ligo_binary_classifier, one_channel_ligo_binary_classifier
from src.dataset import two_channel_LigoBinaryData, one_channel_LigoBinaryData
from src.utils import EarlyStopper, save_plot

from matplotlib import rcParams

# Global font settings
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
rcParams['font.size'] = 7

def evaluate(model, data_loader, device, criterion=nn.BCEWithLogitsLoss()):
    all_labels = []
    all_preds = []
    all_raw_preds = []
    all_snr = []
    total_loss = 0.0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in tqdm(data_loader):
            inputs, labels, snr = batch
            inputs = inputs.to(device)
            labels = labels.view(-1, 1).float().to(device)
            snr = snr.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.sigmoid(logits).round()
            raw_preds = torch.sigmoid(logits)
            
            all_raw_preds.append(raw_preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_snr.append(snr.cpu().numpy())

    all_labels = np.concatenate(all_labels, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_raw_preds = np.concatenate(all_raw_preds, axis=0)
    all_snr = np.concatenate(all_snr, axis=0)

    loss = total_loss / len(data_loader)
    auc = roc_auc_score(all_labels, all_raw_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_raw_preds)
    report = classification_report(all_labels, all_preds, target_names=['injection', 'noise'])
    f1 = f1_score(all_labels, all_preds, average='macro')

    eval_out = {
        'loss': loss,
        'auc': auc,
        'f1': f1,
        'all_labels': all_labels,
        'all_preds': all_preds,
        'all_raw_preds': all_raw_preds,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'all_snr': all_snr
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return eval_out

def load_models(lora_weights_path, dense_layers_path, num_classes=1, model_type='2D'):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder

    peft_model = PeftModel.from_pretrained(whisper_model, lora_weights_path)

    if model_type == '2D':
        model = two_channel_ligo_binary_classifier(encoder=peft_model, num_classes=num_classes)
    else:
        model = one_channel_ligo_binary_classifier(encoder=peft_model, num_classes=num_classes)
    
    model.classifier.load_state_dict(torch.load(dense_layers_path))
    return model

def evaluate_dataset(model, dataset, device, batch_size=128):
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2, pin_memory=True)
    eval_out = evaluate(model, test_loader, device)
    return eval_out['fpr'], eval_out['tpr'], eval_out['auc'], eval_out['all_labels'], eval_out['all_preds'], eval_out['all_raw_preds'], eval_out['all_snr']

def bootstrap_roc_curve(all_labels, all_preds, num_bootstrap=1000):
    all_fprs = []
    all_tprs = []

    for _ in range(num_bootstrap):
        labels_resampled, preds_resampled = resample(all_labels, all_preds)
        fpr, tpr, _ = roc_curve(labels_resampled, preds_resampled)
        all_fprs.append(fpr)
        all_tprs.append(np.interp(np.logspace(-4, 0, num=500), fpr, tpr))
    
    mean_tpr = np.mean(all_tprs, axis=0)
    std_tpr = np.std(all_tprs, axis=0)
    return np.logspace(-4, 0, num=500), mean_tpr, std_tpr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate LIGO models with 1D or 2D data")
    parser.add_argument('--model_type', type=str, choices=['1D', '2D'], required=True, help="Model type: 1D or 2D")
    parser.add_argument('--lora_weights_path', type=str, required=True, help="Path to LoRA weights")
    parser.add_argument('--dense_layers_path', type=str, required=True, help="Path to dense layer weights")
    args = parser.parse_args()

    device = torch.device('cuda')

    if args.model_type == '2D':
        dataset_paths = [
            '/workspace/ligo_data/Binary_classification_tests/Test_Whisper_SNR-6_resampled',
            '/workspace/ligo_data/Binary_classification_tests/Test_Whisper_SNR-8_resampled_train',
            '/workspace/ligo_data/Binary_classification_tests/Test_Whisper_SNR-10_resampled'
        ]
        dataset_class = two_channel_LigoBinaryData
    else:
        dataset_paths = [
            '/workspace/ligo_data/data_ts/Test_Whisper_SNR-5_single_det_resampled',
            '/workspace/ligo_data/data_ts/Test_Whisper_SNR-7_single_det_resampled',
            '/workspace/ligo_data/data_ts/Test_Whisper_SNR-9_single_det_resampled'
        ]
        dataset_class = one_channel_LigoBinaryData

    datasets = [load_from_disk(path) for path in dataset_paths]
    test_sets = [dataset_class(ds, device, 'small') for ds in datasets]

    model = load_models(args.lora_weights_path, args.dense_layers_path, model_type=args.model_type)
    model.eval()
    model = model.to(device)

    for i, test_set in enumerate(test_sets):
        fpr, tpr, auc, all_labels, all_preds, all_raw_preds, all_snr = evaluate_dataset(model, test_set, device)
        fpr_bootstrap, mean_tpr, std_tpr = bootstrap_roc_curve(all_labels, all_raw_preds)

        plt.figure()
        plt.plot(fpr, mean_tpr, label=f'ROC Curve (AUC={auc:.2f})')
        plt.fill_between(fpr_bootstrap, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(f'/workspace/ligo_results/ROC_curve_SNR_{i}_{args.model_type}.png')

        
# Example usage: python script.py --model_type 2D --lora_weights_path path_to_lora_weights --dense_layers_path path_to_dense_layers
