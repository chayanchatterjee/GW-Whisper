import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
from transformers import WhisperModel
from dataset import one_channel_LigoBinaryData
from model import one_channel_ligo_binary_classifier
from utils import EarlyStopper

from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'  # Use sans-serif fonts
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']  # Specify preferred fonts
rcParams['font.size'] = 7  # Set default font size to 7 pt

def load_concatenated_dataset(data_path):
    chunk_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and "chunk" in d]
    if chunk_dirs:
        chunk_dirs = sorted(chunk_dirs)
        print(f"Detected chunks: {chunk_dirs}")
        datasets = [load_from_disk(chunk_dir) for chunk_dir in chunk_dirs]
        full_dataset = concatenate_datasets(datasets)
    else:
        full_dataset = load_from_disk(data_path)
    return full_dataset

def save_classification_report(report, results_path, model_name):
    report_path = os.path.join(results_path, f"{model_name}_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Classification report saved to {report_path}")

def plot_confusion_matrix(all_labels, all_preds, label_encoder, save_path):
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(label_encoder.classes_)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # Normalize along rows
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45, colorbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Normalized confusion matrix saved to {save_path}")

def evaluate(model, data_loader, device, criterion, label_encoder):
    model.to(device)
    all_labels = []
    all_preds = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, snr = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(data_loader)
    report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro')

    return {
        'loss': loss,
        'f1': f1,
        'report': report,
        'all_labels': all_labels,
        'all_preds': all_preds,
    }

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, results_path, whisper_weights_path, dense_weights_path, model_name, writer, label_encoder, start_epoch=0):
    model.to(device)
    criterion = criterion.to(device)
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=60)
    best_cm_path = None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels, snr in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        eval_out = evaluate(model, val_loader, device, criterion, label_encoder)
        val_loss = eval_out['loss']
        val_f1 = eval_out['f1']

        writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
        writer.add_scalar(f'{model_name}/val_f1', val_f1, epoch)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(model.encoder.state_dict(), whisper_weights_path)
            torch.save(model.classifier.state_dict(), dense_weights_path)

            best_cm_path = os.path.join(results_path, f"{model_name}_best_confusion_matrix.png")
            plot_confusion_matrix(eval_out['all_labels'], eval_out['all_preds'], label_encoder, best_cm_path)

        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best confusion matrix saved to: {best_cm_path}")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.log_dir)

    train_ds = load_concatenated_dataset(args.train_data_path)
    test_ds = load_concatenated_dataset(args.test_data_path)

    label_encoder = LabelEncoder()
    all_labels = train_ds['labels'] + test_ds['labels']

    modified_labels = ["GW" if label == "GW" else " ".join(label.split("_")).title() for label in all_labels]
    label_encoder.fit(modified_labels)

    train_ds = train_ds.map(lambda x: {'encoded_labels': label_encoder.transform(["GW" if x['labels'] == "GW" else " ".join(x['labels'].split("_")).title()])[0]}, batched=False)
    test_ds = test_ds.map(lambda x: {'encoded_labels': label_encoder.transform(["GW" if x['labels'] == "GW" else " ".join(x['labels'].split("_")).title()])[0]}, batched=False)

    train_data = one_channel_LigoBinaryData(train_ds, device, encoder=args.encoder)
    valid_data = one_channel_LigoBinaryData(test_ds, device, encoder=args.encoder)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    whisper_model = WhisperModel.from_pretrained(f"openai/whisper-{args.encoder}").encoder.to(device)
    model = one_channel_ligo_binary_classifier(whisper_model, num_classes=len(label_encoder.classes_))
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    print(f"Training model...")
    train(
        model, train_loader, valid_loader, optimizer, criterion, device, args.num_epochs, args.results_path,
        os.path.join(args.results_path, f"{args.model_name}_best_whisper_weights.pth"),
        os.path.join(args.results_path, f"{args.model_name}_best_dense_weights.pth"),
        args.model_name, writer, label_encoder, start_epoch
    )

    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--log_dir", type=str, default="/GW-Whisper/Glitch_classification/results/logs", help="TensorBoard log directory")
    parser.add_argument("--results_path", type=str, default="/GW-Whisper/Glitch_classification/results/models/generic", help="Path to save results and models")
    parser.add_argument("--encoder", type=str, default="tiny", help="Whisper encoder size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=8e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--model_name", type=str, default="multi_class_model", help="Name of the model")
    
    args = parser.parse_args()
   
