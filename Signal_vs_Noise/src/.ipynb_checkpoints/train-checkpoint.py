import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk, concatenate_datasets
from sklearn.metrics import classification_report, f1_score, roc_auc_score, roc_curve
import numpy as np
import fnmatch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperModel, WhisperFeatureExtractor
from peft import LoraConfig, PeftModel, PeftConfig
from transformers import WhisperFeatureExtractor
import gc
from peft import LoraConfig, get_peft_model

from .dataset import two_channel_LigoBinaryData
from .model import two_channel_ligo_binary_classifier, TwoChannelLIGOBinaryClassifierCNN
from .utils import EarlyStopper, save_plot, get_paths


def load_concatenated_dataset(data_path):
    # Check if the data_path contains chunk directories
    chunk_dirs = [os.path.join(data_path, d) for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d)) and "chunk" in d]
    
    if chunk_dirs:
        # Sort the chunk directories to maintain the correct order
        chunk_dirs = sorted(chunk_dirs)
        print(f"Detected chunks: {chunk_dirs}")
        
        # Load each chunk and concatenate them
        datasets = [load_from_disk(chunk_dir) for chunk_dir in chunk_dirs]
        full_dataset = concatenate_datasets(datasets)
    else:
        # Load as a single dataset if no chunks are found
        full_dataset = load_from_disk(data_path)
    
    return full_dataset

def load_models(lora_weights_path, dense_layers_path, num_classes=1):
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder

    # Load the PEFT model with LoRA weights
    peft_model = PeftModel.from_pretrained(whisper_model, lora_weights_path)

    # Create the main model using the encoder
    model = two_channel_ligo_binary_classifier(encoder=peft_model, num_classes=num_classes)

    # Load the dense classifier weights
    model.classifier.load_state_dict(torch.load(dense_layers_path))

    # Freeze only the Whisper encoder weights
    for param in peft_model.base_model.parameters():
        param.requires_grad = False

    # Ensure PEFT weights and classifier weights remain trainable
    for name, param in peft_model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def evaluate(model, data_loader, device, criterion=nn.BCEWithLogitsLoss()):
    model.to(device)
    all_labels = []
    all_preds = []
    all_raw_preds = []
    all_snr = []
    total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            input_0, input_1, labels, snr = batch

            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.view(-1, 1).float().to(device)
            snr = snr.to(device)
            
            logits = model(input_0, input_1)
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

def save_models(models_path, peft_model, dense_layers, lora_weights_path, dense_layers_path):
    peft_model.to("cpu")
    dense_layers.to("cpu")

    peft_model.save_pretrained(os.path.join(models_path, lora_weights_path))
    torch.save(dense_layers.state_dict(), os.path.join(models_path, dense_layers_path))

def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, models_path, model_name, writer, save_paths):
    model.to(device)
    criterion = criterion.to(device)
    best_val_loss = float('inf')
    early_stopper = EarlyStopper(patience=15)
    
    train_losses = []
    val_losses = []
    val_aucs = []
    train_times = []
    
    best_lora_weights_path = os.path.join(models_path, f"best_{save_paths['lora_weights']}")
    best_dense_layers_path = os.path.join(models_path, f"best_{save_paths['dense_layers']}")
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        train_loss = 0.0
        for input_0, input_1, labels, snr in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            input_0 = input_0.to(device)
            input_1 = input_1.to(device)
            labels = labels.view(-1, 1).float().to(device)
            snr = snr.to(device)
            
            optimizer.zero_grad()
            outputs = model(input_0, input_1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        end_time = time.time()
        epoch_time = end_time - start_time
        train_times.append(epoch_time)
        
        eval_out = evaluate(model, val_loader, device, criterion)
        val_loss = eval_out['loss']
        val_auc = eval_out['auc']
        
        val_losses.append(val_loss)
        val_aucs.append(val_auc)
        
        writer.add_scalar(f'{model_name}/train_loss', train_loss, epoch)
        writer.add_scalar(f'{model_name}/val_loss', val_loss, epoch)
        writer.add_scalar(f'{model_name}/val_auc', val_auc, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save the best LoRA and dense layer weights
            model.encoder.save_pretrained(best_lora_weights_path)
            torch.save(model.classifier.state_dict(), best_dense_layers_path)
        
        if early_stopper.early_stop(val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    save_models(
        models_path,
        peft_model=model.encoder,
        dense_layers=model.classifier,
        lora_weights_path=save_paths['lora_weights'],
        dense_layers_path=save_paths['dense_layers']
    )
    
    return train_losses, val_losses, val_aucs, train_times

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Load the dataset, handling chunked data if present
    ds = load_concatenated_dataset(args.data_path)
    ds_split = ds.train_test_split(test_size=args.test_size, seed=args.seed, shuffle=True)
    
    train_data = two_channel_LigoBinaryData(ds_split['train'], device, encoder=args.encoder)
    valid_data = two_channel_LigoBinaryData(ds_split['test'], device, encoder=args.encoder)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder

    module_names = [name for name, module in whisper_model.named_modules()]

    patterns = ["layers.*.self_attn.q_proj", "layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj", "layers.*.self_attn.o_proj"]

    # Fetching all strings that match the patterns
    matched_modules = []
    for pattern in patterns:
        matched_modules.extend(fnmatch.filter(module_names, pattern))

    models = []
    encoder = WhisperModel.from_pretrained(f"openai/whisper-{args.encoder}").encoder.to(device)
    
    
    if args.method == 'full_finetune':
        models.append(('full_finetune', two_channel_ligo_binary_classifier(encoder)))
        
        for param in two_channel_ligo_binary_classifier(encoder).parameters():
            param.requires_grad = True
            
            
    elif args.method == 'LoRA':
        
        lora_config = LoraConfig(use_dora=False, r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=matched_modules)
        whisper_model_with_lora = get_peft_model(whisper_model, lora_config).to(device)

        for name, param in whisper_model_with_lora.named_parameters():
            param.requires_grad = 'lora' in name
        
        models.append(('LoRA', two_channel_ligo_binary_classifier(whisper_model_with_lora)))
        
    
    elif args.method == 'DoRA':
        
        lora_config = LoraConfig(use_dora=True, r=args.lora_rank, lora_alpha=args.lora_alpha, target_modules=matched_modules)
        whisper_model_with_dora = get_peft_model(whisper_model, lora_config).to(device)

        for name, param in whisper_model_with_dora.named_parameters():
            param.requires_grad = 'lora' in name
        
        models.append(('DoRA', two_channel_ligo_binary_classifier(whisper_model_with_dora)))
        
    
    for model_name, model in models:
        
        model.to(device)
        
        criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
        save_paths = {
            'lora_weights': f'lora_weights_{args.lora_rank}_{args.lora_alpha}',
            'dense_layers': f'dense_layers_{args.lora_rank}_{args.lora_alpha}.pth'
        }
        
        if args.load_model_path is None:
            print(f"Training {model_name} model from scratch...")
            train_losses, val_losses, val_aucs, train_times = train(
                model, train_loader, valid_loader, optimizer, criterion, device, args.num_epochs, args.models_path, 
            model_name, writer, save_paths
            )
        
        else:
            print(f"Training {model_name} model from {args.load_model_path}...")
            model = load_models(args.load_model_path + args.load_lora_weights, args.load_model_path + load_dense_weights)
            
            model.to(device)
        
            criterion = nn.BCEWithLogitsLoss().to(device)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
            save_paths = {
                'lora_weights': f'lora_weights_{args.lora_rank}_{args.lora_alpha}',
                'dense_layers': f'dense_layers_{args.lora_rank}_{args.lora_alpha}.pth'
            }
            
            train_losses, val_losses, val_aucs, train_times = train(
                model, train_loader, valid_loader, optimizer, criterion, device, args.num_epochs, args.models_path, 
                model_name, writer, save_paths
            )
        
        plt.figure()
        plt.rc('font', family='serif')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
#        plt.title(f'{model_name} Loss vs Epoch')
        plt.savefig(f'{args.figures_path}/{model_name}_loss.png')
        
        plt.figure()
        plt.rc('font', family='serif')
        plt.plot(val_aucs)
        plt.xlabel('Epoch')
        plt.ylabel('Validation AUC')
#        plt.title(f'{model_name} Val AUC vs Epoch')
        plt.savefig(f'{args.figures_path}/{model_name}_val_auc.png')
    
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define your arguments here
    args = parser.parse_args()
    main(args)
