import numpy as np
from numpy import fft as npfft

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import logging

# from tools import SavedDataset, UnsavedDataset, reg_BCELoss, SensitivityEstimator
from tools import load_resampled_dataset, reg_BCELoss
from network import BaselineModel, ligo_binary_classifier, one_channel_ligo_binary_classifier, LoRA_layer, LoRa_linear

from pars import *
from scheduler_pars import *

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperModel, WhisperFeatureExtractor
import gc
import fnmatch
from peft import LoraConfig, get_peft_model

import sys
if len(sys.argv) > 1:
    i_run_init = int(sys.argv[1])
else:
    i_run_init = 0
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

outfiles_dir = 'outfiles'
state_dicts_dir = 'state_dicts'

# range of optimal SNRs (uniform distribution)
# in the current curriculum learning implementation, this is irrelevant but required to define
snr_range = (5, 15)

# load datasets
logging.info('Loading datasets...')
DSs = [load_resampled_dataset(path, prefix+waveform_fname, prefix+noise_fname, prefix, snr_range, index_array, dtype=dtype, device=device, store_device=store_device) for prefix, index_array in ((train_prefix, train_index_array), (valid_prefix, valid_index_array))]

TrainDS, ValidDS = DSs
logging.info('Datasets loaded successfully.')

# initialize data loaders as training convenience
logging.info('Initializing data loaders...')
TrainDL = DataLoader(TrainDS, batch_size=batch_size, shuffle=True)
ValidDL = DataLoader(ValidDS, batch_size=32, shuffle=True)
logging.info('Data loaders initialized.')

logging.info('Loading Whisper model...')
whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
whisper_model = whisper_model.encoder
logging.info('Whisper model loaded.')

module_names = [name for name, module in whisper_model.named_modules()]

patterns = ["layers.*.self_attn.k_proj", "layers.*.self_attn.v_proj"]

# Fetching all strings that match the patterns
matched_modules = []
for pattern in patterns:
    matched_modules.extend(fnmatch.filter(module_names, pattern))
logging.info('LoRA called. Matching modules found: %s', matched_modules)

lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=matched_modules)
whisper_model_with_lora = get_peft_model(whisper_model, lora_config)
whisper_model_with_lora.to(device)
logging.info('LoRA model created.')

# Freeze base model parameters and unfreeze LoRA parameters
for name, param in whisper_model_with_lora.named_parameters():
    if 'lora' in name:
        param.requires_grad = True  # LoRA parameters are trainable
    else:
        param.requires_grad = False  # Base model parameters are frozen

for name, param in whisper_model_with_lora.named_parameters():
    logging.debug(f"Parameter: {name}, requires_grad: {param.requires_grad}")

def run_training(i_run, DSs, epochs_done=0, network_state_dict=None, optim_state_dict=None):
    logging.info('Initializing training for run %04i...', i_run)
    # regularized binary cross entropy to prevent exploding gradients (gradient clipping is also an option, though)
    crit = reg_BCELoss(dim=2, epsilon=1.e-6)

    logging.info('Opening training output file...')
    tr_outfile = open(os.path.join(outfiles_dir, 'out_train_%04i.txt' % i_run), 'w', buffering=1)

    # set model in training mode and move to the desired device (cpu/cuda)
    logging.info('Initializing network...')
    Network = one_channel_ligo_binary_classifier(whisper_model_with_lora).to(device)
    Network.train()

    # if given, load network state from state dictionary
    if network_state_dict is not None:
        logging.info('Loading network state from state dictionary...')
        Network.load_state_dict(network_state_dict)

    # initialize optimizer and curriculum learning scheduler
    logging.info('Initializing optimizer and curriculum learning scheduler...')
    opt = optim.Adam(Network.parameters(), lr=lr)
    CLSched = CLSchedClass(snr_ranges, (TrainDS, ValidDS), optim=opt, **CLSched_kwargs)

    # if given, load optimizer state from state dictionary (momentum, second order moments in adaptive methods)
    if optim_state_dict is not None:
        logging.info('Loading optimizer state from state dictionary...')
        opt.load_state_dict(optim_state_dict)

    logging.info('Starting training loop for run %04i...', i_run)
    # training loop
    min_valid_loss = 1.e100
    for e in range(epochs_done + 1, epochs_done + epochs + 1):
        logging.info('Starting epoch %04i...', e)
        # training epoch
        Network.train()
        train_loss = 0.
        batches = 0

        # optimization step
        for train_inputs, train_labels in TrainDL:
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            opt.zero_grad()
            train_outputs = Network(train_inputs)
            loss = crit(train_outputs, train_labels)
            loss.backward()
            opt.step()
            train_loss += loss.detach().item()
            batches += 1
        train_loss /= batches
        logging.info('Epoch %04i training loss: %1.12e', e, train_loss)

        # intermediate testing
        with torch.no_grad():
            logging.info('Evaluating validation set...')
            Network.eval()
            # validation loss and accuracy
            valid_loss = 0.
            samples = 0
            valid_accuracy = 0
            batches = 0
            for valid_inputs, valid_labels in ValidDL:
                valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)
                valid_outputs = Network(valid_inputs)
                valid_loss += crit(valid_outputs, valid_labels).detach().item()
                batches += 1
                for p1, p2 in zip(valid_labels, valid_outputs):
                    samples += 1
                    if torch.argmax(p1) == torch.argmax(p2):
                        valid_accuracy += 1
            valid_loss /= batches
            valid_accuracy /= samples
            logging.info('Epoch %04i validation loss: %1.12e, accuracy: %f', e, valid_loss, valid_accuracy)

        # file and stdout output of losses
        tr_outfile.write('%04i    %1.12e    %1.12e    %f\n' % (e, train_loss, valid_loss, valid_accuracy))
        logging.info('Epoch %04i results saved to file.', e)

        # save the best-performing (on validation set) model state and optimizer state to file
        if CLSched.done:
            if valid_loss < min_valid_loss:
                logging.info('New best model found at epoch %04i with validation loss %1.12e. Saving model...', e, valid_loss)
                torch.save(Network.state_dict(), os.path.join(state_dicts_dir, 'best_state_dict_%04i.pt' % i_run))
                min_valid_loss = valid_loss
                save_models(
                    state_dicts_dir,
                    peft_model=Network.encoder,
                    dense_layers=Network.classifier,
                    lora_weights_path='best_lora_weights_run_%04i.pt' % i_run,
                    dense_layers_path='best_dense_layers_run_%04i.pth' % i_run
                )

        torch.save(Network.state_dict(), os.path.join(state_dicts_dir, 'state_dict_run_%04i_epoch_%04i.pt' % (i_run, e)))
        torch.save(opt.state_dict(), os.path.join(state_dicts_dir, 'optim_state_dict_run_%04i_epoch_%04i.pt' % (i_run, e)))
        logging.info('Model and optimizer states saved for epoch %04i.', e)

        save_models(
            state_dicts_dir,
            peft_model=Network.encoder,
            dense_layers=Network.classifier,
            lora_weights_path='lora_weights_run_%04i_epoch_%04i.pt' % (i_run, e),
            dense_layers_path='dense_layers_run_%04i_epoch_%04i.pth' % (i_run, e)
        )
        CLSched.step(valid_loss, valid_accuracy)

        # kill the training loop if the curriculum learning scheduler says so
        if CLSched.interrupt:
            logging.info('Training interrupted by curriculum learning scheduler at epoch %04i.', e)
            break

    # Save models after training
    logging.info('Training completed for run %04i. Saving final models...', i_run)
    save_models(
        state_dicts_dir,
        peft_model=Network.encoder,
        dense_layers=Network.classifier,
        lora_weights_path='final_lora_weights_run_%04i.pt' % i_run,
        dense_layers_path='final_dense_layers_run_%04i.pth' % i_run
    )

    # close output files
    tr_outfile.close()
    logging.info('Training output file closed for run %04i.', i_run)

def save_models(results_path, peft_model, dense_layers, lora_weights_path, dense_layers_path):
    logging.info('Saving models...')
    # Save the entire PEFT (LoRA) model
    peft_model.save_pretrained(os.path.join(results_path, lora_weights_path))
    logging.info('LoRA model saved to %s.', lora_weights_path)

    # Save the Dense layers separately
    torch.save(dense_layers.state_dict(), os.path.join(results_path, dense_layers_path))
    logging.info('Dense layers saved to %s.', dense_layers_path)

# run the training function multiple times
if __name__ == '__main__':
    for i_run in range(i_run_init, i_run_init + runs_number):
        logging.info('Starting run %04i...', i_run)
        run_training(i_run, DSs)
        logging.info('Run %04i completed.', i_run)
