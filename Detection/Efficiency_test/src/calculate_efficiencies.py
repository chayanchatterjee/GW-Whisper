### This script calculates the efficiency estimates of a set of runs which have already finished from the saved state dictionaries
#from network import get_network
from tools import EfficiencyEstimator, load_resampled_dataset, ResampledDataset
from pars import *

from transformers import WhisperFeatureExtractor, AdamW

import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim
from whisper.model import Whisper
from whisper.tokenizer import Tokenizer, get_tokenizer

from network import BaselineModel, ligo_binary_classifier, one_channel_ligo_binary_classifier, LoRA_layer, LoRa_linear

from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperModel
from peft import LoraConfig, PeftModel, PeftConfig
import numpy as np


#efficiency_snrs = list(range(5, 20))
efficiency_snrs = list(np.arange(5, 25, 2))
faps = [0.1, 0.01, 0.001, 0.0001, 0.00001]

device = 'cuda'
remove_softmax = True
output_directory = 'Detection/Efficiency_test/src/efficiencies'

import torch
import sys
import os

if len(sys.argv)==1:
	indices_run = range(runs_number)
elif len(sys.argv)==2:
	indices_run = [int(sys.argv[1])]
elif len(sys.argv)==3:
	indices_run = range(int(sys.argv[1]), int(sys.argv[2]))
else:
	raise ValueError
print('indices_run = %s' % str(indices_run), flush=True)
#indiceses_epoch = [range(1, epochs+1) for _ in range(len(indices_run))]
indiceses_epoch = [[40, 55, 70]]

state_dict_fnameses = [[os.path.join('Detection/Efficiency_test/src/state_dicts', 'state_dict_run_%04i_epoch_%04i.pt' % (i_run, i_epoch)) for i_epoch in indices_epoch] for i_run, indices_epoch in zip(indices_run, indiceses_epoch)]	# in case of training split over multiple folders or inequal lengths of individual runs, modify this line

lora_weights_fnameses = [[os.path.join('Detection/Efficiency_test/src/state_dicts', 'lora_weights_run_%04i_epoch_%04i.pt' % (i_run, i_epoch)) for i_epoch in indices_epoch] for i_run, indices_epoch in zip(indices_run, indiceses_epoch)]	# in case of training split over multiple folders or inequal lengths of individual runs, modify this line
dense_layers_fnameses = [[os.path.join('Detection/Efficiency_test/src/state_dicts', 'dense_layers_run_%04i_epoch_%04i.pth' % (i_run, i_epoch)) for i_epoch in indices_epoch] for i_run, indices_epoch in zip(indices_run, indiceses_epoch)]	# in case of training split over multiple folders or inequal lengths of individual runs, modify this line

# range of optimal SNRs (uniform distribution)
# in the current curriculum learning implementation, this is irrelevant but required to define
snr_range = (0, 0)

# construct dataset and dataloader (PyTorch convenience for batches)
TestDS = load_resampled_dataset(path, test_prefix+waveform_fname, test_prefix+noise_fname, test_prefix, snr_range, test_index_array, dtype=dtype, device=device, store_device=store_device)

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

# UnsavedDataset --> ResampledDataset
SEWaveDS = ResampledDataset(TestDS.wave_tensor, TestDS.noise_tensor, (0., 0.), TestDS.wave_lim, TestDS.noise_comb_lim, (0, 0), feature_extractor, noises_per_signal=TestDS.noises_per_signal, device=device, dtype=dtype, bool=True)
SENoiseDS = ResampledDataset(TestDS.wave_tensor, TestDS.noise_tensor, (0., 0.), (0, 0), (0, 0), TestDS.noise_pure_lim, feature_extractor, noises_per_signal=TestDS.noises_per_signal, device=device, dtype=dtype, bool=True)


EEst = EfficiencyEstimator(SEWaveDS, SENoiseDS, efficiency_snrs, faps=faps)

def load_model(lora_weights_path, dense_layers_path):
    
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    whisper_model = whisper_model.encoder

    peft_model = PeftModel.from_pretrained(whisper_model, lora_weights_path)

    # Create the full model with loaded LoRA and dense layers
    model = one_channel_ligo_binary_classifier(encoder=peft_model)
    
    # Load the Dense layer weights
    model.classifier.load_state_dict(torch.load(dense_layers_path, map_location=device))

    return model

# main loop over runs, epochs, batches
for i_run, indices_epoch, lora_weights_fnames, dense_layers_fnames in zip(indices_run, indiceses_epoch, lora_weights_fnameses, dense_layers_fnameses):
	for e, lora_weights_fname, dense_layers_fname in zip(indices_epoch, lora_weights_fnames, dense_layers_fnames):
		ef_outfile = open(os.path.join(output_directory, 'out_efficiencies_run_%04i_epoch_%04i.txt' % (i_run, e)), 'w', buffering=1)
		ef_outfile.write('# FAPs: %f' % faps[0])
		for fap in faps[1:]:
			ef_outfile.write('    %f' % fap)
		ef_outfile.write('\n')
		print(lora_weights_fname, flush=True)
#		Network = get_network(device=device, dtype=dtype)
		Network = load_model(lora_weights_fname, dense_layers_fname).to(device)
		Network.eval()
#		Network.load_state_dict(torch.load(state_dict_fname, map_location=device))
		if remove_softmax:
			new_layer = torch.nn.Linear(2, 2, bias=False)
			new_layer._parameters['weight'] = torch.nn.Parameter(torch.Tensor([[1., -1.], [-1., 1.]]), requires_grad=False)
			new_layer.to(device=device)
#			Network[-1] = new_layer

			# Replace the Softmax layer in the classifier
			# Assuming that Softmax is the last layer in the classifier
			layers = list(Network.classifier.children())
			if isinstance(layers[-1], torch.nn.Softmax):
				layers[-1] = new_layer
				Network.classifier = torch.nn.Sequential(*layers)
			else:
				raise ValueError("The last layer of the classifier is not a Softmax layer.")

		with torch.no_grad():
			estimated_efficiencies = EEst(Network)
		for snr, effs in zip(efficiency_snrs, estimated_efficiencies):
			ef_outfile.write('%f' % snr)
			for num in effs:
				ef_outfile.write('    %f' % num)
			ef_outfile.write('\n')
		ef_outfile.close()