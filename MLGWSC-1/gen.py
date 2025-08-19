#! /usr/bin/env python

# Modifications copyright 2025 Chayan Chatterjee
# Based on original work by Ondřej Zelenka
# Licensed under the Apache License, Version 2.0


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

### Import modules
from argparse import ArgumentParser
import logging
import numpy as np
import h5py
import pycbc.waveform, pycbc.noise, pycbc.psd, pycbc.distributions, pycbc.detector, pycbc.types
# from pycbc import DYN_RANGE_FAC
import os, os.path
from tqdm import tqdm
from itertools import cycle

from train import Dataset
from apply import regularize_psd, whiten

np_gen = np.random.default_rng()
delta_t = 1./2048.

### Whitening parameters
segment_duration = 0.5
max_filter_length = 512
max_filter_duration = max_filter_length*delta_t
max_filter_halfduration = max_filter_duration*.5

class HDFNoiseGetter:
    def __init__(self, fpath, detectors=None):
        self.detectors = [det.name for det in detectors]
        self.noises = []
        self.psds = []
        self.whitening_psds = []
        with h5py.File(fpath, 'r') as inf:
            self.psd_indices = inf['psd_indices'][()]
            noises = inf['noises'][()]
            psds = inf['psds'][()]
            whitening_psds = inf['whitening_psds'][()]
            inf_det_names = np.array(inf.attrs['detectors'])
            for det in self.detectors:
                index = np.where(inf_det_names == det)[0][0]
                self.noises.append(noises[:, index])
                self.psds.append([pycbc.types.FrequencySeries(data[index], delta_f=inf['psds'].attrs['delta_f']) for data in psds])
                self.whitening_psds.append([pycbc.types.FrequencySeries(data[index], delta_f=inf['whitening_psds'].attrs['delta_f']) for data in whitening_psds])
        self.noises = np.stack(self.noises, axis=1)
        self.psds = list(zip(*(self.psds)))
        self.whitening_psds = list(zip(*(self.whitening_psds)))


    def __iter__(self):
        self.noise_iter = iter(zip(self.noises, self.psd_indices))
        return self
    def __next__(self):
        new_noise, new_index = next(self.noise_iter)
        return new_noise, self.psds[new_index], self.whitening_psds[new_index]

class GaussianNoiseGetter:
    def __init__(self, fpath=None, detectors=None, low_frequency_cutoff=None,
                       sample_length=2560, segment_duration=0.5,
                       max_filter_duration=0.25):
        if fpath is None:
            psd_fun = pycbc.psd.analytical.aLIGOZeroDetHighPower
            self.all_psds = [[psd_fun(sample_length//2+1, 1./(delta_t*sample_length), low_frequency_cutoff) for _ in detectors]]
        else:
            with h5py.File(fpath, 'r') as psd_file:
                self.all_psds = [[pycbc.types.FrequencySeries(psd_ds[()], delta_f=psd_ds.attrs['delta_f'], dtype=np.float64) for psd_ds in det_grp.values()] for det_grp in psd_file.values()]
                self.all_psds = list(zip(*(self.all_psds)))
        self.all_whitening_psds = [[regularize_psd(psd, low_frequency_cutoff) for psd in psds] for psds in self.all_psds]
        self.noise_fun = pycbc.noise.gaussian.frequency_noise_from_psd
        self.low_frequency_cutoff = low_frequency_cutoff
        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        return
    def __iter__(self):
        self.psd_iter = iter(zip(cycle(self.all_psds), cycle(self.all_whitening_psds)))
        return self
    def __next__(self):
        psds, whitening_psds = next(self.psd_iter)
        noise = [self.noise_fun(psd).to_timeseries().numpy() for psd in psds]
        noise = np.stack(noise, axis=0)
        logging.debug("Whitening noise")
        noise = whiten(noise, delta_t=delta_t, segment_duration=self.segment_duration,
                max_filter_duration=self.max_filter_duration, psd=whitening_psds,
                low_frequency_cutoff=self.low_frequency_cutoff)
        logging.debug("Noise whitened")
        return noise, psds, whitening_psds

def generate_spin(low, high, generator):
    unnormalized_spin = generator.standard_normal(size=3)
    old_spin_norm = np.sqrt(sum(unnormalized_spin**2))
    new_spin_norm = generator.uniform(low, high)
    return unnormalized_spin*new_spin_norm/old_spin_norm

def add_spins(parameter_dictionary, i, spin_vector):
    vec_name = 'spin%i' % i
    for direction, num in zip(('x', 'y', 'z'), spin_vector):
        parameter_dictionary[vec_name + direction] = num
    return parameter_dictionary

def generate_dataset(samples, detectors, low_frequency_cutoff=20., verbose=False,
        approximant="IMRPhenomD", noise_getter_iter=None,
        sample_length=2048):
    """Generate a dataset that can be used for training and/or
    validation purposes.
    
    Arguments
    ---------
    samples : int
        The number of training samples to generate.
    detectors : iterable of pycbc.detector.Detector objects.
    low_frequency_cutoff : {float, 20.}
        Lower frequency cutoff for signal generation and processing.
    verbose : {bool, False}
        Print update messages.
    approximant : {IMRPhenomD, IMRPhenomXPHM}
        Waveform approximant to be passed to pycbc.waveform.get_td_waveform.
    psd_filename : {string, None}
        Path to file containing PSDs to be used. If None, aLIGOZeroDetHighPower is used.
        Overridden by real_noise_filename.
    real_noise_filename : {string, None}
        Path to file containing real noise to be used.
        If None, Gaussian noise will be generated.
    """

    ### Check the approximant selection
    if approximant=='IMRPhenomD':
        spins_required = False
    elif approximant=='IMRPhenomXPHM':
        spins_required = True
    else:
        raise ValueError("Approximant %s not allowed." % approximant)

    ### Initialize the random distribution
    skylocation_dist = pycbc.distributions.sky_location.UniformSky()

    ### Generate data
    datasets = []
    num_waveforms, num_noises = samples
    logging.info(("Generating dataset with %i injections and %i pure "
                "noise samples") % (num_waveforms, num_noises))

    sample_duration = sample_length*delta_t

    colored_sample_length = sample_length + max_filter_length
    colored_sample_duration = colored_sample_length*delta_t

    noises = []
    waveforms = []
    for i in tqdm(range(num_waveforms+num_noises), disable=(not verbose), ascii=True):
        noise, psds, whitening_psds = next(noise_getter_iter)
        logging.debug("Starting new sample")
        is_waveform = i<num_waveforms
        # Generate noise
        noises.append(noise)

        # If in the first part of the dataset, generate waveform
        if is_waveform:
            logging.debug("Sample includes waveform, generating parameters")
            # Generate source parameters
            waveform_kwargs = {'delta_t': delta_t, 'f_lower': low_frequency_cutoff}
            waveform_kwargs['approximant'] = approximant
            masses = np_gen.uniform(10., 50., 2)
            waveform_kwargs['mass1'] = max(masses)
            waveform_kwargs['mass2'] = min(masses)
            angles = np_gen.uniform(0., 2*np.pi, 3)
            waveform_kwargs['coa_phase'] = angles[0]
            waveform_kwargs['inclination'] = angles[1]
            declination, right_ascension = skylocation_dist.rvs()[0]
            pol_angle = angles[2]
            if spins_required:
                for i in (1, 2):
                    add_spins(waveform_kwargs, i, generate_spin(0., 0.99, np_gen))
            # Take the injection time randomly in the LIGO O3a era
            injection_time = np_gen.uniform(1238166018, 1253977218)
            # Generate the full waveform
            logging.debug("Generating waveform")
            waveform = pycbc.waveform.get_td_waveform(**waveform_kwargs)
            logging.debug("Waveform generated")
            h_plus, h_cross = waveform
            # Properly time and project the waveform
            start_time = injection_time + h_plus.get_sample_times()[0]
            h_plus.start_time = start_time
            h_cross.start_time = start_time
            h_plus.append_zeros(colored_sample_length)
            h_cross.append_zeros(colored_sample_length)
            h_plus.prepend_zeros(colored_sample_length)
            h_cross.prepend_zeros(colored_sample_length)
            strains = [det.project_wave(h_plus, h_cross, right_ascension, declination, pol_angle) for det in detectors]
            # Place merger randomly within the window between 0.5 s and 0.7 s of the time series and form the PyTorch sample
            time_placement = np_gen.uniform(sample_duration-0.4, sample_duration-0.2)+max_filter_halfduration
            time_interval = injection_time-time_placement
            time_interval = (time_interval, time_interval+colored_sample_duration-1.e-3)    # -1.e-3 to not get a too long strain
            strains = [strain.time_slice(*time_interval) for strain in strains]
            for strain in strains:
                to_append = colored_sample_length - len(strain)
                if to_append>0:
                    strain.append_zeros(to_append)
            # Compute network SNR, rescale to generated target network SNR and inject into noise
            logging.debug("Computing network SNR of strain at %s using PSD at %s" % (strains[0].dtype, psds[0].dtype))
            network_snr = np.sqrt(sum([pycbc.filter.matchedfilter.sigmasq(strain, psd=psd,
                low_frequency_cutoff=low_frequency_cutoff) for strain, psd in zip(strains, psds)]))
            waveform = np.stack([strain.numpy() for strain in strains], axis=0)/network_snr
            logging.debug("Waveform generated and projected, whitening")
            waveform = whiten(waveform, delta_t=delta_t, segment_duration=segment_duration,
                    max_filter_duration=max_filter_duration, psd=whitening_psds,
                    low_frequency_cutoff=low_frequency_cutoff)
            logging.debug("Waveform whitened")
            waveforms.append(waveform)
        
    # Merge into just two tensors (more memory efficient) and initialize dataset
    noises = np.stack(noises, axis=0)
    waveforms = np.stack(waveforms, axis=0)
    return Dataset(noises=noises, waveforms=waveforms)


def main():
    parser = ArgumentParser(description="CNN training data generation script for the MLGWSC-1 mock data challenge. Written by Ondřej Zelenka.")

    parser.add_argument('-o', '--output-file', type=str, help="Path to the file where the datasets will be stored.")
    parser.add_argument('-a', '--approximant', type=str, default="IMRPhenomD", help="Name of waveform approximant to be passed to pycbc.waveform.get_td_waveform. Allowed: IMRPhenomD, IMRPhenomXPHM. Default: IMRPhenomD.")
    parser.add_argument('-d', '--detectors', type=int, default=1, help="Number of detectors.")
    parser.add_argument('-s', '--sample-length', type=int, default=2048, help="Number of values (length) of each sample. Sampling rate fixed at 2048 Hz. Default: 2048 (i.e. 1 s samples)")
    parser.add_argument('--training-samples', type=int, nargs=2, default=[10000, 10000], help="Numbers of training samples as 'injections' 'pure noise samples'. Default: 100000 100000")
    parser.add_argument('--validation-samples', type=int, nargs=2, default=[2000, 2000], help="Numbers of validation samples as 'injections' 'pure noise samples'. Default: 20000 20000")
    parser.add_argument('--psd-file', type=str, default=None, help="Path to file with PSDs for noise generation. If not given, aLIGOZeroDetHighPower will be used.")
    parser.add_argument('--real-noise-file', type=str, default=None, help="Path to file with real noise to be used in training/validation dataset. If not given, Gaussian noise will be generated.")
    parser.add_argument('--verbose', action='store_true', help="Print update messages.")
    parser.add_argument('--debug', action='store_true', help="Show debug messages.")
    parser.add_argument('--force', action='store_true', help="Overwrite existing output file.")


    args = parser.parse_args()

    ### Set up logging
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARN
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s', level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    ### Check existence of output file
    if os.path.isfile(args.output_file) and not args.force:
        raise RuntimeError("Output file exists.")
    else:
        pass
    filemode = 'w' if args.force else 'w-'

    ### Create the detectors
    detectors_abbr = ('H1', 'L1', 'V1', 'K1')[:args.detectors]
    detectors = []
    for det_abbr in detectors_abbr:
        detectors.append(pycbc.detector.Detector(det_abbr))

    sample_duration = args.sample_length*delta_t

    colored_sample_length = args.sample_length + max_filter_length
    colored_sample_duration = colored_sample_length*delta_t

    low_frequency_cutoff = 20.

    ### Create the power spectral densities of the respective detectors
    if args.real_noise_file is None:
        noise_getter = GaussianNoiseGetter(fpath=args.psd_file, detectors=detectors,
            sample_length=colored_sample_length, low_frequency_cutoff=low_frequency_cutoff,
            segment_duration=segment_duration, max_filter_duration=max_filter_duration)
    else:
        noise_getter = HDFNoiseGetter(args.real_noise_file, detectors=detectors)
    noise_getter_iter = iter(noise_getter)

    ### Initialize network
    logging.debug("Generating training dataset.")
    TrainDS = generate_dataset(args.training_samples, detectors, verbose=args.verbose,
        approximant=args.approximant, noise_getter_iter=noise_getter_iter, sample_length=args.sample_length)
    with h5py.File(args.output_file, filemode) as outfile:
        TrainDS.save(outfile, 'training')

    logging.debug("Generating validation dataset.")
    ValidDS = generate_dataset(args.validation_samples, detectors, verbose=args.verbose,
        approximant=args.approximant, noise_getter_iter=noise_getter_iter, sample_length=args.sample_length)
    with h5py.File(args.output_file, 'a') as outfile:
        ValidDS.save(outfile, 'validation')

if __name__=='__main__':
    main()
  
