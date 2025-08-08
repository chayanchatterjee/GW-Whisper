#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Two-detector inference script. Adapted from https://github.com/ondrzel/ml-gw-search/blob/main/mlgwsc-1/apply.py

"""

from __future__ import annotations

import os
import sys
import time as t
import logging
import multiprocessing as mp
from argparse import ArgumentParser
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

from transformers import WhisperModel
from peft import PeftModel
from ml4gw.transforms import QScan

# Optional pycbc utils used in whitening
import pycbc.waveform  # noqa: F401
import pycbc.noise     # noqa: F401
import pycbc.psd
import pycbc.distributions  # noqa: F401
import pycbc.detector       # noqa: F401


# =============================================================================
# Logging
# =============================================================================

def configure_logging(verbose: bool = False, debug: bool = False) -> None:
    level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
    logging.basicConfig(
        format='%(levelname)s | %(asctime)s: %(message)s',
        level=level,
        datefmt='%d-%m-%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# =============================================================================
# Whitening / preprocessing
# =============================================================================

def whiten(
    strain: np.ndarray,
    delta_t: float = 1.0 / 2048.0,
    segment_duration: float = 0.5,
    max_filter_duration: float = 0.25,
    trunc_method: Optional[str] = "hann",
    remove_corrupted: bool = True,
    low_frequency_cutoff: Optional[float] = None,
    psd: Optional[np.ndarray] = None,
    return_psd: bool = False,
    **kwargs: Any,
) -> np.ndarray | Tuple[np.ndarray, Any]:
    """Whiten a 1D or 2D strain array using PyCBC inverse spectrum truncation."""
    if strain.ndim == 1:
        from pycbc.psd import inverse_spectrum_truncation, interpolate
        colored_ts = pycbc.types.TimeSeries(strain, delta_t=delta_t)

        if psd is None:
            psd_est = colored_ts.psd(segment_duration, **kwargs)
        elif isinstance(psd, np.ndarray):
            assert psd.ndim == 1
            logging.warning("Assuming PSD delta_f from delta_t and PSD length; ensure even TS length.")
            assumed_duration = delta_t * (2 * len(psd) - 2)
            psd_est = pycbc.types.FrequencySeries(psd, delta_f=1.0 / assumed_duration)
        elif isinstance(psd, pycbc.types.FrequencySeries):
            psd_est = psd
        else:
            raise ValueError("Unknown PSD format.")

        unprocessed_psd = psd_est
        psd_est = interpolate(psd_est, colored_ts.delta_f)
        max_filter_len = int(max_filter_duration * colored_ts.sample_rate)

        psd_est = inverse_spectrum_truncation(
            psd_est,
            max_filter_len=max_filter_len,
            low_frequency_cutoff=low_frequency_cutoff,
            trunc_method=trunc_method,
        )

        inv_psd = 1.0 / psd_est
        white_ts = (colored_ts.to_frequencyseries() * inv_psd ** 0.5).to_timeseries().numpy()

        if remove_corrupted:
            white_ts = white_ts[max_filter_len // 2 : (len(colored_ts) - max_filter_len // 2)]

        if return_psd:
            return white_ts, unprocessed_psd
        return white_ts

    if strain.ndim == 2:
        # Per-channel whitening
        if isinstance(psd, np.ndarray) and psd.ndim == 1:
            psd_list = [psd for _ in strain]
        elif (psd is None) or isinstance(psd, pycbc.types.FrequencySeries):
            psd_list = [psd for _ in strain]
        else:
            assert len(psd) == len(strain)
            psd_list = psd

        results = [
            whiten(
                sd,
                delta_t=delta_t,
                segment_duration=segment_duration,
                max_filter_duration=max_filter_duration,
                trunc_method=trunc_method,
                remove_corrupted=remove_corrupted,
                low_frequency_cutoff=low_frequency_cutoff,
                psd=psd_i,
                return_psd=return_psd,
                **kwargs,
            )
            for sd, psd_i in zip(strain, psd_list)
        ]
        if return_psd:
            psds = [r[1] for r in results]
            whites = np.stack([r[0] for r in results], axis=0)
            return whites, psds
        return np.stack(results, axis=0)

    raise ValueError("Strain must be 1D or 2D.")


def get_clusters(
    triggers: Dict[str, List[List[float]]],
    cluster_threshold: float = 0.35
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cluster per-key triggers; pick max within each cluster."""
    all_clusters: List[List[List[float]]] = []
    for trig_list in triggers.values():
        clusters: List[List[List[float]]] = []
        for trig in trig_list:
            t_new = trig[0]
            if not clusters or (t_new - clusters[-1][-1][0]) > cluster_threshold:
                clusters.append([trig])
            else:
                clusters[-1].append(trig)
        all_clusters.extend(clusters)

    logging.info("Clustering produced %d clusters (max-centered).", len(all_clusters))

    times, vals, tvars = [], [], []
    for cl in all_clusters:
        ts = [x[0] for x in cl]
        vs = np.array([x[1] for x in cl])
        k = int(np.argmax(vs))
        times.append(ts[k])
        vals.append(vs[k])
        tvars.append(0.2)
    return np.array(times), np.array(vals), np.array(tvars)


# =============================================================================
# Data slicing
# =============================================================================

class SegmentSlicer:
    """Slice multi-detector strain after (optional) whitening for inference."""
    def __init__(
        self,
        infile: h5py.File,
        key: str,
        step_size: float = 0.1,
        peak_offset: float = 0.6,
        slice_length: int = 2048,
        detectors: Optional[List[str]] = None,
        white: bool = False,
        whitened_file: Optional[str] = None,
        save_psd: bool = False,
        low_frequency_cutoff: Optional[float] = None,
        segment_duration: float = 0.5,
        max_filter_duration: float = 0.25,
    ) -> None:
        self.step_size = step_size
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors or ["H1", "L1"]
        self.white = white

        # sampling interval is stored inverted in attrs
        self.delta_t = 1.0 / (1.0 / infile[self.detectors[0]][key].attrs["delta_t"])
        self.index_step_size = int(self.step_size / self.delta_t)
        self.time_step_size = self.delta_t * self.index_step_size

        self.segment_duration = segment_duration
        self.max_filter_duration = max_filter_duration
        self.whitened_file = whitened_file
        self.low_frequency_cutoff = low_frequency_cutoff
        self.key = key

        self.dss = [infile[det][self.key] for det in self.detectors]
        self.start_time = self.dss[0].attrs["start_time"]
        for ds in self.dss:
            assert ds.attrs["start_time"] == self.start_time

        logging.debug(
            "SegmentSlicer init: step_idx=%d, step_t=%.3f, key=%s, dtype=%s",
            self.index_step_size, self.time_step_size, self.key, self.dss[0].dtype,
        )
        self.process(save_psd)

    def process(self, save_psd: bool) -> None:
        whitened_dss: List[np.ndarray] = []
        self.psds: List[Any] = []
        for ds, det in zip(self.dss, self.detectors):
            if self.white:
                new_ds = ds[()]
            else:
                new_ds = whiten(
                    ds,
                    delta_t=self.delta_t,
                    low_frequency_cutoff=self.low_frequency_cutoff,
                    segment_duration=self.segment_duration,
                    max_filter_duration=self.max_filter_duration,
                    return_psd=save_psd,
                )
                if save_psd:
                    new_ds, psd = new_ds
                    self.psds.append(psd)
            whitened_dss.append(new_ds)

            if self.whitened_file is not None:
                with h5py.File(self.whitened_file, "a") as wfile:
                    wfile.require_group(det).create_dataset(self.key, data=new_ds)

        self.dss = np.stack(whitened_dss, axis=0)
        if not self.white:
            # whitening discards edges; compensate start time (approx 0.125 s)
            self.start_time += 0.125
        self.white = True

    def __len__(self) -> int:
        full_slice = self.slice_length if self.white else 512 + self.slice_length
        return 1 + (self.dss.shape[1] - full_slice) // self.index_step_size

    def __iter__(self) -> "SegmentSlicer":
        self.current_index = 0
        self.current_time = self.start_time
        return self

    def get_next_slice(self) -> Tuple[np.ndarray, float]:
        if self.current_index + self.slice_length > self.dss.shape[1]:
            raise StopIteration
        sl = self.dss[:, self.current_index : self.current_index + self.slice_length]
        ts = self.current_time + self.peak_offset
        self.current_index += self.index_step_size
        self.current_time += self.time_step_size
        return sl, ts

    def __next__(self) -> Tuple[np.ndarray, float]:
        return self.get_next_slice()

    # shared-memory helpers
    def split_and_pop(self, ds_dict, size: int) -> None:
        data = self.dss
        self.dss = None
        max_split_index = int(np.floor(data.shape[1] / size)) * size
        split_indices = list(range(size, max_split_index + 1, size))
        self.ds_dict_keys: List[str] = []
        for i, ds in enumerate(np.split(data, split_indices, axis=1)):
            k = f"{self.key}_{i}"
            self.ds_dict_keys.append(k)
            logging.debug("Saving chunk %s to shared dict", k)
            ds_dict[k] = ds
        logging.debug("Worker %s finished!", self.key)

    def stack_and_load(self, ds_dict) -> None:
        dss = [ds_dict.pop(k) for k in self.ds_dict_keys]
        self.dss = np.concatenate(dss, axis=1)


class TorchSegmentSlicer(SegmentSlicer, IterableDataset):
    """IterableDataset wrapper returning torch tensors."""
    def __init__(self, *args, **kwargs) -> None:
        IterableDataset.__init__(self)
        SegmentSlicer.__init__(self, *args, **kwargs)

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        sl, ts = self.get_next_slice()
        return torch.from_numpy(sl), torch.tensor(ts)


# =============================================================================
# Model
# =============================================================================

class QTransformAdapter(nn.Module):
    """Convert raw strain [B, D, T] into Whisper-like features [B, D, F, T*]."""
    def __init__(
        self,
        kernel_length: float = 1.0,
        sample_rate: int = 2048,
        q_range: List[int] = [4, 128],
        spectrogram_shape: List[int] = [512, 512],
        target_shape: Tuple[int, int] = (80, 3000),
        n_detectors: int = 2,
    ) -> None:
        super().__init__()
        self.n_detectors = n_detectors
        self.q_transform = QScan(
            duration=kernel_length,
            sample_rate=sample_rate,
            spectrogram_shape=spectrogram_shape,
            qrange=q_range,
        )
        self.freq_adapter = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),
        )
        self.final_pool = nn.AdaptiveAvgPool2d(target_shape)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.film_gamma = nn.Parameter(torch.ones(self.n_detectors))
        self.film_beta = nn.Parameter(torch.zeros(self.n_detectors))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        B, D, _ = x.shape
        outs: List[torch.Tensor] = []
        for i in range(D):
            with torch.no_grad():
                qspec = self.q_transform(x[:, i]).unsqueeze(1)  # [B,1,F,Tq]
            y = self.freq_adapter(qspec)                        # [B,1,F',T']
            y = self.final_pool(y).squeeze(1)                   # [B,F,T*]
            y = self.scale * y + self.bias
            y = y * self.film_gamma[i] + self.film_beta[i]      # FiLM
            outs.append(y)
        return torch.stack(outs, dim=1)


class GWWhisperClassifier(nn.Module):
    """Q-Adapter -> Whisper encoder (per-detector) -> MLP classifier."""
    def __init__(
        self,
        whisper_encoder: nn.Module,
        n_detectors: int,
        num_classes: int = 2,
        q_adapter: Optional[QTransformAdapter] = None,
        use_last_token: bool = True,
    ) -> None:
        super().__init__()
        self.n_detectors = n_detectors
        self.encoder = whisper_encoder
        self.adapter = q_adapter if q_adapter is not None else QTransformAdapter(n_detectors=n_detectors)
        self.use_last_token = use_last_token

        hidden = self.encoder.config.d_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden * n_detectors, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),  # removed in USR mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.adapter(x)  # [B, D, F, T*]
        reps: List[torch.Tensor] = []
        for i in range(feats.size(1)):
            enc = self.encoder(feats[:, i])
            seq = enc.last_hidden_state
            reps.append(seq[:, -1, :] if self.use_last_token else seq.mean(dim=1))
        combined = torch.cat(reps, dim=1)
        return self.classifier(combined)


def remove_softmax_from_classifier(model: GWWhisperClassifier) -> None:
    """Switch to USR mode (raw logits)."""
    if isinstance(model.classifier, nn.Sequential) and len(model.classifier) > 0:
        layers = list(model.classifier.children())
        if isinstance(layers[-1], nn.Softmax):
            model.classifier = nn.Sequential(*layers[:-1])


# =============================================================================
# Builders
# =============================================================================

def build_encoder_with_lora(lora_weights_path: str, device: str) -> nn.Module:
    whisper_model = WhisperModel.from_pretrained("openai/whisper-tiny")
    encoder = whisper_model.encoder.to(device)
    encoder.gradient_checkpointing_enable()
    peft_encoder = PeftModel.from_pretrained(encoder, lora_weights_path).to(device)
    return peft_encoder


def build_model(
    lora_weights_path: str,
    dense_weights_path: str,
    adapter_weights_path: str,
    device: str,
    n_detectors: int = 2,
    usr: bool = False,
) -> GWWhisperClassifier:
    adapter = QTransformAdapter(n_detectors=n_detectors).to(device)
    adapter.load_state_dict(torch.load(adapter_weights_path, map_location=device))
    encoder = build_encoder_with_lora(lora_weights_path, device)
    model = GWWhisperClassifier(whisper_encoder=encoder, n_detectors=n_detectors, q_adapter=adapter).to(device)
    model.classifier.load_state_dict(torch.load(dense_weights_path, map_location=device))
    if usr:
        remove_softmax_from_classifier(model)
    return model


# =============================================================================
# Evaluation
# =============================================================================

def worker(inp: Dict[str, Any]) -> TorchSegmentSlicer:
    """Pool worker to prepare one slicer instance."""
    fpath = inp.pop("fpath")
    key = inp.pop("key")
    wdata_dict = inp.pop("wdata_dict", None)

    with h5py.File(fpath, "r") as infile:
        slicer = TorchSegmentSlicer(infile, key, **inp)

    if wdata_dict is not None:
        logging.debug("Worker %s: splitting & sharing", key)
        slicer.split_and_pop(wdata_dict, 10 ** 6)
    else:
        logging.debug("Worker %s: returning slicer", key)
    return slicer


def evaluate_slices(
    slicer: TorchSegmentSlicer,
    network: nn.Module,
    device: str = "cuda",
    trigger_threshold: float = 0.2,
    verbose: bool = False,
) -> Tuple[List[List[float]], List[np.ndarray]]:
    """Run `network` over all slices; return triggers and raw scores."""
    new_triggers: List[List[float]] = []
    all_vals: List[np.ndarray] = []

    loader = DataLoader(slicer, batch_size=256, pin_memory=True, shuffle=False)
    num_batches = (len(slicer) + 256 - 1) // 256
    logging.info("Evaluating %s (%d slices → %d batches)", slicer.key, len(slicer), num_batches)

    with torch.no_grad():
        for slice_batch, slice_times in tqdm(
            loader,
            desc=f"Evaluating {slicer.key}",
            total=num_batches,
            leave=False,
            disable=not verbose,
        ):
            slice_batch = slice_batch.to(device=device, dtype=torch.float32)
            slice_times = slice_times.to(device=device, non_blocking=True)

            outputs = network(slice_batch)              # [B, 2] (prob or logits)
            signal_scores = outputs[:, 0]               # consistent signal score
            all_vals.append(signal_scores.cpu().numpy())

            keep = signal_scores > trigger_threshold
            for ts, k, sc in zip(slice_times, keep, signal_scores):
                if k.item():
                    new_triggers.append([ts.item(), sc.item()])

    return new_triggers, all_vals


def get_triggers(
    lora_weights_path: str,
    dense_weights_path: str,
    adapter_weights_path: str,
    inputfile: str,
    step_size: float = 0.1,
    trigger_threshold: float = 0.2,
    device: str = "cuda",
    verbose: bool = False,
    white: bool = False,
    whitened_file: Optional[str] = None,
    low_frequency_cutoff: float = 20.0,
    num_workers: int = -1,
    usr: bool = False,
) -> Tuple[Dict[str, List[List[float]]], List[np.ndarray]]:
    """Compute triggers for all segments in `inputfile` using trained model."""
    # Build model once
    network = build_model(
        lora_weights_path=lora_weights_path,
        dense_weights_path=dense_weights_path,
        adapter_weights_path=adapter_weights_path,
        device=device,
        n_detectors=2,
        usr=usr,
    )
    network.eval()

    if num_workers < 0:
        num_workers = mp.cpu_count()

    detectors = ["H1", "L1"]
    if whitened_file is not None:
        with h5py.File(whitened_file, "w") as wfile:
            for d in detectors:
                wfile.create_group(d)

    triggers: Dict[str, List[List[float]]] = {}
    all_vals_all: List[np.ndarray] = []

    arguments: List[Dict[str, Any]] = []
    with h5py.File(inputfile, "r") as infile:
        det_grp = next(iter(infile.values()))
        for key in list(det_grp.keys()):
            tmp = dict(
                fpath=inputfile,
                key=key,
                step_size=step_size,
                low_frequency_cutoff=low_frequency_cutoff,
                white=white,
                whitened_file=whitened_file,
                detectors=detectors,
            )
            arguments.append(tmp)

        arguments.sort(key=(lambda x: len(infile[x["detectors"][0]][x["key"]])), reverse=True)

    if num_workers > 0:
        mp.set_start_method("forkserver", force=True)
        if whitened_file is not None:
            m = mp.Manager()
            wdata_dict = m.dict()
            for tmp in arguments:
                tmp["wdata_dict"] = wdata_dict
        else:
            wdata_dict = None

        with mp.Pool(num_workers) as pool:
            for slicer in tqdm(
                pool.imap_unordered(worker, arguments),
                disable=not verbose,
                ascii=True,
                total=len(arguments),
            ):
                if wdata_dict is not None:
                    slicer.stack_and_load(wdata_dict)
                sub_trigs, sub_vals = evaluate_slices(
                    slicer,
                    network,
                    device=device,
                    trigger_threshold=trigger_threshold,
                    verbose=verbose,
                )
                triggers[slicer.key] = sub_trigs
                all_vals_all.extend(sub_vals)
    else:
        for kwargs in tqdm(arguments, disable=not verbose, ascii=True):
            slicer = worker(kwargs)
            sub_trigs, sub_vals = evaluate_slices(
                slicer,
                network,
                device=device,
                trigger_threshold=trigger_threshold,
                verbose=verbose,
            )
            triggers[slicer.key] = sub_trigs
            all_vals_all.extend(sub_vals)

    return dict(sorted(triggers.items(), key=lambda x: x[0])), all_vals_all


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> Any:
    parser = ArgumentParser(description="Apply a trained two-detector GW-Whisper model and save triggers.")
    parser.add_argument("--verbose", action="store_true", help="Print update messages.")
    parser.add_argument("--debug", action="store_true", help="Show debug messages.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file.")

    parser.add_argument("inputfile", type=str, help="Path to input HDF5.")
    parser.add_argument("outputfile", type=str, help="Path to output HDF5 (must not exist unless --force).")

    parser.add_argument("--white", action="store_true", help="Input is already whitened (skip whitening).")
    parser.add_argument("--softmax", action="store_true", help="Use Softmax outputs (default is USR logits).")
    parser.add_argument("--coinc-window", type=float, default=0.1, help="(Reserved) coincidence window; not used.")
    parser.add_argument("--lora-weights", type=str, required=True, help="Path to LoRA weights dir.")
    parser.add_argument("--dense-weights", type=str, required=True, help="Path to dense head weights (.pth).")
    parser.add_argument("--adapter-weights", type=str, required=True, help="Path to Q-Adapter weights (.pt).")
    parser.add_argument("-t", "--trigger-threshold", type=float, default=-0.5, help="Trigger threshold on signal score.")
    parser.add_argument("--step-size", type=float, default=0.1, help="Sliding window step (s).")
    parser.add_argument("--cluster-threshold", type=float, default=0.35, help="Time gap for clustering (s).")
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. 'cuda', 'cuda:1', 'cpu'.")
    parser.add_argument("--debug-triggers-file", type=str, default=None, help="Save pre-cluster triggers here (optional).")
    parser.add_argument("--debug-whitened-file", type=str, default=None, help="Save whitened inputs to this HDF5 (optional).")
    parser.add_argument("--num-workers", type=int, default=8, help="Process pool size (0 => sequential).")
    return parser.parse_args()


def main() -> None:
    start_time = t.time()
    args = parse_args()
    configure_logging(verbose=args.verbose, debug=args.debug)

    # Sanity checks
    if os.path.isfile(args.outputfile) and not args.force:
        raise RuntimeError("Output file exists. Use --force to overwrite.")
    if args.debug_whitened_file is not None and os.path.isfile(args.debug_whitened_file) and not args.force:
        raise RuntimeError("Whitened file exists. Use --force to overwrite.")
    if args.debug_triggers_file is not None and os.path.isfile(args.debug_triggers_file) and not args.force:
        raise RuntimeError("Triggers file exists. Use --force to overwrite.")

    # Run inference
    triggers, all_vals = get_triggers(
        lora_weights_path=args.lora_weights,
        dense_weights_path=args.dense_weights,
        adapter_weights_path=args.adapter_weights,
        inputfile=args.inputfile,
        step_size=args.step_size,
        trigger_threshold=args.trigger_threshold,
        device=args.device,
        verbose=args.verbose,
        white=args.white,
        whitened_file=args.debug_whitened_file,
        low_frequency_cutoff=20.0,
        num_workers=args.num_workers,
        usr=not args.softmax,
    )

    logging.info(
        "Total slices above threshold %.3f: %d",
        args.trigger_threshold,
        sum(len(v) for v in triggers.values()),
    )

    # Optional: save raw (pre-cluster) triggers
    if args.debug_triggers_file is not None:
        with h5py.File(args.debug_triggers_file, "w") as dbg:
            for key, trig_list in triggers.items():
                dbg.create_dataset(key, data=np.array(trig_list, dtype=np.float32))

    # Cluster & save main output
    time_arr, stat_arr, var_arr = get_clusters(triggers, args.cluster_threshold)
    all_vals_flat = np.concatenate(all_vals).astype("float32") if len(all_vals) else np.array([], dtype="float32")

    with h5py.File(args.outputfile, "w") as outfile:
        logging.debug("Saving clustered triggers to %s", args.outputfile)
        outfile.create_dataset("time", data=time_arr)
        outfile.create_dataset("stat", data=stat_arr)
        outfile.create_dataset("var", data=var_arr)
        outfile.create_dataset("all_vals", data=all_vals_flat)

    print(f"Total execution time: {t.time() - start_time:.2f} seconds")
    sys.stdout.flush()


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()