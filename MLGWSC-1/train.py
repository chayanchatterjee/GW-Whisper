#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GW-Whisper (Q-Scan front-end) – Modular training script with documentation.

This script trains a binary classifier for gravitational-wave detection using:
  • A per-detector Q-Transform adapter (QScan) to produce time–frequency inputs
  • A Whisper encoder backbone (with optional LoRA/DoRA)
  • A small MLP head for classification

Features:
  • Optional contrastive pretraining (InfoNCE) on injection/noise pairs
  • Supervised fine-tuning with checkpointing, resume, and early stopping
  • Clear modular builders and strong docstrings

Original copyright:
  Copyright 2022 Ondřej Zelenka (with modifications)
Licensed under the Apache License, Version 2.0
"""

from __future__ import annotations

# ========== Standard library ==========
import os
import fnmatch
import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# ========== Third-party ==========
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset as TorchDataset, ConcatDataset

# Transformers / PEFT
from transformers import WhisperModel
from peft import LoraConfig, get_peft_model

# ml4gw
from ml4gw.transforms import QScan


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set RNG seeds across numpy/torch for reproducibility.

    Args:
        seed: Random seed.
        deterministic: If True, configures cudnn for deterministic ops.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# =============================================================================
# Q-Transform Adapter
# =============================================================================

class QTransformAdapter(nn.Module):
    """Converts raw strain into Whisper-compatible feature maps via Q-Scan.

    Pipeline:
      for each detector independently:
        strain [B, T] -> QScan -> [B, 1, F, T'] -> small CNN -> [B, 1, F', T'']
        -> adaptive pool to target_shape -> FiLM (per-detector) -> [B, F*, T*]
      Stack across detectors -> [B, D, F*, T*]

    Notes:
      • The Whisper encoder is called with each detector’s [B, F*, T*] map.
      • FiLM provides a lightweight per-detector modulation (scale/shift).

    Args:
        kernel_length: Duration (s) of Q-Scan kernel/window.
        sample_rate: Sampling rate of the input strain.
        q_range: [q_min, q_max] range for Q-Scan tiles.
        spectrogram_shape: Base (freq, time) resolution for Q-Scan before pooling.
        target_shape: (freq', time') shape after adaptive pooling.
        n_detectors: Number of detectors (channels) in input.
    """
    def __init__(
        self,
        kernel_length: float = 1.0,
        sample_rate: int = 2048,
        q_range: List[int] = [4, 128],
        spectrogram_shape: List[int] = [128, 128],
        target_shape: Tuple[int, int] = (80, 3000),
        n_detectors: int = 2,
    ):
        super().__init__()
        self.n_detectors = n_detectors
        self.q_transform = QScan(
            duration=kernel_length,
            sample_rate=sample_rate,
            spectrogram_shape=spectrogram_shape,
            qrange=q_range,
        )

        # A slightly beefier adapter than the original.
        self.freq_adapter = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )

        self.final_pool = nn.AdaptiveAvgPool2d(target_shape)

        # Global (all-detector) affine
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

        # FiLM (per-detector) affine
        self.film_gamma = nn.Parameter(torch.ones(self.n_detectors))
        self.film_beta = nn.Parameter(torch.zeros(self.n_detectors))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, D, T] raw strain

        Returns:
            Tensor shaped [B, D, F*, T*] (after pooling and FiLM)
        """
        B, D, _ = x.shape
        outs: List[torch.Tensor] = []
        for i in range(D):
            # Q-Scan (can be heavy; keep no_grad if QScan is not learnable)
            with torch.no_grad():
                qspec = self.q_transform(x[:, i]).unsqueeze(1)  # [B,1,F,T]

            y = self.freq_adapter(qspec)            # [B,1,F',T']
            y = self.final_pool(y).squeeze(1)       # [B,F*,T*]
            y = self.scale * y + self.bias          # global affine
            y = y * self.film_gamma[i] + self.film_beta[i]  # FiLM
            outs.append(y)
        return torch.stack(outs, dim=1)             # [B, D, F*, T*]


# =============================================================================
# Whisper-based Classifier
# =============================================================================

class GWWhisperClassifier(nn.Module):
    """Classifier: Q-Adapter -> Whisper encoder (per-detector) -> MLP head.

    For each detector's feature map, we extract the encoder's last hidden state
    and take the final token embedding ([:, -1, :]) as a sequence summary.
    Detector embeddings are concatenated before the MLP.

    Args:
        whisper_encoder: The encoder submodule of a WhisperModel (possibly PEFT).
        n_detectors: Number of detectors.
        num_classes: Output classes (default 2: [waveform, noise]).
        q_adapter: Optional external adapter; created if None.
    """
    def __init__(
        self,
        whisper_encoder: nn.Module,
        n_detectors: int,
        num_classes: int = 2,
        q_adapter: Optional[QTransformAdapter] = None,
    ):
        super().__init__()
        self.n_detectors = n_detectors
        self.encoder = whisper_encoder
        self.adapter = q_adapter if q_adapter is not None else QTransformAdapter(n_detectors=n_detectors)

        hidden_size = self.encoder.config.d_model

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * self.n_detectors, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, D, T] raw strain

        Returns:
            probs: [B, num_classes] class probabilities
        """
        feats = self.adapter(x)                       # [B, D, F*, T*]
        reps: List[torch.Tensor] = []
        for i in range(feats.size(1)):
            enc = self.encoder(feats[:, i])           # ModelOutput
            reps.append(enc.last_hidden_state[:, -1, :])  # [B, hidden]
        combined = torch.cat(reps, dim=1)             # [B, hidden*D]
        return self.classifier(combined)


# =============================================================================
# Datasets
# =============================================================================

class BinaryGWDataset(TorchDataset):
    """Simple dataset mixing injections and noise-only samples.

    Labeling convention:
      • First `len(waveforms)` indices return an injected signal (class=0 ⇒ [1,0])
      • Remaining indices return noise-only (class=1 ⇒ [0,1])

    Args:
        noises: np.ndarray or torch.Tensor of shape [N, D, T]
        waveforms: np.ndarray or torch.Tensor of shape [M, D, T]
        store_device: Device to store tensors ("cpu" or "cuda[:idx]").
        train_device: Device to move batches to during __getitem__.
        snr_range: Uniform range for SNR scaling of injections.
    """
    def __init__(
        self,
        noises=None,
        waveforms=None,
        store_device: str = "cpu",
        train_device: str = "cuda",
        snr_range: Tuple[float, float] = (5.0, 15.0),
    ):
        super().__init__()
        self.noises = noises
        self.waveforms = waveforms
        self.store_device = store_device
        self.train_device = train_device
        self.snr_range = snr_range

        # one-hot labels (binary)
        self.wave_label = torch.tensor([1.0, 0.0], dtype=torch.float32, device=train_device)
        self.noise_label = torch.tensor([0.0, 1.0], dtype=torch.float32, device=train_device)

        if self.noises is not None:
            self._to_tensors()

        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.noises)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if i < len(self.waveforms):
            snr_val = self.rng.uniform(*self.snr_range)
            signal = self.noises[i] + snr_val * self.waveforms[i]
            label = self.wave_label
        else:
            signal = self.noises[i]
            label = self.noise_label

        if signal.device.type != self.train_device.split(":")[0]:
            signal = signal.to(self.train_device)
        return signal, label

    def _to_tensors(self) -> None:
        if isinstance(self.noises, np.ndarray):
            self.noises = torch.from_numpy(self.noises)
        if isinstance(self.waveforms, np.ndarray):
            self.waveforms = torch.from_numpy(self.waveforms)
        self.noises = self.noises.to(dtype=torch.float32, device=self.store_device)
        self.waveforms = self.waveforms.to(dtype=torch.float32, device=self.store_device)

    # HDF5 helpers for compatibility
    def save(self, h5py_file: h5py.File, group_name: str) -> None:
        if group_name in h5py_file.keys():
            raise IOError(f"Group '{group_name}' already exists.")
        g = h5py_file.create_group(group_name)
        g.create_dataset("waveforms", data=self.waveforms.cpu().numpy())
        g.create_dataset("noises", data=self.noises.cpu().numpy())

    def load(self, h5py_file: h5py.File, group_name: str) -> None:
        if group_name not in h5py_file.keys():
            raise IOError(f"Group '{group_name}' not found.")
        g = h5py_file[group_name]
        self.noises = g["noises"][()]
        self.waveforms = g["waveforms"][()]
        self._to_tensors()


class PretrainDataset(TorchDataset):
    """Contrastive dataset for InfoNCE with mixed pairs.

    Returns:
        (X1, X2) – two views, each [D, T]

    Positive (injection) pair:
        Xk = noise_k + SNR * waveform
    Negative (noise-only) pair:
        Xk = noise_k
    """
    def __init__(
        self,
        noises: torch.Tensor,
        waveforms: torch.Tensor,
        snr_range: Tuple[float, float] = (5.0, 15.0),
        noise_only_prob: float = 0.25,
        device: str = "cuda",
    ):
        super().__init__()
        assert 0.0 <= noise_only_prob <= 1.0, "`noise_only_prob` must be in [0,1]"
        if noises.ndim == 2:
            noises = noises.unsqueeze(1)
        if waveforms.ndim == 2:
            waveforms = waveforms.unsqueeze(1)
        assert noises.shape[1:] == waveforms.shape[1:], (
            f"shape mismatch: noises {noises.shape[1:]} vs waveforms {waveforms.shape[1:]}"
        )
        self.noises = noises.to(device)
        self.waveforms = waveforms.to(device)
        self.snr_low, self.snr_high = snr_range
        self.noise_only_prob = noise_only_prob
        self.device = device
        self.rng = np.random.default_rng()

    def __len__(self) -> int:
        return self.waveforms.size(0)

    def _sample_noise(self) -> torch.Tensor:
        idx = int(self.rng.integers(0, len(self.noises)))
        return self.noises[idx]  # [D, T]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rng.random() < self.noise_only_prob:
            X1, X2 = self._sample_noise(), self._sample_noise()
        else:
            w = self.waveforms[idx]  # [D, T]
            snr_val = self.rng.uniform(self.snr_low, self.snr_high)
            snr = torch.tensor(snr_val, device=self.device).view(1, 1)
            n1, n2 = self._sample_noise(), self._sample_noise()
            X1, X2 = n1 + snr * w, n2 + snr * w
        return X1, X2


# =============================================================================
# Loss
# =============================================================================

class RegBCELoss(nn.BCELoss):
    """BCELoss with small epsilon regularization on inputs to avoid log(0)."""
    def __init__(self, *args, epsilon: float = 1e-6, dim: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(dim, int)
        self.regularization_dim = dim
        self.regularization_A = epsilon
        self.regularization_B = 1.0 - epsilon * self.regularization_dim

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert inputs.shape[-1] == self.regularization_dim
        x = self.regularization_A + self.regularization_B * inputs
        return super().forward(x, target, *args, **kwargs)


# =============================================================================
# Contrastive pretrainer (InfoNCE)
# =============================================================================

class ContrastivePretrainer:
    """Pretrainer for Q-Adapter + Whisper encoder using InfoNCE."""
    def __init__(
        self,
        q_adapter: QTransformAdapter,
        whisper_encoder: nn.Module,
        n_detectors: int,
        device: str = "cuda",
        proj_dim: int = 256,
        lr: float = 1e-4,
        temperature: float = 0.1,
    ):
        self.device = device
        self.q_adapter = q_adapter.to(device)
        self.encoder = whisper_encoder.to(device)
        self.n_detectors = n_detectors

        d_model = whisper_encoder.config.d_model * n_detectors
        self.proj = nn.Sequential(
            nn.Linear(d_model, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        ).to(device)

        self.opt = torch.optim.AdamW(
            list(self.q_adapter.parameters())
            + list(self.encoder.parameters())
            + list(self.proj.parameters()),
            lr=lr,
        )
        self.temp = temperature

    @staticmethod
    def _l2norm(z: torch.Tensor) -> torch.Tensor:
        return F.normalize(z, dim=1)

    def _info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1, z2 = self._l2norm(z1), self._l2norm(z2)
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)                 # [2B, d]
        sim = (z @ z.T) / self.temp                    # [2B, 2B]
        mask = ~torch.eye(2 * B, device=z.device, dtype=torch.bool)
        exp_sim = torch.exp(sim) * mask
        pos = torch.exp((z1 * z2).sum(dim=1) / self.temp)  # [B]
        denom1 = exp_sim[:B].sum(dim=1)
        denom2 = exp_sim[B:].sum(dim=1)
        loss = -torch.log(pos / denom1) - torch.log(pos / denom2)
        return loss.mean()

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.q_adapter(x)               # [B, D, F*, T*]
        reps: List[torch.Tensor] = []
        for d in range(feats.size(1)):
            out = self.encoder(feats[:, d])
            reps.append(out.last_hidden_state[:, -1, :])
        return torch.cat(reps, dim=1)           # [B, d_model*D]

    def train(self, loader: DataLoader, steps: int = 25_000) -> None:
        """Run contrastive pretraining for a fixed number of updates."""
        self.q_adapter.train()
        self.encoder.train()
        pbar = tqdm(total=steps, desc="Contrastive Pre-train", ascii=True)

        data_iter = iter(loader)
        step = 0
        while step < steps:
            try:
                X1, X2 = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                X1, X2 = next(data_iter)

            X1 = X1.to(self.device)
            X2 = X2.to(self.device)

            z1 = self.proj(self._embed(X1))
            z2 = self.proj(self._embed(X2))
            loss = self._info_nce(z1, z2)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            pbar.update(1)
        pbar.close()


# =============================================================================
# Checkpointing & Trainer
# =============================================================================

@dataclass
class Checkpoint:
    epoch: int
    best_val_loss: float
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict] = None


class SupervisedTrainer:
    """Handles supervised fine-tuning lifecycle with checkpoints & early stop."""
    def __init__(
        self,
        model: nn.Module,
        device: str,
        lr: float = 5e-5,
        clip_norm: float = 100.0,
        loss_fn: Optional[nn.Module] = None,
        trainable_param_filter: Optional[Callable[[str, nn.Parameter], bool]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.clip_norm = clip_norm
        self.loss_fn = loss_fn or RegBCELoss(dim=2)

        # Collect parameters to update (adapter + LoRA + classifier)
        params: List[nn.Parameter] = []
        for name, p in self.model.adapter.named_parameters():
            params.append(p)
        for name, p in self.model.encoder.named_parameters():
            if ("lora" in name) and p.requires_grad:
                params.append(p)
        for name, p in self.model.classifier.named_parameters():
            params.append(p)

        self.optimizer = torch.optim.Adam(params, lr=lr)

    @staticmethod
    def _run_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: str,
        loss_fn: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        clip_norm: float = 0.0,
        desc: str = "",
        log: bool = False,
    ) -> float:
        running = 0.0
        batches = 0
        is_train = optimizer is not None
        model.train(is_train)

        iterator = loader
        if log:
            iterator = tqdm(loader, desc=desc, leave=False, ascii=True)

        for X, y in iterator:
            X = X.to(device)
            y = y.to(device)

            with torch.set_grad_enabled(is_train):
                out = model(X)
                loss = loss_fn(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
                optimizer.step()

            running += loss.detach().float().cpu().item()
            batches += 1

        return running / max(1, batches)

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        outdir: str,
        epochs: int = 100,
        resume: Optional[str] = None,   # "latest" | "best" | None
        force: bool = False,
        early_stop_patience: int = 10,
    ) -> None:
        os.makedirs(outdir, exist_ok=True)
        losses_path = os.path.join(outdir, "losses.txt")
        if os.path.isfile(losses_path) and not force:
            raise RuntimeError(f"Output file exists: {losses_path}")

        # Resume
        start_epoch, best_val = 1, float("inf")
        if resume:
            start_epoch, best_val = self._resume(outdir, resume)

        patience = 0
        with open(losses_path, "a", buffering=1) as f:
            for epoch in range(start_epoch, epochs + 1):
                train_loss = self._run_epoch(
                    self.model, train_loader, self.device, self.loss_fn,
                    optimizer=self.optimizer, clip_norm=self.clip_norm,
                    desc=f"Train {epoch}", log=True
                )
                val_loss = self._run_epoch(
                    self.model, valid_loader, self.device, self.loss_fn,
                    optimizer=None, clip_norm=0.0,
                    desc=f"Valid {epoch}", log=True
                )

                f.write(f"{epoch:04d}\t{train_loss:.6f}\t{val_loss:.6f}\n")

                # Save "last"
                last_ckpt = Checkpoint(
                    epoch=epoch,
                    best_val_loss=best_val,
                    model_state=self.model.state_dict(),
                    optimizer_state=self.optimizer.state_dict(),
                )
                torch.save(last_ckpt.__dict__, os.path.join(outdir, "last.pt"))

                # Save per-epoch state_dict for compatibility
                torch.save(self.model.state_dict(), os.path.join(outdir, f"state_dict_e_{epoch:04d}.pt"))

                # Check best
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(self.model.state_dict(), os.path.join(outdir, "best_state_dict.pt"))
                    patience = 0
                    logging.info(f"New best @ epoch {epoch:04d} — val_loss={val_loss:.6e}")
                    # also export components
                    save_model_components(
                        outdir,
                        adapter=self.model.adapter,
                        peft_model=self.model.encoder,
                        dense_layers=self.model.classifier,
                        adapter_path="best_adapter.pt",
                        lora_weights_path="best_lora_weights",
                        dense_layers_path="best_dense_layers.pth",
                    )
                else:
                    patience += 1
                    if patience >= early_stop_patience:
                        logging.info(f"Early stopping (patience {early_stop_patience}) at epoch {epoch:04d}.")
                        break

        logging.info(f"Training complete. Best validation loss: {best_val:.6f}")

    def _resume(self, outdir: str, which: str) -> Tuple[int, float]:
        """Resume from 'latest' or 'best' checkpoint."""
        device = self.device
        if which == "best":
            path = os.path.join(outdir, "best_state_dict.pt")
            if not os.path.isfile(path):
                logging.warning("No best_state_dict.pt found; starting fresh.")
                return 1, float("inf")
            state = torch.load(path, map_location=device)
            self.model.load_state_dict(state)
            logging.info("Resumed model from best_state_dict.pt (optimizer not restored).")
            return 1, float("inf")
        else:
            last_path = os.path.join(outdir, "last.pt")
            if not os.path.isfile(last_path):
                logging.warning("No last.pt found; starting fresh.")
                return 1, float("inf")
            payload = torch.load(last_path, map_location=device)
            self.model.load_state_dict(payload["model_state"])
            if payload.get("optimizer_state"):
                self.optimizer.load_state_dict(payload["optimizer_state"])
            logging.info("Resumed model+optimizer from last.pt.")
            return payload.get("epoch", 1) + 1, payload.get("best_val_loss", float("inf"))


# =============================================================================
# Builders (Whisper, LoRA/DoRA, Network)
# =============================================================================

def build_q_adapter(args) -> QTransformAdapter:
    return QTransformAdapter(
        kernel_length=args.kernel_length,
        sample_rate=args.sample_rate,
        q_range=args.q_range,
        spectrogram_shape=args.spectrogram_shape,
        target_shape=tuple(args.target_shape),
        n_detectors=args.n_detectors,
    )


def build_whisper_encoder(model_id: str = "openai/whisper-tiny", device: str = "cuda") -> nn.Module:
    """Load Whisper encoder and enable gradient checkpointing."""
    whisper = WhisperModel.from_pretrained(model_id)
    enc = whisper.encoder
    enc.gradient_checkpointing_enable()
    return enc.to(device)


def apply_lora(
    encoder: nn.Module,
    r: int = 8,
    alpha: int = 32,
    use_dora: bool = True,
    patterns: Optional[List[str]] = None,
) -> nn.Module:
    """Wrap the Whisper encoder with PEFT LoRA/DoRA on attention projections.

    Args:
        encoder: Whisper encoder module to wrap.
        r: LoRA rank.
        alpha: LoRA alpha.
        use_dora: Enable DoRA variant.
        patterns: Glob patterns of target module names within encoder.
    """
    module_names = [name for name, _ in encoder.named_modules()]
    if patterns is None:
        patterns = [
            "layers.*.self_attn.q_proj",
            "layers.*.self_attn.k_proj",
            "layers.*.self_attn.v_proj",
            "layers.*.self_attn.out_proj",
        ]
    matched = []
    for p in patterns:
        matched.extend(fnmatch.filter(module_names, p))
    logging.info(f"LoRA targeting modules: {matched}")

    cfg = LoraConfig(use_dora=use_dora, r=r, lora_alpha=alpha, target_modules=matched)
    peft_encoder = get_peft_model(encoder, cfg)

    # Freeze base params; train LoRA params only
    for name, param in peft_encoder.named_parameters():
        param.requires_grad = ("lora" in name)

    return peft_encoder


def build_network(
    encoder: nn.Module,
    adapter: QTransformAdapter,
    n_detectors: int,
    num_classes: int,
    device: str,
) -> GWWhisperClassifier:
    model = GWWhisperClassifier(
        whisper_encoder=encoder,
        q_adapter=adapter,
        n_detectors=n_detectors,
        num_classes=num_classes,
    ).to(device)
    total, trainable = count_parameters(model)
    logging.info(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    return model


def save_model_components(
    results_path: str,
    adapter: nn.Module,
    peft_model: nn.Module,
    dense_layers: nn.Module,
    adapter_path: str,
    lora_weights_path: str,
    dense_layers_path: str,
) -> None:
    """Save adapter state_dict, LoRA weights (PEFT), and classifier head."""
    os.makedirs(results_path, exist_ok=True)
    torch.save(adapter.state_dict(), os.path.join(results_path, adapter_path))
    peft_model.save_pretrained(os.path.join(results_path, lora_weights_path))
    torch.save(dense_layers.state_dict(), os.path.join(results_path, dense_layers_path))
    logging.info("Saved components: adapter, LoRA weights, dense layers.")


# =============================================================================
# Data loading helpers
# =============================================================================

def load_concat_datasets(
    dataset_dir: str,
    store_device: str,
    train_device: str,
    snr_range: Tuple[float, float],
) -> Tuple[ConcatDataset, ConcatDataset, List[BinaryGWDataset]]:
    """Load all HDF5 files in a directory, returning ConcatDatasets for train/valid.

    Also returns the list of training datasets for pretraining tensor concat.
    """
    dataset_files = [
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if os.path.isfile(os.path.join(dataset_dir, f))
    ]

    train_dses: List[BinaryGWDataset] = []
    valid_dses: List[BinaryGWDataset] = []

    for path in dataset_files:
        logging.info(f"Loading datasets from {path}")
        tr = BinaryGWDataset(store_device=store_device, train_device=train_device, snr_range=snr_range)
        va = BinaryGWDataset(store_device=store_device, train_device=train_device, snr_range=snr_range)
        with h5py.File(path, "r") as f:
            tr.load(f, "training")
            va.load(f, "validation")
        train_dses.append(tr)
        valid_dses.append(va)

    return ConcatDataset(train_dses), ConcatDataset(valid_dses), train_dses


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = ArgumentParser(description="GW-Whisper (Q-Scan) training script")
    # Logging & reproducibility
    parser.add_argument("--verbose", action="store_true", help="Print info logs.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logs.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic torch backend.")

    # Data
    parser.add_argument("-d", "--dataset-dir", type=str, required=True, help="Directory with HDF5 dataset files.")
    parser.add_argument("--n-detectors", type=int, default=2, help="Number of detectors (channels).")
    parser.add_argument("--sample-rate", type=int, default=2048, help="Input sample rate.")
    parser.add_argument("--spectrogram-shape", type=int, nargs=2, default=[128, 128], help="Q-Scan base (F, T) shape.")
    parser.add_argument("--target-shape", type=int, nargs=2, default=[80, 3000], help="Pooled (F*, T*) shape.")
    parser.add_argument("--q-range", type=int, nargs=2, default=[4, 128], help="Q-Transform range.")
    parser.add_argument("--kernel-length", type=float, default=1.0, help="Q-Transform kernel/window length (s).")

    # Training
    parser.add_argument("-o", "--output-training", type=str, required=True, help="Output directory (must exist).")
    parser.add_argument("--snr", type=float, nargs=2, default=(5.0, 15.0), help="Uniform SNR range for injections.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Fine-tune learning rate.")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tune epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--clip-norm", type=float, default=100.0, help="Gradient clipping norm.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin host memory in DataLoader.")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="Epochs of no improvement to stop.")
    parser.add_argument("--num-classes", type=int, default=2, help="Classifier output size (default 2).")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, choices=["latest", "best"],
                        help="Resume training: 'latest' or 'best'. If flag given without value => 'latest'.")

    # Devices
    parser.add_argument("--train-device", type=str, default="cuda", help="Device for training (e.g., 'cuda', 'cuda:1', 'cpu').")
    parser.add_argument("--store-device", type=str, default="cpu", help="Device to store datasets.")
    # Pretraining
    parser.add_argument("--pretrain-steps", type=int, default=60000, help="Contrastive pretraining steps (0 to skip).")
    parser.add_argument("--pretrain-lr", type=float, default=1e-4, help="Pretraining learning rate.")
    parser.add_argument("--pretrain-temp", type=float, default=0.1, help="InfoNCE temperature.")
    parser.add_argument("--noise-only-prob", type=float, default=0.25, help="Probability of a noise-only pair.")

    # PEFT (LoRA/DoRA)
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--use-dora", action="store_true", help="Enable DoRA variant for LoRA.")

    return parser.parse_args()


def main():
    args = parse_args()

    # Logging
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    else:
        level = logging.WARN
    logging.basicConfig(
        format="%(levelname)s | %(asctime)s: %(message)s",
        level=level,
        datefmt="%d-%m-%Y %H:%M:%S",
    )

    # Reproducibility
    set_seed(args.seed, deterministic=args.deterministic)

    # Data
    TrainDS, ValidDS, TrainDS_list = load_concat_datasets(
        args.dataset_dir, args.store_device, args.train_device, tuple(args.snr)
    )
    logging.info("Datasets loaded.")

    TrainDL = DataLoader(
        TrainDS, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    ValidDL = DataLoader(
        ValidDS, batch_size=max(32, args.batch_size), shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    # Models
    q_adapter = build_q_adapter(args)
    logging.info("Q-Transform adapter ready.")

    encoder = build_whisper_encoder(device=args.train_device)
    encoder = apply_lora(
        encoder, r=args.lora_rank, alpha=args.lora_alpha, use_dora=args.use_dora
    ).to(args.train_device)
    logging.info("Whisper encoder (PEFT) ready.")

    Network = build_network(
        encoder=encoder,
        adapter=q_adapter,
        n_detectors=args.n_detectors,
        num_classes=args.num_classes,
        device=args.train_device,
    )

    # =========================
    # Contrastive pretraining
    # =========================
    if args.pretrain_steps > 0:
        logging.info("Starting contrastive pretraining…")
        all_noises = torch.cat([ds.noises for ds in TrainDS_list], dim=0)
        all_waveforms = torch.cat([ds.waveforms for ds in TrainDS_list], dim=0)

        pre_ds = PretrainDataset(
            noises=all_noises,
            waveforms=all_waveforms,
            snr_range=tuple(args.snr),
            noise_only_prob=args.noise_only_prob,
            device=args.train_device,
        )
        pre_dl = DataLoader(
            pre_ds, batch_size=min(128, args.batch_size), shuffle=True,
            num_workers=args.num_workers, pin_memory=args.pin_memory
        )

        pretrainer = ContrastivePretrainer(
            q_adapter=Network.adapter,
            whisper_encoder=Network.encoder,
            n_detectors=Network.adapter.n_detectors,
            device=args.train_device,
            proj_dim=256,
            lr=args.pretrain_lr,
            temperature=args.pretrain_temp,
        )
        pretrainer.train(pre_dl, steps=args.pretrain_steps)

        # Save & reload best-pretrained weights
        pre_adapter_path = os.path.join(args.output_training, "q_adapter_pretrained.pt")
        pre_encoder_path = os.path.join(args.output_training, "encoder_pretrained.pt")
        torch.save(Network.adapter.state_dict(), pre_adapter_path)
        torch.save(Network.encoder.state_dict(), pre_encoder_path)
        logging.info("Saved pretraining weights.")
        Network.adapter.load_state_dict(torch.load(pre_adapter_path, map_location=args.train_device))
        Network.encoder.load_state_dict(torch.load(pre_encoder_path, map_location=args.train_device))
        logging.info("Reloaded pretraining weights.")

    # =========================
    # Supervised fine-tuning
    # =========================
    trainer = SupervisedTrainer(
        model=Network,
        device=args.train_device,
        lr=args.learning_rate,
        clip_norm=args.clip_norm,
        loss_fn=RegBCELoss(dim=args.num_classes),
    )
    trainer.fit(
        TrainDL, ValidDL, outdir=args.output_training,
        epochs=args.epochs, resume=args.resume, force=args.force,
        early_stop_patience=args.early_stop_patience,
    )


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    main()