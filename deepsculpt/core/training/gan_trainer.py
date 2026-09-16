"""
GAN trainer for DeepSculpt PyTorch implementation.

This module provides specialized training infrastructure for GAN models
with advanced training techniques, distributed training support, and
comprehensive monitoring capabilities.
"""

import os
import time
import logging
import copy
import json
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

import numpy as np
from datetime import datetime

from .base_trainer import BaseTrainer, TrainingConfig
from .training_metrics import TrainingMetrics


def r1_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    gamma: float = 10.0,
) -> torch.Tensor:
    """StyleGAN2 R1 gradient penalty (3D), a stable WGAN-GP alternative.

    Port of ``D_logistic_r1`` from sh4174/3DStyleGAN (``training/loss.py:52``).
    Penalizes the squared gradient norm of the *real* logits w.r.t. the real
    inputs only — no interpolation pass, so it is cheaper and more stable than
    WGAN-GP at growing 3D resolutions. Returns ``0.5 * gamma * E[||∇D(x)||²]``.

    Standalone (module-level) twin of ``GANTrainer._compute_r1_penalty`` so it
    can be unit-tested and reused outside the trainer. Config-toggled in the
    trainer via ``TrainingConfig.gan_loss_type`` ("softplus" uses R1,
    "wgan-gp" uses the interpolated gradient penalty) and ``r1_gamma``.

    NEEDS-GPU-VALIDATION: mode-collapse suppression vs WGAN-GP is unverified
    on real training runs; shape/gradient behaviour is unit-tested.
    """
    real = real_samples.detach().requires_grad_(True)
    logits = discriminator(real)
    (grads,) = torch.autograd.grad(
        outputs=logits.sum(),
        inputs=real,
        create_graph=True,
    )
    r1 = grads.square().reshape(grads.shape[0], -1).sum(dim=1).mean()
    return 0.5 * gamma * r1


class GANTrainer(BaseTrainer):
    """
    Specialized trainer for GAN models with advanced training techniques.
    
    Features:
    - Adversarial training with generator and discriminator
    - Progressive growing support
    - Curriculum learning
    - Gradient penalty and spectral normalization
    - Advanced loss functions and training strategies
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        gen_optimizer: torch.optim.Optimizer,
        disc_optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        gen_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        disc_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        noise_dim: int = 100,
        loss_type: str = "bce",  # "bce", "wgan", "lsgan", "hinge"
        use_gradient_penalty: bool = False,
        gradient_penalty_weight: float = 10.0
    ):
        """
        Initialize GAN trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            gen_optimizer: Generator optimizer
            disc_optimizer: Discriminator optimizer
            config: Training configuration
            gen_scheduler: Generator learning rate scheduler
            disc_scheduler: Discriminator learning rate scheduler
            device: Device for training
            noise_dim: Dimension of noise vector
            loss_type: Type of adversarial loss
            use_gradient_penalty: Whether to use gradient penalty (WGAN-GP)
            gradient_penalty_weight: Weight for gradient penalty
        """
        # Initialize base trainer with generator as main model
        super().__init__(generator, gen_optimizer, config, gen_scheduler, device)
        
        self.generator = generator
        self.discriminator = discriminator
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.noise_dim = noise_dim
        self.loss_type = loss_type
        self.use_gradient_penalty = use_gradient_penalty
        self.gradient_penalty_weight = gradient_penalty_weight

        # Color mode is derived from the discriminator's input width:
        # 1 channel = mono occupancy (legacy path, bit-identical), >1 = the
        # 13-channel semantic-class field (channel 0 = empty, 1-12 = element
        # classes). Everything color-specific below gates on self.is_color.
        self.num_classes = int(getattr(discriminator, "input_channels", 1))
        # RGBA colour mode: a palette_cfg (set post-construction by the CLI)
        # switches reals to build_rgba [alpha,R,G,B] fields; without it, >1
        # disc channels means the 13-class semantic path.
        self.palette_cfg = None
        self.is_color = self.num_classes > 1
        
        # Additional scaler for discriminator if using mixed precision
        try:
            # New PyTorch 2.9+ API
            from torch.amp import GradScaler
            # AMP GradScaler is CUDA-only — guard so MPS/CPU (e.g. M5 Max) don't crash here
            self.disc_scaler = GradScaler('cuda') if config.mixed_precision and str(device).startswith('cuda') else None
        except ImportError:
            # Fallback for older PyTorch versions
            from torch.cuda.amp import GradScaler
            self.disc_scaler = GradScaler() if config.mixed_precision else None
        
        # GAN-specific metrics
        self.metrics.update({
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'gradient_penalty': [],
            'disc_real_loss': [],
            'disc_fake_loss': [],
            'r1_penalty': [],
            'augment_p': [],
            'fake_occupancy': [],
            'real_occupancy': [],
            'occupancy_gap': [],
            'occupancy_penalty': [],
            'gen_grad_norm': [],
            'disc_grad_norm': [],
        })
        
        # Progressive growing parameters
        self.progressive_level = 0
        self.progressive_alpha = 1.0
        
        # Curriculum learning parameters
        self.curriculum_stage = 0
        self.curriculum_threshold = 0.8
        
        # Training balance parameters
        self.gen_train_freq = 1  # Train generator every N discriminator updates
        self.disc_train_freq = 1  # Train discriminator every N generator updates
        
        # Fixed noise for consistent evaluation — CPU-seeded so the same
        # vectors survive process restarts (chained cloud slices sample
        # identical latents, keeping snapshots comparable across slices).
        fixed_gen = torch.Generator().manual_seed(1234)
        self.fixed_noise = torch.randn(16, noise_dim, generator=fixed_gen).to(device)
        
        # Setup distributed training for discriminator
        if config.distributed:
            self.discriminator = DDP(self.discriminator, device_ids=[config.rank])
        
        # Initialize metrics tracker
        self.metrics_tracker = TrainingMetrics()
        self.ema_generator = self._create_ema_generator() if config.use_ema else None
        self.ema_discriminator = self._create_ema_discriminator() if getattr(config, 'use_disc_ema', True) else None
        self.augment_p = float(getattr(config, "augment_p", 0.0))
        self.collapse_events = 0
        self.nan_events = 0
        self.last_epoch_metrics: Dict[str, float] = {}
        self.last_run_summary: Dict[str, Any] = {}

        # TTUR: when gen and disc start with the same LR, reduce disc to 25%
        # to prevent the discriminator from winning too easily on sparse data.
        gen_lr = self.gen_optimizer.param_groups[0]["lr"]
        disc_lr = self.disc_optimizer.param_groups[0]["lr"]
        if abs(gen_lr - disc_lr) < 1e-8:
            ttur_ratio = float(getattr(config, "ttur_ratio", 0.25))
            for pg in self.disc_optimizer.param_groups:
                pg["lr"] = gen_lr * ttur_ratio
            self.logger.info("TTUR: disc LR set to %.6f (gen LR %.6f × %.2f)", gen_lr * ttur_ratio, gen_lr, ttur_ratio)

        # Adaptive balance controller state
        self._occupancy_health = 1.0  # EMA: 1.0 = healthy, 0.0 = collapsed
        self._ema_disc_loss = 0.5  # EMA of disc loss to detect dominance
        self._adaptive_occ_weight = float(getattr(config, "occupancy_loss_weight", 5.0))
        self._base_occ_weight = self._adaptive_occ_weight
        self._adaptive_gen_steps = 1  # how many gen steps per disc step
        self._disc_lr_scale = 1.0  # multiplicative factor on disc LR
        self._base_disc_lr = self.disc_optimizer.param_groups[0]["lr"]

    def _create_ema_generator(self) -> nn.Module:
        """Create EMA version of the generator for stable sampling."""
        ema_generator = copy.deepcopy(self.generator)
        ema_generator.eval()
        return ema_generator.to(self.device)

    def _update_ema_generator(self):
        """Update EMA generator weights."""
        if self.ema_generator is None:
            return

        ema_decay = getattr(self.config, "ema_decay", 0.999)
        with torch.no_grad():
            for ema_param, param in zip(self.ema_generator.parameters(), self.generator.parameters()):
                ema_param.data.mul_(ema_decay).add_(param.data, alpha=1 - ema_decay)

    def _create_ema_discriminator(self) -> nn.Module:
        """Create EMA discriminator for stable adversarial signal to the generator."""
        ema_disc = copy.deepcopy(self.discriminator)
        ema_disc.eval()
        return ema_disc.to(self.device)

    def _update_ema_discriminator(self):
        """Update EMA discriminator weights."""
        if self.ema_discriminator is None:
            return
        ema_decay = getattr(self.config, "disc_ema_decay", 0.999)
        with torch.no_grad():
            for ema_p, p in zip(self.ema_discriminator.parameters(), self.discriminator.parameters()):
                ema_p.data.mul_(ema_decay).add_(p.data, alpha=1 - ema_decay)

    def _compute_gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        """WGAN-GP gradient penalty on interpolated samples."""
        alpha = torch.rand(real.size(0), 1, 1, 1, 1, device=real.device)
        interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        d_interp = self.discriminator(interpolated)
        grads = torch.autograd.grad(
            outputs=d_interp.sum(), inputs=interpolated, create_graph=True,
        )[0]
        return ((grads.reshape(grads.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

    def _generator_for_sampling(self) -> nn.Module:
        """Return the generator to use for checkpoints and sample exports."""
        if getattr(self.config, "sample_from_ema", True) and self.ema_generator is not None:
            return self.ema_generator
        return self.generator

    def _apply_ada_lite(self, batch: torch.Tensor, update_probability: bool = False) -> torch.Tensor:
        """Apply lightweight discriminator-side augmentations."""
        if getattr(self.config, "augment", "none") != "ada-lite" or self.augment_p <= 0:
            return batch

        augmented = batch
        if torch.rand(1, device=batch.device).item() < self.augment_p:
            if torch.rand(1, device=batch.device).item() < 0.5:
                augmented = torch.flip(augmented, dims=[2])
            if torch.rand(1, device=batch.device).item() < 0.5:
                augmented = torch.flip(augmented, dims=[3])
            if torch.rand(1, device=batch.device).item() < 0.5:
                augmented = torch.flip(augmented, dims=[4])
            if torch.rand(1, device=batch.device).item() < 0.5:
                k = int(torch.randint(1, 4, (1,), device=batch.device).item())
                augmented = torch.rot90(augmented, k=k, dims=(3, 4))
            if torch.rand(1, device=batch.device).item() < 0.3:
                augmented = augmented + torch.randn_like(augmented) * 0.01

        if update_probability:
            self.augment_p = float(np.clip(self.augment_p, 0.0, 0.8))

        return augmented

    def _update_augment_probability(self, real_logits: torch.Tensor):
        """Heuristic ADA-lite probability controller."""
        if getattr(self.config, "augment", "none") != "ada-lite":
            return

        target = getattr(self.config, "augment_target", 0.6)
        adjust = np.sign((real_logits > 0).float().mean().item() - target) * (real_logits.shape[0] / 1000.0)
        self.augment_p = float(np.clip(self.augment_p + adjust, 0.0, 0.8))

    def _compute_r1_penalty(self, real_samples: torch.Tensor) -> torch.Tensor:
        """Compute StyleGAN-style R1 penalty on real samples."""
        real_samples = real_samples.detach().requires_grad_(True)
        real_logits = self.discriminator(self._apply_ada_lite(real_samples))
        gradients = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_samples,
            create_graph=True,
        )[0]
        return gradients.square().reshape(gradients.shape[0], -1).sum(dim=1).mean()

    def _gradient_norm(self, parameters) -> float:
        total_norm_sq = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            total_norm_sq += float(param.grad.detach().norm(2).item() ** 2)
        return total_norm_sq ** 0.5

    def _compute_occupancy(self, batch: torch.Tensor, differentiable: bool = False) -> torch.Tensor:
        """Compute scalar occupancy ratio for a voxel batch.

        Mono: with straight-through binarization the generator output is
        already {0,1}, so ``batch.mean()`` equals the hard-threshold occupancy
        AND is differentiable (STE passes gradients through the binarization).

        Color (13ch categorical, channel 0 = empty): differentiable occupancy
        is the probability mass outside the empty channel; the hard variant is
        the argmax-occupied voxel fraction. For real (smoothed) one-hot
        batches the hard variant equals the exact (colors > 0) fraction.
        """
        if self.palette_cfg is not None:
            # RGBA: channel 0 (alpha) IS the occupancy (sigmoid in [0,1]).
            if differentiable:
                return batch[:, 0].mean()
            return (batch[:, 0].detach() > 0.5).float().mean()
        if self.is_color:
            if differentiable:
                return (1.0 - batch[:, 0]).mean()
            return (batch.detach().argmax(dim=1) > 0).float().mean()
        if differentiable:
            return batch.mean()
        return batch.detach().mean()

    def _occupancy_target(self, real_occupancy: torch.Tensor) -> torch.Tensor:
        """Resolve the occupancy target for generator regularization."""
        if getattr(self.config, "occupancy_target_mode", "batch_real") == "dataset_mean":
            dataset_mean = getattr(self.config, "dataset_occupancy_mean", None)
            if dataset_mean is not None:
                return real_occupancy.new_tensor(float(dataset_mean))
        return real_occupancy.detach()

    def _occupancy_penalty(self, fake_batch: torch.Tensor, real_occupancy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Penalize collapse toward all-empty or occupancy-mismatched outputs.

        Uses differentiable soft-threshold occupancy so gradients flow back
        to the generator.  Returns the hard-threshold occupancy for logging.
        """
        soft_occ = self._compute_occupancy(fake_batch, differentiable=True)
        hard_occ = self._compute_occupancy(fake_batch, differentiable=False)
        target = self._occupancy_target(real_occupancy)
        occupancy_gap = soft_occ - target
        floor = float(getattr(self.config, "occupancy_floor", 0.01))
        floor_deficit = F.relu(soft_occ.new_tensor(floor) - soft_occ)
        penalty = occupancy_gap.abs() + 10.0 * floor_deficit
        return penalty, hard_occ

    # ------------------------------------------------------------------
    # Adaptive balance controller
    # ------------------------------------------------------------------

    def _update_adaptive_balance(self, metrics: Dict[str, float]):
        """Continuous feedback loop that adjusts training balance based on
        occupancy health.  Runs every step — no fixed patience or hard resets.

        Three control surfaces:
        1. ``_adaptive_occ_weight`` — occupancy penalty multiplier on gen loss
        2. ``_adaptive_gen_steps`` — extra generator steps per batch (1–4)
        3. ``_disc_lr_scale`` — multiplicative factor applied to disc LR

        The controller uses an EMA ``_occupancy_health`` score (1 = healthy,
        0 = fully collapsed) to smoothly ramp these up during collapse and
        back off during recovery.
        """
        occupancy_floor = float(getattr(self.config, "occupancy_floor", 0.01))
        fake_occ = metrics.get("fake_occupancy", 0.5)
        real_occ = metrics.get("real_occupancy", 0.05)

        # --- compute instantaneous health (0–1) -------------------------
        if real_occ > 1e-6:
            ratio = min(fake_occ / real_occ, 2.0)  # cap at 2× to ignore overshoot
        else:
            ratio = 1.0 if fake_occ > occupancy_floor else 0.0
        # Below floor is always unhealthy regardless of ratio
        if fake_occ < occupancy_floor:
            ratio = 0.0
        instant_health = min(ratio, 1.0)

        # --- EMA update --------------------------------------------------
        ema_beta = 0.95
        self._occupancy_health = ema_beta * self._occupancy_health + (1 - ema_beta) * instant_health

        # --- Disc dominance detection ------------------------------------
        # When disc loss stays near zero, the disc wins trivially even if
        # occupancy looks fine. Penalize health to trigger throttling.
        disc_loss = metrics.get("disc_loss", 0.5)
        self._ema_disc_loss = ema_beta * self._ema_disc_loss + (1 - ema_beta) * disc_loss
        if self._ema_disc_loss < 0.05:
            disc_penalty = 1.0 - (self._ema_disc_loss / 0.05)  # 0→1 as loss→0
            self._occupancy_health *= (1.0 - 0.2 * disc_penalty)
            # Disc dominance alone shouldn't trigger full collapse response
            self._occupancy_health = max(self._occupancy_health, 0.3)

        h = self._occupancy_health  # shorthand

        # --- 1. Occupancy penalty weight: boost when unhealthy -----------
        # Ranges from 1× (healthy) to 5× (collapsed) of base weight
        occ_boost = 1.0 + 4.0 * (1.0 - h)
        self._adaptive_occ_weight = self._base_occ_weight * occ_boost

        # --- 2. Extra generator steps: 1 when healthy, up to 4 collapsed -
        self._adaptive_gen_steps = max(1, min(4, int(1 + 3 * (1.0 - h))))

        # --- 3. Disc LR scale: 1× healthy → 0.1× collapsed/dominated ---
        self._disc_lr_scale = 0.1 + 0.9 * h
        for pg in self.disc_optimizer.param_groups:
            pg["lr"] = self._base_disc_lr * self._disc_lr_scale

    def _check_step_health(self, metrics: Dict[str, float]):
        """Emit warnings on obvious collapse or non-finite states and run
        the adaptive balance controller."""
        bad = [k for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isfinite(v)]
        if bad:
            self.nan_events += 1
            self.logger.warning("Non-finite GAN metrics detected; run may be unstable: %s", bad)
            if getattr(self.config, "nan_guard", True):
                raise RuntimeError(f"Non-finite GAN metrics: {bad}")

        occupancy_floor = float(getattr(self.config, "occupancy_floor", 0.01))
        # Collapse is defined by occupancy only — low disc_loss is normal
        # with TTUR and doesn't indicate collapse by itself.
        is_collapsed = (
            metrics.get("fake_occupancy", 0.5) < occupancy_floor
            or metrics.get("fake_occupancy", 0.5) > 0.9999
        )

        if is_collapsed:
            self.collapse_events += 1
            if self.collapse_events >= 3:
                status = "empty"
                if metrics.get("fake_occupancy", 0.5) > 0.9999:
                    status = "full"
                self.logger.warning(
                    "Potential %s GAN collapse detected: disc_loss=%.6f fake_occupancy=%.6f real_occupancy=%.6f gap=%.6f",
                    status,
                    metrics.get("disc_loss", 0.0),
                    metrics.get("fake_occupancy", 0.0),
                    metrics.get("real_occupancy", 0.0),
                    metrics.get("occupancy_gap", 0.0),
                )
                if metrics.get("augment_p", 0.0) >= 0.8 and metrics.get("fake_occupancy", 1.0) < occupancy_floor:
                    self.logger.warning(
                        "ADA-lite is saturated while occupancy remains below floor; discriminator pressure is likely too high for this dataset."
                    )
        else:
            self.collapse_events = 0

        # Run the adaptive controller every step
        self._update_adaptive_balance(metrics)
    
    def adversarial_loss(self, output: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Calculate adversarial loss based on loss type.
        
        Args:
            output: Discriminator output
            target_is_real: Whether target should be real or fake
            
        Returns:
            Computed loss
        """
        if self.loss_type == "bce":
            if target_is_real:
                target = torch.ones_like(output)
            else:
                target = torch.zeros_like(output)
            return F.binary_cross_entropy_with_logits(output, target)
        
        elif self.loss_type == "wgan":
            if target_is_real:
                return -output.mean()
            else:
                return output.mean()
        
        elif self.loss_type == "lsgan":
            if target_is_real:
                target = torch.ones_like(output)
            else:
                target = torch.zeros_like(output)
            return F.mse_loss(output, target)
        
        elif self.loss_type == "hinge":
            if target_is_real:
                return F.relu(1.0 - output).mean()
            else:
                return F.relu(1.0 + output).mean()
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def gradient_penalty(self, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        """
        Calculate gradient penalty for WGAN-GP.
        
        Args:
            real_samples: Real data samples
            fake_samples: Generated fake samples
            
        Returns:
            Gradient penalty value
        """
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1, 1, device=self.device)
        
        # Interpolate between real and fake samples
        interpolates = alpha * real_samples + (1 - alpha) * fake_samples
        interpolates.requires_grad_(True)
        
        # Get discriminator output for interpolates
        disc_interpolates = self.discriminator(interpolates)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Calculate penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(self, real_data: torch.Tensor) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            real_data: Batch of real data
            
        Returns:
            Dictionary of step metrics
        """
        real_data = real_data.float()
        batch_size = real_data.size(0)
        autocast_enabled = self.config.mixed_precision and self.device != "cpu"
        # bf16, not fp16: the WGAN critic's logits are unbounded and overflow
        # fp16's 65504 max within the first steps (first observed on L4).
        autocast_dtype = torch.bfloat16 if autocast_enabled else None
        r1_penalty_value = 0.0
        real_occupancy = self._compute_occupancy(real_data)

        # Instance noise: decaying Gaussian added to disc inputs so it can't
        # instantly memorize "sparse = real" in the first few batches.
        decay_steps = float(getattr(self.config, "instance_noise_decay_steps", 2000))
        noise_std = max(0.0, 0.1 * (1.0 - self.global_step / decay_steps))

        # --- Discriminator step (skipped when health is critically low) ---
        skip_disc = self._occupancy_health < 0.1
        disc_grad_norm = 0.0

        use_wgan = getattr(self.config, "gan_loss_type", "softplus") == "wgan-gp"
        fm_weight = float(getattr(self.config, "feature_matching_weight", 1.0))

        if not skip_disc:
            self.disc_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            with torch.autocast(device_type=self.device, dtype=autocast_dtype, enabled=autocast_enabled):
                fake_for_disc = self.generator(noise).detach()
                # Add instance noise to disc inputs (not gen path)
                real_noisy = real_data + noise_std * torch.randn_like(real_data) if noise_std > 0 else real_data
                fake_noisy = fake_for_disc + noise_std * torch.randn_like(fake_for_disc) if noise_std > 0 else fake_for_disc
                real_logits = self.discriminator(self._apply_ada_lite(real_noisy))
                fake_logits = self.discriminator(self._apply_ada_lite(fake_noisy))

                if use_wgan:
                    # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
                    disc_real_loss = -real_logits.mean()
                    disc_fake_loss = fake_logits.mean()
                    disc_loss = disc_real_loss + disc_fake_loss
                else:
                    disc_real_loss = F.softplus(-real_logits).mean()
                    disc_fake_loss = F.softplus(fake_logits).mean()
                    disc_loss = disc_real_loss + disc_fake_loss

            disc_loss.backward()

            # WGAN-GP: gradient penalty on interpolated samples
            if use_wgan:
                self.disc_optimizer.zero_grad(set_to_none=True)
                gp = self._compute_gradient_penalty(real_data, fake_for_disc)
                r1_penalty_value = float(gp.item())
                gp_loss = 10.0 * gp
                gp_loss.backward()
            elif self.global_step % max(1, getattr(self.config, "r1_interval", 16)) == 0:
                # R1 penalty for softplus mode
                self.disc_optimizer.zero_grad(set_to_none=True)
                r1_penalty = self._compute_r1_penalty(real_data)
                r1_penalty_value = float(r1_penalty.item())
                r1_loss = 0.5 * getattr(self.config, "r1_gamma", 10.0) * r1_penalty * max(1, getattr(self.config, "r1_interval", 16))
                r1_loss.backward()

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.gradient_clip)
            disc_grad_norm = self._gradient_norm(self.discriminator.parameters())
            self.disc_optimizer.step()
        else:
            # Still need logits for metrics even when skipping disc update
            with torch.no_grad():
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                fake_for_disc = self.generator(noise)
                real_logits = self.discriminator(real_data)
                fake_logits = self.discriminator(fake_for_disc)
                disc_real_loss = F.softplus(-real_logits).mean()
                disc_fake_loss = F.softplus(fake_logits).mean()
                disc_loss = disc_real_loss + disc_fake_loss

        # --- Generator steps (adaptive count: 1 when healthy, up to 4) ---
        occ_weight = self._adaptive_occ_weight
        gen_steps = self._adaptive_gen_steps
        adv_gate = max(0.1, self._occupancy_health)

        # Generator trains against the LIVE discriminator. The EMA copy must
        # not be used here: EMA'd weights don't get functioning spectral-norm
        # power iteration (stale u/v buffers → unbounded logits), which blew
        # gen_loss to 1e5-1e8 across runs gan-cr-002/003/004 while the raw
        # disc stayed perfectly bounded.
        disc_for_gen = self.discriminator

        # Cache real features for feature matching (computed once per step)
        fm_loss_value = 0.0
        if fm_weight > 0 and hasattr(disc_for_gen, 'forward') and 'return_features' in disc_for_gen.forward.__code__.co_varnames:
            with torch.no_grad():
                _, real_features = disc_for_gen(real_data, return_features=True)
                real_feat_means = [rf.mean(0).detach() for rf in real_features]
        else:
            real_feat_means = None

        for _ in range(gen_steps):
            self.gen_optimizer.zero_grad(set_to_none=True)
            noise = torch.randn(batch_size, self.noise_dim, device=self.device)
            with torch.autocast(device_type=self.device, dtype=autocast_dtype, enabled=autocast_enabled):
                fake_for_gen = self.generator(noise)

                if use_wgan:
                    gen_logits = disc_for_gen(fake_for_gen)
                    adv_loss = -gen_logits.mean() * adv_gate
                else:
                    gen_logits = disc_for_gen(self._apply_ada_lite(fake_for_gen))
                    adv_loss = F.softplus(-gen_logits).mean() * adv_gate

                occupancy_penalty, fake_occupancy = self._occupancy_penalty(fake_for_gen, real_occupancy)
                occupancy_loss = occupancy_penalty * occ_weight

                # Feature matching: match normalized intermediate disc feature statistics
                if real_feat_means is not None and fm_weight > 0:
                    _, fake_features = disc_for_gen(fake_for_gen, return_features=True)
                    # Normalize each layer's contribution by its dimensionality
                    fm_loss = sum(F.mse_loss(ff.mean(0), rm) / (rm.numel() + 1)
                                 for ff, rm in zip(fake_features, real_feat_means))
                    fm_loss_value = float(fm_loss.item())
                else:
                    fm_loss = torch.tensor(0.0, device=self.device)

                gen_loss = adv_loss + occupancy_loss + fm_weight * fm_loss

            gen_loss.backward()
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config.gradient_clip)
            gen_grad_norm = self._gradient_norm(self.generator.parameters())
            self.gen_optimizer.step()

        self._update_ema_generator()
        self._update_ema_discriminator()
        if not skip_disc:
            self._update_augment_probability(real_logits.detach())

        step_metrics = {
            "loss": float(gen_loss.item()),
            "gen_loss": float(gen_loss.item()),
            "disc_loss": float(disc_loss.item()),
            "disc_real_loss": float(disc_real_loss.item()),
            "disc_fake_loss": float(disc_fake_loss.item()),
            "disc_real_acc": float((real_logits.detach() > 0).float().mean().item()),
            "disc_fake_acc": float((fake_logits.detach() < 0).float().mean().item()),
            "feature_matching_loss": fm_loss_value,
            "gradient_penalty": r1_penalty_value if use_wgan else 0.0,
            "r1_penalty": r1_penalty_value,
            "augment_p": float(self.augment_p),
            "real_occupancy": float(real_occupancy.item()),
            "fake_occupancy": float(fake_occupancy.item()),
            "occupancy_gap": float((fake_occupancy - self._occupancy_target(real_occupancy)).item()),
            "occupancy_penalty": float(occupancy_penalty.item()),
            "gen_grad_norm": gen_grad_norm,
            "disc_grad_norm": disc_grad_norm,
            "occ_health": self._occupancy_health,
            "adaptive_occ_weight": occ_weight,
            "adaptive_gen_steps": float(gen_steps),
            "disc_lr_scale": self._disc_lr_scale,
            "disc_skipped": float(skip_disc),
        }

        self._check_step_health(step_metrics)
        step_metrics["collapse_events"] = float(self.collapse_events)
        self.metrics_tracker.update_step_metrics(step_metrics)
        return step_metrics
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of epoch metrics
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': [],
            'gradient_penalty': [],
            'disc_real_loss': [],
            'disc_fake_loss': [],
            'r1_penalty': [],
            'augment_p': [],
            'fake_occupancy': [],
            'real_occupancy': [],
            'occupancy_gap': [],
            'occupancy_penalty': [],
            'gen_grad_norm': [],
            'disc_grad_norm': [],
        }
        
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                real_data = batch[0].to(self.device)
            elif isinstance(batch, dict):
                # Handle dictionary batch format from StreamingDataset
                structure = batch["structure"].to(self.device)

                if self.palette_cfg is not None:
                    # RGBA colour mode: reals are the procedural palette field
                    # [alpha, R, G, B] — same targets the colour diffusion uses.
                    from deepsculpt.core.data.transforms.palette import build_rgba
                    real_data = build_rgba(
                        structure, batch["colors"].to(self.device),
                        batch["index"].to(self.device), self.palette_cfg,
                    )
                elif self.is_color:
                    # Semantic-class mode: reals come from the colors tensor
                    # (class indices 0-12; colors > 0 == structure > 0), one-hot
                    # encoded over the channel dim: [B, D, H, W] -> [B, 13, D, H, W].
                    colors = batch["colors"].to(self.device)
                    if colors.dim() == 5:  # legacy collector layout [B, 1, D, H, W]
                        colors = colors.squeeze(1)
                    # .contiguous(): the permute leaves a channels-last layout
                    # that propagates through the disc convs and crashes the
                    # flatten .view (mono never hits this — no permute).
                    real_data = F.one_hot(
                        colors.long().clamp(0, self.num_classes - 1),
                        num_classes=self.num_classes,
                    ).permute(0, 4, 1, 2, 3).float().contiguous()
                    # Label smoothing (0.95 one-hot + 0.05 uniform): the disc
                    # never sees exact simplex vertices, so it can't key on
                    # the one-hot spikes to separate real from softmax fakes.
                    real_data = real_data * 0.95 + 0.05 / self.num_classes
                else:
                    # Monochrome: [batch, depth, height, width] -> [batch, 1, D, H, W]
                    if structure.dim() == 4:
                        real_data = structure.unsqueeze(1)
                    else:
                        real_data = structure

                    # Convert to float if needed (models expect float tensors)
                    if real_data.dtype != torch.float32:
                        real_data = real_data.float()
            else:
                real_data = batch.to(self.device)
            
            # Progressive growing curriculum
            if self.config.progressive_growing:
                self._update_progressive_growing(batch_idx)
            
            # Curriculum learning
            if self.config.curriculum_learning:
                self._update_curriculum_learning(epoch_metrics)
            
            # Training step
            step_metrics = self.train_step(real_data)
            
            # Accumulate metrics
            for key, value in step_metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key].append(value)
            
            # Log step metrics (every N batches + always the last batch)
            is_last_batch = (batch_idx == len(dataloader) - 1)
            if batch_idx % self.config.log_freq == 0 or is_last_batch:
                self.log_metrics(step_metrics, self.global_step, "train_step")
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"G: {step_metrics.get('gen_loss', 0):.4f}, "
                    f"D: {step_metrics.get('disc_loss', 0):.4f}, "
                    f"FM: {step_metrics.get('feature_matching_loss', 0):.4f}, "
                    f"FakeOcc: {step_metrics.get('fake_occupancy', 0):.4f}, "
                    f"Health: {step_metrics.get('occ_health', 1):.2f}"
                )

            self.global_step += 1

        # Calculate epoch averages
        avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items() if values}
        
        # Update learning rate schedulers
        if self.gen_scheduler:
            self.gen_scheduler.step()
        if self.disc_scheduler:
            self.disc_scheduler.step()
        
        # Update metrics tracker
        self.metrics_tracker.update_epoch_metrics(avg_metrics)
        self.last_epoch_metrics = avg_metrics

        return avg_metrics

    def _snapshot_sample_stats(self, samples: torch.Tensor) -> Dict[str, float]:
        s = samples.detach()
        if self.palette_cfg is not None and s.dim() == 5 and s.shape[1] == 4:
            # RGBA occupancy is the alpha channel thresholded at 0.5 (already
            # a {0,1} mask — don't .abs() a bool below).
            occ_mask = (s[:, 0:1] > 0.5)
        else:
            occ_mask = (s.abs() > 0)
        occupancy = occ_mask.float().reshape(s.shape[0], -1).mean(dim=1)
        nonzero = occ_mask.reshape(s.shape[0], -1).sum(dim=1).float()
        return {
            "mean_occupancy": float(occupancy.mean().item()),
            "min_occupancy": float(occupancy.min().item()),
            "max_occupancy": float(occupancy.max().item()),
            "mean_nonzero_voxels": float(nonzero.mean().item()),
            "min_nonzero_voxels": float(nonzero.min().item()),
            "max_nonzero_voxels": float(nonzero.max().item()),
        }

    def _save_epoch_snapshot(self, epoch: int, train_metrics: Dict[str, float]) -> None:
        os.makedirs(self.config.snapshot_dir, exist_ok=True)
        samples = self.generate_samples(num_samples=4, use_fixed_noise=True).detach().cpu()
        if self.palette_cfg is not None:
            # RGBA: keep the [alpha,R,G,B] field (fp16) — render_walk colours it.
            samples = samples.half()
        elif self.is_color:
            # Save class-id volumes (int8, (N, D, H, W)) instead of 13ch floats
            # — ~50x smaller and directly renderable with the class palette.
            samples = samples.argmax(dim=1).to(torch.int8)
        snapshot_stats = self._snapshot_sample_stats(samples)
        snapshot_stem = Path(self.config.snapshot_dir) / f"epoch_{epoch + 1:03d}"
        torch.save(samples, snapshot_stem.with_suffix(".pt"))
        walk_stats = self._save_epoch_walk(epoch, snapshot_stem)
        with open(snapshot_stem.with_suffix(".json"), "w") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "train_metrics": train_metrics,
                    "sample_stats": snapshot_stats,
                    "walk_stats": walk_stats,
                },
                f,
                indent=2,
            )

    def _save_epoch_walk(self, epoch: int, snapshot_stem: Path) -> Dict[str, float]:
        """Dump a fixed-anchor slerp walk through the RAW generator (the EMA
        copy looks diffuse mid-training). Same path every epoch, so latent
        smoothness is judgeable across the whole run."""
        from deepsculpt.core.latent.ops import slerp

        zs = slerp(self.fixed_noise[0], self.fixed_noise[1], 16).to(self.device)
        was_training = self.generator.training
        self.generator.eval()
        with torch.no_grad():
            walk = torch.cat(
                [self.generator(zs[i:i + 4]) for i in range(0, len(zs), 4)]
            ).detach().cpu()
        if was_training:
            self.generator.train()
        if self.is_color:
            # Class-id volumes (int8) — see _save_epoch_snapshot.
            walk = walk.argmax(dim=1).to(torch.int8)
            torch.save(walk, snapshot_stem.with_name(f"walk_epoch_{epoch + 1:03d}.pt"))
        else:
            torch.save(walk.half(), snapshot_stem.with_name(f"walk_epoch_{epoch + 1:03d}.pt"))
        return self._snapshot_sample_stats(walk)

    def _after_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
        is_best: bool,
    ) -> None:
        if (epoch + 1) % max(1, self.config.snapshot_freq) == 0:
            self._save_epoch_snapshot(epoch, train_metrics)

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'disc_real_acc': [],
            'disc_fake_acc': []
        }
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    real_data = batch[0].to(self.device)
                else:
                    real_data = batch.to(self.device)
                
                batch_size = real_data.size(0)
                noise = torch.randn(batch_size, self.noise_dim, device=self.device)
                
                # Generate fake data
                fake_data = self.generator(noise)
                
                # Discriminator outputs
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data)
                
                # Losses
                real_loss = self.adversarial_loss(real_output, True)
                fake_loss = self.adversarial_loss(fake_output, False)
                disc_loss = (real_loss + fake_loss) / 2
                gen_loss = self.adversarial_loss(fake_output, True)
                
                # Accuracies
                if self.loss_type == "bce":
                    real_acc = (torch.sigmoid(real_output) > 0.5).float().mean()
                    fake_acc = (torch.sigmoid(fake_output) < 0.5).float().mean()
                else:
                    # For other loss types, use sign-based accuracy
                    real_acc = (real_output > 0).float().mean()
                    fake_acc = (fake_output < 0).float().mean()
                
                val_metrics['gen_loss'].append(gen_loss.item())
                val_metrics['disc_loss'].append(disc_loss.item())
                val_metrics['disc_real_acc'].append(real_acc.item())
                val_metrics['disc_fake_acc'].append(fake_acc.item())
        
        return {key: np.mean(values) for key, values in val_metrics.items()}
    
    def _update_progressive_growing(self, batch_idx: int):
        """Update progressive growing parameters."""
        if hasattr(self.generator, 'grow') and hasattr(self.generator, 'set_alpha'):
            # Simple progressive growing logic
            if batch_idx > 0 and batch_idx % 1000 == 0:  # Grow every 1000 batches
                self.generator.grow()
                if hasattr(self.discriminator, 'grow'):
                    self.discriminator.grow()
                self.progressive_level += 1
                self.logger.info(f"Progressive growing: advanced to level {self.progressive_level}")
    
    def _update_curriculum_learning(self, epoch_metrics: Dict[str, List[float]]):
        """Update curriculum learning parameters."""
        if len(epoch_metrics.get('disc_real_acc', [])) > 10:  # Need some history
            recent_acc = np.mean(epoch_metrics['disc_real_acc'][-10:])
            if recent_acc > self.curriculum_threshold and self.curriculum_stage < 3:
                self.curriculum_stage += 1
                self.logger.info(f"Curriculum learning: advanced to stage {self.curriculum_stage}")
    
    def generate_samples(self, num_samples: int = 16, use_fixed_noise: bool = True) -> torch.Tensor:
        """
        Generate samples for visualization.
        
        Args:
            num_samples: Number of samples to generate
            use_fixed_noise: Whether to use fixed noise for consistency
            
        Returns:
            Generated samples
        """
        generator = self._generator_for_sampling()
        generator.eval()
        
        with torch.no_grad():
            if use_fixed_noise and num_samples <= len(self.fixed_noise):
                noise = self.fixed_noise[:num_samples]
            else:
                noise = torch.randn(num_samples, self.noise_dim, device=self.device)
            
            samples = generator(noise)
        
        return samples
    
    def save_checkpoint(self, path: str, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save GAN training checkpoint.
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'ema_generator_state_dict': self.ema_generator.state_dict() if self.ema_generator is not None else None,
            'discriminator_state_dict': self.discriminator.state_dict(),
            'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
            'disc_optimizer_state_dict': self.disc_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'best_loss': self.best_loss,
            'progressive_level': self.progressive_level,
            'curriculum_stage': self.curriculum_stage,
            'loss_type': self.loss_type,
            'use_gradient_penalty': self.use_gradient_penalty,
            'gradient_penalty_weight': self.gradient_penalty_weight
        }
        
        if self.gen_scheduler:
            checkpoint['gen_scheduler_state_dict'] = self.gen_scheduler.state_dict()
        if self.disc_scheduler:
            checkpoint['disc_scheduler_state_dict'] = self.disc_scheduler.state_dict()
        
        if self.scaler:
            checkpoint['gen_scaler_state_dict'] = self.scaler.state_dict()
        if self.disc_scaler:
            checkpoint['disc_scaler_state_dict'] = self.disc_scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
        
        self.logger.info(f"GAN checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Load GAN training checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint data
        """
        # weights_only=False: our own checkpoints pickle TrainingConfig
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if self.ema_generator is not None and checkpoint.get('ema_generator_state_dict') is not None:
            self.ema_generator.load_state_dict(checkpoint['ema_generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        
        if self.gen_scheduler and 'gen_scheduler_state_dict' in checkpoint:
            self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler_state_dict'])
        if self.disc_scheduler and 'disc_scheduler_state_dict' in checkpoint:
            self.disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        
        if self.scaler and 'gen_scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['gen_scaler_state_dict'])
        if self.disc_scaler and 'disc_scaler_state_dict' in checkpoint:
            self.disc_scaler.load_state_dict(checkpoint['disc_scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.progressive_level = checkpoint.get('progressive_level', 0)
        self.curriculum_stage = checkpoint.get('curriculum_stage', 0)
        
        # Update training parameters
        self.loss_type = checkpoint.get('loss_type', self.loss_type)
        self.use_gradient_penalty = checkpoint.get('use_gradient_penalty', self.use_gradient_penalty)
        self.gradient_penalty_weight = checkpoint.get('gradient_penalty_weight', self.gradient_penalty_weight)
        
        self.logger.info(f"GAN checkpoint loaded: {path}")
        return checkpoint
    
    def get_training_info(self) -> Dict[str, Any]:
        """Get comprehensive training information."""
        info = super().get_training_info()
        info.update({
            "loss_type": self.loss_type,
            "use_gradient_penalty": self.use_gradient_penalty,
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "progressive_level": self.progressive_level,
            "curriculum_stage": self.curriculum_stage,
            "noise_dim": self.noise_dim,
            "generator_params": sum(p.numel() for p in self.generator.parameters()),
            "discriminator_params": sum(p.numel() for p in self.discriminator.parameters()),
            "augment": getattr(self.config, "augment", "none"),
            "augment_p": self.augment_p,
            "use_ema": self.ema_generator is not None,
            "collapse_events": self.collapse_events,
            "dataset_occupancy_mean": getattr(self.config, "dataset_occupancy_mean", None),
            "last_epoch_metrics": self.last_epoch_metrics,
        })
        return info
