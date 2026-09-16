"""
GAN Discriminator models for DeepSculpt PyTorch implementation.

This module contains all discriminator architectures for 3D GAN models,
including simple, complex, progressive, and conditional discriminators.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List, Union
import math

from ..base_models import BaseDiscriminator, SparseConv3d, SparseBatchNorm3d


class SimpleDiscriminator(BaseDiscriminator):
    """Simple discriminator model equivalent to TensorFlow version."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        # Convolution layers
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(512)
        
        # Final classification layer
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.Linear(512 * final_size ** 3, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Don't apply sigmoid - loss function expects logits
        
        return x


class ComplexDiscriminator(BaseDiscriminator):
    """Complex discriminator with additional layers and features."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        # Convolution layers with more complexity
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 32, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(32)
        
        self.conv2 = Conv(32, 64, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(64)
        
        self.conv3 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(128)
        
        self.conv4 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(256)
        
        self.conv5 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn5 = BatchNorm(512)
        
        # Additional feature extraction
        self.conv6 = Conv(512, 1024, 3, 1, 1, bias=False)
        self.bn6 = BatchNorm(1024)
        
        # Final classification layers
        final_size = void_dim // 32  # After 5 conv layers with stride 2
        self.fc1 = nn.Linear(1024 * final_size ** 3, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.leaky_relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Don't apply sigmoid - loss function expects logits
        
        return x


class ProgressiveDiscriminator(BaseDiscriminator):
    """Progressive discriminator for high-resolution 3D data."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, max_resolution: int = 128, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.max_resolution = max_resolution
        
        # Progressive blocks for different resolutions
        self.progressive_blocks = nn.ModuleList()
        self.from_rgb_layers = nn.ModuleList()
        
        # Create progressive blocks for each resolution level
        current_res = max_resolution
        current_channels = 16
        
        while current_res >= 8:
            # From RGB layer for this resolution
            Conv = SparseConv3d if sparse else nn.Conv3d
            from_rgb = Conv(self.input_channels, current_channels, 1, 1, 0)
            self.from_rgb_layers.append(from_rgb)
            
            # Progressive block
            block = self._make_progressive_block(current_channels, current_channels * 2)
            self.progressive_blocks.append(block)
            
            current_channels *= 2
            current_res //= 2
        
        # Final classification block
        self.final_block = self._make_final_block(current_channels)
        
        self.current_level = 0  # Current progressive level
        self.alpha = 1.0  # Blending factor for progressive growing
    
    def _make_progressive_block(self, in_channels: int, out_channels: int):
        """Create a progressive block that halves the resolution."""
        Conv = SparseConv3d if self.sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            BatchNorm(in_channels),
            nn.LeakyReLU(0.2),
            Conv(in_channels, out_channels, 3, 1, 1),
            BatchNorm(out_channels),
            nn.LeakyReLU(0.2),
            nn.AvgPool3d(2, 2)
        )
    
    def _make_final_block(self, in_channels: int):
        """Create the final classification block."""
        Conv = SparseConv3d if self.sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if self.sparse else nn.BatchNorm3d
        
        return nn.Sequential(
            Conv(in_channels, in_channels, 3, 1, 1),
            BatchNorm(in_channels),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convert from RGB at current resolution
        x = self.from_rgb_layers[self.current_level](x)
        
        # Progressive blocks up to current level
        for i in range(self.current_level, len(self.progressive_blocks)):
            x = self.progressive_blocks[i](x)
        
        # Final classification
        x = self.final_block(x)
        
        return x
    
    def grow(self):
        """Grow the network by one level."""
        if self.current_level < len(self.progressive_blocks) - 1:
            self.current_level += 1
            self.alpha = 0.0  # Start with full blend to new layer
    
    def set_alpha(self, alpha: float):
        """Set the blending factor for progressive growing."""
        self.alpha = max(0.0, min(1.0, alpha))


class ConditionalDiscriminator(BaseDiscriminator):
    """Conditional discriminator for controlled discrimination."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, condition_dim: int = 10, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.condition_dim = condition_dim
        
        # Condition embedding
        self.condition_embedding = nn.Embedding(condition_dim, 128)
        
        # Convolution layers
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1, bias=False)
        self.bn1 = BatchNorm(64)
        
        self.conv2 = Conv(64, 128, 4, 2, 1, bias=False)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1, bias=False)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 2, 1, bias=False)
        self.bn4 = BatchNorm(512)
        
        # Final classification layer with condition
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.Linear(512 * final_size ** 3 + 128, 1)  # +128 for condition embedding
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Add condition if provided
        if condition is not None:
            condition_embedded = self.condition_embedding(condition)
            x = torch.cat([x, condition_embedded], dim=1)
        
        # Final classification — return logits for softplus loss
        x = self.fc(x)

        return x


class SpectralNormDiscriminator(BaseDiscriminator):
    """Discriminator with spectral normalization for training stability."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        # Convolution layers with spectral normalization
        Conv = SparseConv3d if sparse else nn.Conv3d
        
        self.conv1 = nn.utils.spectral_norm(Conv(self.input_channels, 64, 4, 2, 1, bias=False))
        self.conv2 = nn.utils.spectral_norm(Conv(64, 128, 4, 2, 1, bias=False))
        self.conv3 = nn.utils.spectral_norm(Conv(128, 256, 4, 2, 1, bias=False))
        self.conv4 = nn.utils.spectral_norm(Conv(256, 512, 4, 2, 1, bias=False))
        
        # Final classification layer with spectral normalization
        final_size = void_dim // 16  # After 4 conv layers with stride 2
        self.fc = nn.utils.spectral_norm(nn.Linear(512 * final_size ** 3, 1))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)
        
        # Convolution blocks without batch normalization (spectral norm instead)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.leaky_relu(x)
        
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class MultiScaleDiscriminator(BaseDiscriminator):
    """Multi-scale discriminator for improved training dynamics."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, num_scales: int = 3, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.num_scales = num_scales
        self.discriminators = nn.ModuleList()

        # Each sub-discriminator must match the resolution it actually receives
        # (full, /2, /4, ...). SimpleDiscriminator needs void_dim // 16 >= 1.
        smallest = void_dim // (2 ** (num_scales - 1))
        if smallest < 16:
            raise ValueError(
                f"MultiScaleDiscriminator with num_scales={num_scales} needs "
                f"void_dim >= {16 * 2 ** (num_scales - 1)} (got void_dim={void_dim})"
            )
        for i in range(num_scales):
            scale_discriminator = SimpleDiscriminator(void_dim // (2 ** i), color_mode, sparse)
            self.discriminators.append(scale_discriminator)

        # Downsampling layers for different scales
        self.downsample = nn.AvgPool3d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        current_x = x

        for i, discriminator in enumerate(self.discriminators):
            output = discriminator(current_x)
            outputs.append(output)

            # Downsample for next scale (except for the last one)
            if i < self.num_scales - 1:
                current_x = self.downsample(current_x)

        # Mean of per-scale logits -> (B, 1); keeps gradients flowing to all scales
        # and satisfies the trainer contract shared by every discriminator.
        return torch.stack(outputs, dim=0).mean(dim=0)


class PatchDiscriminator(BaseDiscriminator):
    """Patch-based discriminator (PatchGAN) for local discrimination."""
    
    def __init__(self, void_dim: int = 64, color_mode: int = 1, patch_size: int = 16, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        
        self.patch_size = patch_size
        
        # Convolution layers for patch discrimination
        Conv = SparseConv3d if sparse else nn.Conv3d
        BatchNorm = SparseBatchNorm3d if sparse else nn.BatchNorm3d
        
        self.conv1 = Conv(self.input_channels, 64, 4, 2, 1)
        self.conv2 = Conv(64, 128, 4, 2, 1)
        self.bn2 = BatchNorm(128)
        
        self.conv3 = Conv(128, 256, 4, 2, 1)
        self.bn3 = BatchNorm(256)
        
        self.conv4 = Conv(256, 512, 4, 1, 1)
        self.bn4 = BatchNorm(512)
        
        # Final patch classification
        self.conv5 = Conv(512, 1, 4, 1, 1)

        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input is already in PyTorch channels-first format: (batch, channels, depth, height, width)

        # Convolution blocks
        x = self.conv1(x)
        x = self.leaky_relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)
        
        # Final patch logits (B, 1, d, h, w) -> mean-pool patches to (B, 1):
        # standard PatchGAN aggregation, keeps the trainer contract uniform.
        x = self.conv5(x)
        x = x.mean(dim=[2, 3, 4])

        return x


class LightDiscriminator(BaseDiscriminator):
    """Lightweight projection discriminator for 3D sparse voxel GANs.

    Deliberately small (~500K params) to balance against a stronger generator.
    Features:
    - Spectral norm for Lipschitz stability
    - Projection head: judges mean + variance of features (global statistics)
      instead of specific voxel patterns — harder to memorize
    - Returns intermediate features for feature matching loss
    """

    def __init__(self, void_dim: int = 64, color_mode: int = 1, sparse: bool = False):
        super().__init__(void_dim, color_mode, sparse)
        Conv = SparseConv3d if sparse else nn.Conv3d
        ch = 32
        self.conv1 = nn.utils.spectral_norm(Conv(self.input_channels, ch, 4, 2, 1))
        self.conv2 = nn.utils.spectral_norm(Conv(ch, ch * 2, 4, 2, 1))
        self.conv3 = nn.utils.spectral_norm(Conv(ch * 2, ch * 4, 4, 2, 1))
        self.pool = nn.AdaptiveAvgPool3d(1)
        # Projection: mean + variance of features → harder to overfit
        self.fc = nn.utils.spectral_norm(nn.Linear(ch * 8, 1))
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        f1 = self.lrelu(self.conv1(x))
        f2 = self.lrelu(self.conv2(f1))
        f3 = self.lrelu(self.conv3(f2))

        # Projection: global mean + spatial variance
        pooled = self.pool(f3).flatten(1)         # (B, ch*4)
        std_pool = f3.std(dim=[2, 3, 4])          # (B, ch*4)
        proj = torch.cat([pooled, std_pool], dim=1)  # (B, ch*8)
        logit = self.fc(proj)

        if return_features:
            return logit, [f1, f2, f3]
        return logit


# ---------------------------------------------------------------------------
# Ported building blocks (sh4174/3DStyleGAN + stke9/SliceGAN)
# ---------------------------------------------------------------------------
#
# NEEDS-GPU-VALIDATION: the two classes below are shape-verified only. Their
# training dynamics (mode-collapse suppression for the stddev layer, 2D->3D
# convergence for the slicing critic) have NOT been validated on a real GPU
# training run. See the RunPod validation checklist in the PR.


class MinibatchStdDev3D(nn.Module):
    """3D minibatch standard-deviation layer (StyleGAN2, lifted to 3D).

    Port of ``minibatch_stddev_layer`` from sh4174/3DStyleGAN
    (``training/networks3d_stylegan2.py:139``). Computes the per-feature
    stddev across a group of samples, averages it to a single scalar, and
    appends it as one extra constant channel to every voxel of every sample.

    This gives the discriminator a cheap statistic of batch diversity: when
    the generator mode-collapses the batch stddev drops, the appended channel
    goes flat, and D can trivially call the batch fake. Classic anti-mode-
    collapse trick — a drop-in for the final block of any 3D critic.

    Args:
        group_size: samples per stddev group (clamped to the batch size).
        num_new_features: number of stddev channels to append (default 1).
    """

    def __init__(self, group_size: int = 4, num_new_features: int = 1):
        super().__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, D, H, W)
        n, c, d, h, w = x.shape
        # Group size divides the batch; fall back to the whole batch otherwise.
        g = min(self.group_size, n)
        while n % g != 0:
            g -= 1
        f = self.num_new_features
        # (G, n//G, F, C//F, D, H, W)
        y = x.reshape(g, -1, f, c // f, d, h, w)
        y = y - y.mean(dim=0, keepdim=True)          # subtract group mean
        y = y.square().mean(dim=0)                    # variance over group
        y = (y + 1e-8).sqrt()                         # stddev
        y = y.mean(dim=[2, 3, 4, 5], keepdim=True)    # average over C//F,D,H,W
        y = y.squeeze(2)                              # (n//G, F, 1, 1, 1)
        y = y.repeat(g, 1, d, h, w)                   # broadcast back to (N, F, D, H, W)
        return torch.cat([x, y], dim=1)               # append as new channels


class SliceDiscriminator2D(BaseDiscriminator):
    """SliceGAN 2D critic that judges axial slices of a 3D volume.

    Port of the 2D->3D slicing trick from stke9/SliceGAN
    (``slicegan/model.py:96-135``, ``networks.py``). Instead of a 3D critic,
    a purely 2D WGAN critic scores every slice taken along each of the three
    principal axes; the 3D volume is turned into a batch of 2D images via the
    permute/reshape trick

        vol.permute(0, d1, 1, d2, d3).reshape(l * B, C, l, l)

    with ``(d1,d2,d3)`` cycling ``[2,3,4], [3,2,2], [4,4,3]`` for the x/y/z
    axes. This lets deep-sculpt train a **3D generator from 2D reference
    imagery** (drawings, photos, textures) for which no 3D ground truth
    exists — the single most novel capability in the ported cluster.

    Registry-selectable as ``"slice"``. ``forward`` accepts either a 3D volume
    ``(B, C, D, H, W)`` — sliced internally along ``axis`` — or a pre-sliced
    batch of 2D images ``(B, C, H, W)``.

    The 2D critic's input width is taken from ``BaseDiscriminator.input_channels``
    (6 for ``color_mode=1``, 1 for mono) so a sliced ``(l*B, C, l, l)`` batch
    carries the same channel count as the volume it came from.

    Args:
        void_dim: side length of the (cubic) 3D volume.
        color_mode: 0 mono (1ch), 1 color (6ch) — same convention as the 3D critics.
        axis: which axis to slice when a 3D volume is passed (0=x,1=y,2=z).
        spectral: wrap convs in spectral_norm (WGAN critic stability).
    """

    def __init__(
        self,
        void_dim: int = 64,
        color_mode: int = 1,
        sparse: bool = False,
        axis: int = 0,
        spectral: bool = True,
    ):
        super().__init__(void_dim, color_mode, sparse)
        # self.input_channels is set by BaseDiscriminator (6 color / 1 mono).

        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2 (got {axis})")
        self.axis = axis
        self.spectral = spectral

        def maybe_sn(module: nn.Module) -> nn.Module:
            return nn.utils.spectral_norm(module) if spectral else module

        # 2D critic: 4 strided convs, no batchnorm (WGAN-GP friendly).
        self.conv1 = maybe_sn(nn.Conv2d(self.input_channels, 64, 4, 2, 1))
        self.conv2 = maybe_sn(nn.Conv2d(64, 128, 4, 2, 1))
        self.conv3 = maybe_sn(nn.Conv2d(128, 256, 4, 2, 1))
        self.conv4 = maybe_sn(nn.Conv2d(256, 512, 4, 2, 1))
        final_size = void_dim // 16  # after 4 stride-2 convs
        self.fc = maybe_sn(nn.Linear(512 * final_size * final_size, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)

    def slice_volume(self, volume: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        """Turn a (B, C, D, H, W) volume into a (l*B, C, l, l) batch of slices.

        Implements SliceGAN's 2D->3D trick (``slicegan/model.py:96-135``): the
        volume is sliced along the chosen spatial ``axis`` and every slice
        becomes an independent 2D image in the batch. Equivalent to SliceGAN's
        ``permute(0, d1, 1, d2, d3).reshape(l*B, C, l, l)`` but expressed with
        ``movedim`` so it is axis-correct for all three axes (the literal
        SliceGAN permute table only round-trips for the isotropic-cube case).
        """
        axis = self.axis if axis is None else axis
        b, c = volume.shape[0], volume.shape[1]
        spatial_dim = 2 + axis           # tensor dim of the sliced spatial axis
        length = volume.shape[spatial_dim]
        # Move the sliced axis next to the batch dim, then fold it into batch.
        # (B, C, D, H, W) -> (B, L, C, l, l) -> (L*B, C, l, l)
        moved = volume.movedim(spatial_dim, 1)          # (B, L, C, r1, r2)
        return moved.reshape(b * length, c, moved.shape[3], moved.shape[4])

    def _score_2d(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2(x))
        x = self.leaky_relu(self.conv3(x))
        x = self.leaky_relu(self.conv4(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)  # logits; WGAN loss handled by the trainer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 5D input -> slice the volume; 4D input -> already-sliced 2D batch.
        if x.dim() == 5:
            x = self.slice_volume(x)
        elif x.dim() != 4:
            raise ValueError(
                f"SliceDiscriminator2D expects a 5D volume or 4D slice batch, got {x.dim()}D"
            )
        return self._score_2d(x)