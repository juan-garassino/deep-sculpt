"""Shape / import tests for the SliceGAN + 3DStyleGAN stability port.

These are GPU-training-port smoke tests: they verify that the ported building
blocks import, forward-pass with the correct shapes, and produce finite
gradients on a *tiny* batch. They do NOT attempt convergence / real training —
that is deferred to a RunPod GPU run (see the PR's RunPod validation checklist).

Ports covered:
  - SliceDiscriminator2D  (stke9/SliceGAN 2D->3D slicing critic)
  - MinibatchStdDev3D     (sh4174/3DStyleGAN anti-mode-collapse layer)
  - r1_gradient_penalty   (sh4174/3DStyleGAN R1, WGAN-GP alternative)
  - msssim_3d / soft_dice / occupancy_class_weight / max_connected_component
    (3DStyleGAN + vnet.pytorch + 3dgan-release eval bundle)
"""

import numpy as np
import pytest
import torch

from deepsculpt.core.models.gan.discriminator import (
    MinibatchStdDev3D,
    SliceDiscriminator2D,
)
from deepsculpt.core.models.model_factory import PyTorchModelFactory
from deepsculpt.core.training.gan_trainer import r1_gradient_penalty
from deepsculpt.core.training.training_metrics import (
    max_connected_component,
    msssim_3d,
    occupancy_class_weight,
    soft_dice,
)


# --------------------------------------------------------------------------
# SliceGAN slicing discriminator
# --------------------------------------------------------------------------

def test_slice_discriminator_in_registry():
    assert "slice" in PyTorchModelFactory.DISCRIMINATOR_REGISTRY
    assert (
        PyTorchModelFactory.DISCRIMINATOR_REGISTRY["slice"] is SliceDiscriminator2D
    )


def test_slice_discriminator_factory_build():
    factory = PyTorchModelFactory(device="cpu")
    disc = factory.create_gan_discriminator("slice", void_dim=32, color_mode=1)
    assert isinstance(disc, SliceDiscriminator2D)
    assert disc.input_channels == 6  # color_mode=1 -> 6 channels (main convention)


def test_slice_discriminator_forward_from_volume():
    """A (B,C,D,H,W) volume is sliced to (l*B, C, l, l) and scored -> (l*B, 1)."""
    disc = SliceDiscriminator2D(void_dim=32, color_mode=1)
    batch, void = 2, 32
    vol = torch.randn(batch, 6, void, void, void)
    out = disc(vol)
    # 32 slices per sample, 2 samples -> 64 slice logits.
    assert out.shape == (void * batch, 1)
    assert torch.isfinite(out).all()


def test_slice_discriminator_forward_from_2d_batch():
    """A pre-sliced (B, C, H, W) 2D batch is scored directly -> (B, 1)."""
    disc = SliceDiscriminator2D(void_dim=32, color_mode=1)
    imgs = torch.randn(5, 6, 32, 32)
    out = disc(imgs)
    assert out.shape == (5, 1)


def test_slice_discriminator_mono_channels():
    disc = SliceDiscriminator2D(void_dim=32, color_mode=0)
    assert disc.input_channels == 1
    out = disc(torch.randn(2, 1, 32, 32, 32))
    assert out.shape == (64, 1)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_slice_volume_all_axes(axis):
    """slice_volume must produce (l*B, C, l, l) for every principal axis."""
    disc = SliceDiscriminator2D(void_dim=32, color_mode=1, axis=axis)
    vol = torch.randn(2, 6, 32, 32, 32)
    sliced = disc.slice_volume(vol)
    assert sliced.shape == (32 * 2, 6, 32, 32)


def test_slice_discriminator_rejects_bad_axis():
    with pytest.raises(ValueError):
        SliceDiscriminator2D(void_dim=32, axis=3)


def test_slice_discriminator_rejects_wrong_rank():
    disc = SliceDiscriminator2D(void_dim=32, color_mode=1)
    with pytest.raises(ValueError):
        disc(torch.randn(2, 6, 32))  # 3D -> not a volume or slice batch


# --------------------------------------------------------------------------
# 3DStyleGAN minibatch-stddev layer
# --------------------------------------------------------------------------

def test_minibatch_stddev_appends_channel():
    layer = MinibatchStdDev3D(group_size=4)
    x = torch.randn(4, 8, 6, 6, 6)
    out = layer(x)
    # Spatial dims unchanged, one stddev channel appended.
    assert out.shape == (4, 9, 6, 6, 6)
    assert torch.isfinite(out).all()


def test_minibatch_stddev_handles_odd_batch():
    """group_size is clamped to divide the batch (batch=3, group=4 -> group=3)."""
    layer = MinibatchStdDev3D(group_size=4)
    out = layer(torch.randn(3, 4, 4, 4, 4))
    assert out.shape == (3, 5, 4, 4, 4)


def test_minibatch_stddev_flat_batch_gives_zero_channel():
    """Identical samples -> zero batch stddev -> appended channel is ~0."""
    layer = MinibatchStdDev3D(group_size=4)
    x = torch.ones(4, 3, 4, 4, 4)
    out = layer(x)
    appended = out[:, -1]
    assert torch.allclose(appended, torch.zeros_like(appended), atol=1e-4)


# --------------------------------------------------------------------------
# 3DStyleGAN R1 penalty (WGAN-GP alternative)
# --------------------------------------------------------------------------

def test_r1_penalty_scalar_and_finite():
    disc = SliceDiscriminator2D(void_dim=16, color_mode=1, spectral=False)
    real = torch.randn(2, 6, 16, 16, 16)
    r1 = r1_gradient_penalty(disc, real, gamma=10.0)
    assert r1.dim() == 0            # scalar
    assert torch.isfinite(r1)
    assert r1.item() >= 0.0         # squared-gradient penalty is non-negative


def test_r1_penalty_backprops():
    """R1 must produce finite gradients on the discriminator params (tiny batch)."""
    disc = SliceDiscriminator2D(void_dim=16, color_mode=1, spectral=False)
    real = torch.randn(2, 6, 16, 16, 16)
    r1 = r1_gradient_penalty(disc, real, gamma=10.0)
    r1.backward()
    grads = [p.grad for p in disc.parameters() if p.grad is not None]
    assert grads, "R1 penalty produced no discriminator gradients"
    assert all(torch.isfinite(g).all() for g in grads)


# --------------------------------------------------------------------------
# Volumetric eval bundle
# --------------------------------------------------------------------------

def test_msssim_3d_identity_is_one():
    x = torch.rand(2, 1, 16, 16, 16)
    score = msssim_3d(x, x, levels=3)
    assert score.dim() == 0
    assert score.item() == pytest.approx(1.0, abs=1e-3)


def test_msssim_3d_shape_mismatch_raises():
    with pytest.raises(ValueError):
        msssim_3d(torch.rand(1, 1, 8, 8, 8), torch.rand(1, 1, 4, 8, 8))


def test_soft_dice_perfect_and_disjoint():
    a = (torch.rand(2, 1, 8, 8, 8) > 0.5).float()
    assert soft_dice(a, a).item() == pytest.approx(1.0, abs=1e-3)
    # Disjoint masks -> Dice ~ 0.
    b = torch.zeros_like(a)
    b[a == 0] = 1.0
    assert soft_dice(a, b).item() < 0.1


def test_occupancy_class_weight_recipe():
    # bg_weight = mean / (1 + mean)
    assert occupancy_class_weight(0.0) == pytest.approx(0.0)
    assert occupancy_class_weight(1.0) == pytest.approx(0.5)


def test_max_connected_component_numpy_keeps_largest():
    vol = np.zeros((8, 8, 8), dtype=np.float32)
    vol[1:5, 1:5, 1:5] = 1.0   # big 4x4x4 blob (64 voxels)
    vol[7, 7, 7] = 1.0         # floating debris voxel
    cleaned = max_connected_component(vol, threshold=0.5)
    assert cleaned.shape == vol.shape
    assert cleaned[7, 7, 7] == 0.0            # debris removed
    assert cleaned[2, 2, 2] == 1.0            # main blob kept
    assert int(cleaned.sum()) == 64


def test_max_connected_component_torch_roundtrip():
    vol = torch.zeros(1, 8, 8, 8)
    vol[0, 1:5, 1:5, 1:5] = 1.0
    vol[0, 7, 7, 7] = 1.0
    cleaned = max_connected_component(vol, threshold=0.5)
    assert isinstance(cleaned, torch.Tensor)
    assert cleaned.shape == vol.shape
    assert cleaned[0, 7, 7, 7].item() == 0.0
    assert int(cleaned.sum().item()) == 64
