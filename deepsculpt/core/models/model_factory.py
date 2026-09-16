"""
Model factory for creating PyTorch models in DeepSculpt.

This module provides a centralized factory for creating all types of models
including GAN generators, discriminators, and diffusion models with consistent
interfaces and configuration management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Type, List
import warnings

# Import model classes
from .base_models import BaseGenerator, BaseDiscriminator, BaseDiffusionModel
from .gan.generator import (
    SimpleGenerator, ComplexGenerator, SkipGenerator, MonochromeGenerator,
    AutoencoderGenerator, ProgressiveGenerator, ConditionalGenerator
)
from .gan.discriminator import (
    SimpleDiscriminator, ComplexDiscriminator, ProgressiveDiscriminator,
    ConditionalDiscriminator, SpectralNormDiscriminator, MultiScaleDiscriminator,
    PatchDiscriminator, LightDiscriminator, SliceDiscriminator2D
)
from .diffusion.unet import UNet3D, ConditionalUNet3D
from .diffusion.noise_scheduler import NoiseScheduler, DDIMScheduler, DPMSolverScheduler, AdaptiveScheduler
from .diffusion.pipeline import Diffusion3DPipeline, ConditionalDiffusion3DPipeline, FastSamplingPipeline


class PyTorchModelFactory:
    """
    Factory class for creating PyTorch models with consistent configuration.
    
    Provides methods to create generators, discriminators, diffusion models,
    and complete pipelines with proper parameter validation and device management.
    """
    
    # Registry of available model classes
    GENERATOR_REGISTRY = {
        "simple": SimpleGenerator,
        "complex": ComplexGenerator,
        "skip": SkipGenerator,
        "monochrome": MonochromeGenerator,
        "autoencoder": AutoencoderGenerator,
        "progressive": ProgressiveGenerator,
        "conditional": ConditionalGenerator,
    }
    
    DISCRIMINATOR_REGISTRY = {
        "simple": SimpleDiscriminator,
        "complex": ComplexDiscriminator,
        "progressive": ProgressiveDiscriminator,
        "conditional": ConditionalDiscriminator,
        "spectral_norm": SpectralNormDiscriminator,
        "multi_scale": MultiScaleDiscriminator,
        "patch": PatchDiscriminator,
        "light": LightDiscriminator,
        "slice": SliceDiscriminator2D,  # SliceGAN 2D->3D slicing critic (from-2D-imagery)
    }
    
    DIFFUSION_REGISTRY = {
        "unet3d": UNet3D,
        "conditional_unet3d": ConditionalUNet3D,
    }
    
    SCHEDULER_REGISTRY = {
        "linear": NoiseScheduler,
        "cosine": NoiseScheduler,
        "ddim": DDIMScheduler,
        "dpm_solver": DPMSolverScheduler,
        "adaptive": AdaptiveScheduler,
    }
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the model factory.
        
        Args:
            device: Device to create models on ("auto", "cuda", "cpu")
        """
        self.device = self._setup_device(device)
        self.created_models = []  # Track created models for management
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate the compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA requested but not available, falling back to CPU")
            return "cpu"
        
        return device
    
    def create_gan_generator(
        self,
        model_type: str,
        void_dim: int = 64,
        noise_dim: int = 100,
        color_mode: int = 1,
        sparse: bool = False,
        **kwargs
    ) -> BaseGenerator:
        """
        Create a GAN generator model.
        
        Args:
            model_type: Type of generator ("simple", "complex", "skip", etc.)
            void_dim: Dimension of the 3D output space
            noise_dim: Dimension of the input noise vector
            color_mode: Color mode (0 for monochrome, 1 for color)
            sparse: Whether to use sparse tensor operations
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized generator model
        """
        if model_type not in self.GENERATOR_REGISTRY:
            raise ValueError(f"Unknown generator type: {model_type}. Available: {list(self.GENERATOR_REGISTRY.keys())}")
        
        generator_class = self.GENERATOR_REGISTRY[model_type]
        
        # Create model with common parameters
        model_kwargs = {
            "void_dim": void_dim,
            "noise_dim": noise_dim,
            "color_mode": color_mode,
            "sparse": sparse,
        }
        
        # Add model-specific parameters
        if model_type == "progressive":
            model_kwargs["max_resolution"] = kwargs.get("max_resolution", 128)
        elif model_type == "conditional":
            model_kwargs["condition_dim"] = kwargs.get("condition_dim", 10)
        
        # Add any additional kwargs
        model_kwargs.update(kwargs)
        
        # Create and initialize model
        model = generator_class(**model_kwargs)
        model = model.to(self.device)
        
        # Initialize weights
        self._initialize_weights(model)
        
        self.created_models.append(model)
        return model
    
    def create_gan_discriminator(
        self,
        model_type: str,
        void_dim: int = 64,
        color_mode: int = 1,
        sparse: bool = False,
        **kwargs
    ) -> BaseDiscriminator:
        """
        Create a GAN discriminator model.
        
        Args:
            model_type: Type of discriminator ("simple", "complex", "progressive", etc.)
            void_dim: Dimension of the 3D input space
            color_mode: Color mode (0 for monochrome, 1 for color)
            sparse: Whether to use sparse tensor operations
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized discriminator model
        """
        if model_type not in self.DISCRIMINATOR_REGISTRY:
            raise ValueError(f"Unknown discriminator type: {model_type}. Available: {list(self.DISCRIMINATOR_REGISTRY.keys())}")
        
        discriminator_class = self.DISCRIMINATOR_REGISTRY[model_type]
        
        # Create model with common parameters
        model_kwargs = {
            "void_dim": void_dim,
            "color_mode": color_mode,
            "sparse": sparse,
        }
        
        # Add model-specific parameters
        if model_type == "progressive":
            model_kwargs["max_resolution"] = kwargs.get("max_resolution", 128)
        elif model_type == "conditional":
            model_kwargs["condition_dim"] = kwargs.get("condition_dim", 10)
        elif model_type == "multi_scale":
            model_kwargs["num_scales"] = kwargs.get("num_scales", 3)
        elif model_type == "patch":
            model_kwargs["patch_size"] = kwargs.get("patch_size", 16)
        elif model_type == "slice":
            model_kwargs["axis"] = kwargs.get("axis", 0)
            model_kwargs["spectral"] = kwargs.get("spectral", True)
        
        # Add any additional kwargs
        model_kwargs.update(kwargs)
        
        # Create and initialize model
        model = discriminator_class(**model_kwargs)
        model = model.to(self.device)
        
        # Initialize weights
        self._initialize_weights(model)
        
        self.created_models.append(model)
        return model
    
    def create_diffusion_model(
        self,
        model_type: str,
        void_dim: int = 64,
        in_channels: int = 6,
        out_channels: int = 6,
        timesteps: int = 1000,
        sparse: bool = False,
        **kwargs
    ) -> BaseDiffusionModel:
        """
        Create a diffusion model.
        
        Args:
            model_type: Type of diffusion model ("unet3d", "conditional_unet3d")
            void_dim: Dimension of the 3D space
            in_channels: Number of input channels
            out_channels: Number of output channels
            timesteps: Number of diffusion timesteps
            sparse: Whether to use sparse tensor operations
            **kwargs: Additional model-specific parameters
            
        Returns:
            Initialized diffusion model
        """
        if model_type not in self.DIFFUSION_REGISTRY:
            raise ValueError(f"Unknown diffusion model type: {model_type}. Available: {list(self.DIFFUSION_REGISTRY.keys())}")
        
        diffusion_class = self.DIFFUSION_REGISTRY[model_type]
        
        # Create model with common parameters
        model_kwargs = {
            "void_dim": void_dim,
            "in_channels": in_channels,
            "out_channels": out_channels,
            "timesteps": timesteps,
            "sparse": sparse,
        }
        
        # Add model-specific parameters
        if model_type == "conditional_unet3d":
            model_kwargs["conditioning_dim"] = kwargs.get("conditioning_dim", 512)
            model_kwargs["num_classes"] = kwargs.get("num_classes", None)
        
        # Add common UNet parameters
        model_kwargs.update({
            "model_channels": kwargs.get("model_channels", 128),
            "num_res_blocks": kwargs.get("num_res_blocks", 2),
            "attention_resolutions": kwargs.get("attention_resolutions", [16, 8]),
            "channel_mult": kwargs.get("channel_mult", [1, 2, 4, 8]),
            "num_heads": kwargs.get("num_heads", 8),
            "dropout": kwargs.get("dropout", 0.1),
        })
        
        # Add any additional kwargs
        model_kwargs.update(kwargs)
        
        # Create and initialize model
        model = diffusion_class(**model_kwargs)
        model = model.to(self.device)
        
        # Initialize weights
        self._initialize_weights(model)
        
        self.created_models.append(model)
        return model
    
    def create_noise_scheduler(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        **kwargs
    ) -> NoiseScheduler:
        """
        Create a noise scheduler for diffusion models.
        
        Args:
            schedule_type: Type of schedule ("linear", "cosine", "ddim", etc.)
            timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            **kwargs: Additional scheduler-specific parameters
            
        Returns:
            Initialized noise scheduler
        """
        if schedule_type not in self.SCHEDULER_REGISTRY:
            raise ValueError(f"Unknown scheduler type: {schedule_type}. Available: {list(self.SCHEDULER_REGISTRY.keys())}")
        
        scheduler_class = self.SCHEDULER_REGISTRY[schedule_type]
        
        # Create scheduler with common parameters
        scheduler_kwargs = {
            "schedule_type": schedule_type if schedule_type in ["linear", "cosine"] else "linear",
            "timesteps": timesteps,
            "beta_start": beta_start,
            "beta_end": beta_end,
            "device": self.device,
        }
        
        # Add scheduler-specific parameters
        if schedule_type == "ddim":
            scheduler_kwargs.update({
                "eta": kwargs.get("eta", 0.0),
                "clip_sample": kwargs.get("clip_sample", True),
            })
        elif schedule_type == "dpm_solver":
            scheduler_kwargs.update({
                "solver_order": kwargs.get("solver_order", 2),
                "prediction_type": kwargs.get("prediction_type", "epsilon"),
            })
        elif schedule_type == "adaptive":
            scheduler_kwargs.update({
                "adaptation_rate": kwargs.get("adaptation_rate", 0.01),
            })
        
        # Add any additional kwargs
        scheduler_kwargs.update(kwargs)
        
        return scheduler_class(**scheduler_kwargs)
    
    def create_diffusion_pipeline(
        self,
        model: nn.Module,
        scheduler: NoiseScheduler,
        pipeline_type: str = "standard",
        **kwargs
    ) -> Diffusion3DPipeline:
        """
        Create a complete diffusion pipeline.
        
        Args:
            model: Diffusion model
            scheduler: Noise scheduler
            pipeline_type: Type of pipeline ("standard", "conditional", "fast")
            **kwargs: Additional pipeline parameters
            
        Returns:
            Initialized diffusion pipeline
        """
        pipeline_kwargs = {
            "model": model,
            "noise_scheduler": scheduler,
            "device": self.device,
            "prediction_type": kwargs.get("prediction_type", "epsilon"),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
        }
        
        if pipeline_type == "conditional":
            pipeline_kwargs["conditioning_dropout"] = kwargs.get("conditioning_dropout", 0.1)
            return ConditionalDiffusion3DPipeline(**pipeline_kwargs)
        elif pipeline_type == "fast":
            pipeline_kwargs["scheduler_type"] = kwargs.get("scheduler_type", "ddim")
            return FastSamplingPipeline(**pipeline_kwargs)
        else:
            return Diffusion3DPipeline(**pipeline_kwargs)
    
    def create_gan_pair(
        self,
        generator_type: str,
        discriminator_type: str,
        void_dim: int = 64,
        noise_dim: int = 100,
        color_mode: int = 1,
        sparse: bool = False,
        **kwargs
    ) -> tuple[BaseGenerator, BaseDiscriminator]:
        """
        Create a matched generator-discriminator pair.
        
        Args:
            generator_type: Type of generator
            discriminator_type: Type of discriminator
            void_dim: Dimension of the 3D space
            noise_dim: Dimension of noise vector
            color_mode: Color mode
            sparse: Whether to use sparse operations
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (generator, discriminator)
        """
        generator = self.create_gan_generator(
            generator_type, void_dim, noise_dim, color_mode, sparse, **kwargs
        )
        discriminator = self.create_gan_discriminator(
            discriminator_type, void_dim, color_mode, sparse, **kwargs
        )
        
        return generator, discriminator
    
    def _initialize_weights(self, model: nn.Module):
        """Initialize model weights using best practices."""
        for module in model.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def get_model_info(self, model: nn.Module) -> Dict[str, Any]:
        """Get detailed information about a model."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        info = {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(next(model.parameters()).device),
            "memory_usage_mb": self._estimate_model_memory(model),
        }
        
        # Add model-specific info if available
        if hasattr(model, 'get_model_info'):
            info.update(model.get_model_info())
        
        return info
    
    def _estimate_model_memory(self, model: nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available model types."""
        return {
            "generators": list(self.GENERATOR_REGISTRY.keys()),
            "discriminators": list(self.DISCRIMINATOR_REGISTRY.keys()),
            "diffusion_models": list(self.DIFFUSION_REGISTRY.keys()),
            "schedulers": list(self.SCHEDULER_REGISTRY.keys()),
        }
    
    def clear_created_models(self):
        """Clear the list of created models."""
        self.created_models.clear()
    
    def get_created_models_info(self) -> List[Dict[str, Any]]:
        """Get information about all created models."""
        return [self.get_model_info(model) for model in self.created_models]


# Convenience functions for backward compatibility
def create_pytorch_generator(model_type: str = "skip", **kwargs) -> BaseGenerator:
    """Create a PyTorch generator model."""
    factory = PyTorchModelFactory()
    return factory.create_gan_generator(model_type, **kwargs)


def create_pytorch_discriminator(model_type: str = "simple", **kwargs) -> BaseDiscriminator:
    """Create a PyTorch discriminator model."""
    factory = PyTorchModelFactory()
    return factory.create_gan_discriminator(model_type, **kwargs)


def create_pytorch_diffusion_model(model_type: str = "unet3d", **kwargs) -> BaseDiffusionModel:
    """Create a PyTorch diffusion model."""
    factory = PyTorchModelFactory()
    return factory.create_diffusion_model(model_type, **kwargs)


def create_pytorch_gan_pair(
    generator_type: str = "skip",
    discriminator_type: str = "simple",
    **kwargs
) -> tuple[BaseGenerator, BaseDiscriminator]:
    """Create a matched PyTorch GAN generator-discriminator pair."""
    factory = PyTorchModelFactory()
    return factory.create_gan_pair(generator_type, discriminator_type, **kwargs)