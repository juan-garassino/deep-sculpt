"""
Training metrics and logging utilities for DeepSculpt PyTorch implementation.

This module provides comprehensive metrics tracking, visualization, and analysis
tools for monitoring training progress and model performance.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Union, Tuple
from collections import defaultdict, deque
import json
import time
from pathlib import Path
import warnings

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class MetricsBuffer:
    """
    Circular buffer for storing metrics with efficient statistics computation.
    """
    
    def __init__(self, maxlen: int = 1000):
        """
        Initialize metrics buffer.
        
        Args:
            maxlen: Maximum number of values to store
        """
        self.maxlen = maxlen
        self.buffer = deque(maxlen=maxlen)
        self.sum = 0.0
        self.sum_sq = 0.0
        self.count = 0
    
    def append(self, value: float):
        """Add a value to the buffer."""
        if len(self.buffer) == self.maxlen:
            # Remove oldest value from running sums
            old_value = self.buffer[0]
            self.sum -= old_value
            self.sum_sq -= old_value ** 2
            self.count -= 1
        
        self.buffer.append(value)
        self.sum += value
        self.sum_sq += value ** 2
        self.count += 1
    
    def mean(self) -> float:
        """Get mean of values in buffer."""
        return self.sum / max(self.count, 1)
    
    def std(self) -> float:
        """Get standard deviation of values in buffer."""
        if self.count < 2:
            return 0.0
        variance = (self.sum_sq / self.count) - (self.mean() ** 2)
        return max(variance, 0.0) ** 0.5
    
    def min(self) -> float:
        """Get minimum value in buffer."""
        return min(self.buffer) if self.buffer else 0.0
    
    def max(self) -> float:
        """Get maximum value in buffer."""
        return max(self.buffer) if self.buffer else 0.0
    
    def percentile(self, p: float) -> float:
        """Get percentile of values in buffer."""
        if not self.buffer:
            return 0.0
        return np.percentile(list(self.buffer), p)
    
    def get_stats(self) -> Dict[str, float]:
        """Get comprehensive statistics."""
        return {
            'mean': self.mean(),
            'std': self.std(),
            'min': self.min(),
            'max': self.max(),
            'count': self.count,
            'p25': self.percentile(25),
            'p50': self.percentile(50),
            'p75': self.percentile(75),
            'p95': self.percentile(95)
        }


class TrainingMetrics:
    """
    Comprehensive metrics tracking and analysis for training.
    """
    
    def __init__(self, buffer_size: int = 1000, save_frequency: int = 100):
        """
        Initialize training metrics tracker.
        
        Args:
            buffer_size: Size of metrics buffer for statistics
            save_frequency: Frequency of automatic saves
        """
        self.buffer_size = buffer_size
        self.save_frequency = save_frequency
        
        # Metrics storage
        self.step_metrics = defaultdict(lambda: MetricsBuffer(buffer_size))
        self.epoch_metrics = defaultdict(list)
        self.custom_metrics = defaultdict(list)
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.start_time = time.time()
        self.epoch_start_time = time.time()
        
        # Best metrics tracking
        self.best_metrics = {}
        self.best_epochs = {}
        
        # Convergence tracking
        self.convergence_window = 50
        self.convergence_threshold = 1e-4
        self.stagnation_patience = 100
        
        # Anomaly detection
        self.anomaly_threshold = 3.0  # Standard deviations
        self.anomalies = []
    
    def update_step_metrics(self, metrics: Dict[str, float]):
        """
        Update step-level metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.step_metrics[name].append(value)
                
                # Anomaly detection
                if self.step_metrics[name].count > 10:  # Need some history
                    mean = self.step_metrics[name].mean()
                    std = self.step_metrics[name].std()
                    if std > 0 and abs(value - mean) > self.anomaly_threshold * std:
                        self.anomalies.append({
                            'step': self.current_step,
                            'metric': name,
                            'value': value,
                            'expected_range': (mean - 2*std, mean + 2*std)
                        })
        
        self.current_step += 1
        
        # Periodic save
        if self.current_step % self.save_frequency == 0:
            self._auto_save()
    
    def update_epoch_metrics(self, metrics: Dict[str, float]):
        """
        Update epoch-level metrics.
        
        Args:
            metrics: Dictionary of metric values
        """
        epoch_time = time.time() - self.epoch_start_time
        
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                self.epoch_metrics[name].append(value)
                
                # Track best metrics
                if name.endswith('_loss') or 'error' in name.lower():
                    # Lower is better
                    if name not in self.best_metrics or value < self.best_metrics[name]:
                        self.best_metrics[name] = value
                        self.best_epochs[name] = self.current_epoch
                else:
                    # Higher is better (accuracy, etc.)
                    if name not in self.best_metrics or value > self.best_metrics[name]:
                        self.best_metrics[name] = value
                        self.best_epochs[name] = self.current_epoch
        
        # Add timing information
        self.epoch_metrics['epoch_time'].append(epoch_time)
        self.epoch_metrics['total_time'].append(time.time() - self.start_time)
        
        self.current_epoch += 1
        self.epoch_start_time = time.time()
    
    def add_custom_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Add a custom metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Step number (uses current step if None)
        """
        step = step or self.current_step
        self.custom_metrics[name].append((step, value))
    
    def get_current_stats(self) -> Dict[str, Dict[str, float]]:
        """Get current statistics for all metrics."""
        stats = {}
        
        # Step metrics statistics
        for name, buffer in self.step_metrics.items():
            stats[f"step_{name}"] = buffer.get_stats()
        
        # Epoch metrics statistics
        for name, values in self.epoch_metrics.items():
            if values:
                stats[f"epoch_{name}"] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1],
                    'count': len(values)
                }
        
        return stats
    
    def get_convergence_info(self, metric_name: str = "train_loss") -> Dict[str, Any]:
        """
        Analyze convergence for a specific metric.
        
        Args:
            metric_name: Name of metric to analyze
            
        Returns:
            Dictionary with convergence information
        """
        if metric_name not in self.epoch_metrics or len(self.epoch_metrics[metric_name]) < self.convergence_window:
            return {"converged": False, "reason": "insufficient_data"}
        
        values = self.epoch_metrics[metric_name]
        recent_values = values[-self.convergence_window:]
        
        # Check for convergence (low variance in recent values)
        variance = np.var(recent_values)
        mean_value = np.mean(recent_values)
        relative_variance = variance / (mean_value ** 2) if mean_value != 0 else variance
        
        converged = relative_variance < self.convergence_threshold
        
        # Check for stagnation
        if len(values) > self.stagnation_patience:
            old_mean = np.mean(values[-self.stagnation_patience:-self.convergence_window])
            new_mean = np.mean(recent_values)
            improvement = abs(old_mean - new_mean) / old_mean if old_mean != 0 else 0
            stagnated = improvement < self.convergence_threshold
        else:
            stagnated = False
        
        return {
            "converged": converged,
            "stagnated": stagnated,
            "relative_variance": relative_variance,
            "recent_mean": mean_value,
            "recent_std": np.std(recent_values),
            "trend": self._calculate_trend(recent_values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return "unknown"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def detect_training_issues(self) -> Dict[str, List[str]]:
        """
        Detect common training issues.
        
        Returns:
            Dictionary of detected issues
        """
        issues = defaultdict(list)
        
        # Check for exploding gradients (rapid loss increase)
        if "train_loss" in self.step_metrics:
            recent_losses = list(self.step_metrics["train_loss"].buffer)[-20:]
            if len(recent_losses) >= 10:
                if any(loss > 10 * np.mean(recent_losses[:10]) for loss in recent_losses[-10:]):
                    issues["exploding_gradients"].append("Loss increased rapidly")
        
        # Check for vanishing gradients (loss plateau)
        convergence_info = self.get_convergence_info("train_loss")
        if convergence_info.get("stagnated", False):
            issues["vanishing_gradients"].append("Training loss has stagnated")
        
        # Check for overfitting
        if "train_loss" in self.epoch_metrics and "val_loss" in self.epoch_metrics:
            if len(self.epoch_metrics["train_loss"]) >= 10:
                recent_train = np.mean(self.epoch_metrics["train_loss"][-5:])
                recent_val = np.mean(self.epoch_metrics["val_loss"][-5:])
                if recent_val > 1.5 * recent_train:
                    issues["overfitting"].append("Validation loss much higher than training loss")
        
        # Check for learning rate issues
        if "train_loss" in self.epoch_metrics and len(self.epoch_metrics["train_loss"]) >= 20:
            recent_trend = self._calculate_trend(self.epoch_metrics["train_loss"][-10:])
            if recent_trend == "increasing":
                issues["learning_rate"].append("Loss is increasing - learning rate may be too high")
            elif recent_trend == "stable" and self.epoch_metrics["train_loss"][-1] > 0.1:
                issues["learning_rate"].append("Loss plateaued at high value - learning rate may be too low")
        
        # Check for data issues (NaN/Inf values)
        if self.anomalies:
            recent_anomalies = [a for a in self.anomalies if a['step'] > self.current_step - 1000]
            if len(recent_anomalies) > 10:
                issues["data_quality"].append(f"Many anomalous values detected: {len(recent_anomalies)}")
        
        return dict(issues)
    
    def plot_metrics(
        self,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_anomalies: bool = True
    ) -> plt.Figure:
        """
        Plot training metrics.
        
        Args:
            metrics: List of metrics to plot (plots all if None)
            save_path: Path to save plot
            show_anomalies: Whether to highlight anomalies
            
        Returns:
            Matplotlib figure
        """
        if metrics is None:
            metrics = list(self.epoch_metrics.keys())
        
        # Filter out timing metrics for main plot
        plot_metrics = [m for m in metrics if not m.endswith('_time')]
        
        n_metrics = len(plot_metrics)
        if n_metrics == 0:
            return plt.figure()
        
        # Create subplots
        fig, axes = plt.subplots(
            (n_metrics + 1) // 2, 2 if n_metrics > 1 else 1,
            figsize=(12, 4 * ((n_metrics + 1) // 2))
        )
        
        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 1:
            axes = axes.flatten()
        
        for i, metric_name in enumerate(plot_metrics):
            ax = axes[i] if n_metrics > 1 else axes[0]
            
            if metric_name in self.epoch_metrics:
                values = self.epoch_metrics[metric_name]
                epochs = range(len(values))
                
                ax.plot(epochs, values, label=metric_name, linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric_name)
                ax.set_title(f'{metric_name.replace("_", " ").title()} Over Time')
                ax.grid(True, alpha=0.3)
                
                # Highlight best value
                if metric_name in self.best_metrics:
                    best_epoch = self.best_epochs[metric_name]
                    best_value = self.best_metrics[metric_name]
                    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7)
                    ax.annotate(f'Best: {best_value:.4f}', 
                              xy=(best_epoch, best_value),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                
                # Show anomalies
                if show_anomalies and metric_name in [a['metric'] for a in self.anomalies]:
                    anomaly_epochs = [a['step'] // 100 for a in self.anomalies if a['metric'] == metric_name]  # Rough conversion
                    anomaly_values = [values[min(e, len(values)-1)] for e in anomaly_epochs if e < len(values)]
                    if anomaly_values:
                        ax.scatter(anomaly_epochs[:len(anomaly_values)], anomaly_values, 
                                 color='red', s=50, alpha=0.7, label='Anomalies')
        
        # Remove empty subplots
        if n_metrics > 1:
            for i in range(n_metrics, len(axes)):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_loss_landscape(self, save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training and validation loss landscape.
        
        Args:
            save_path: Path to save plot
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss curves
        if "train_loss" in self.epoch_metrics:
            epochs = range(len(self.epoch_metrics["train_loss"]))
            ax1.plot(epochs, self.epoch_metrics["train_loss"], label='Training Loss', linewidth=2)
        
        if "val_loss" in self.epoch_metrics:
            epochs = range(len(self.epoch_metrics["val_loss"]))
            ax1.plot(epochs, self.epoch_metrics["val_loss"], label='Validation Loss', linewidth=2)
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss distribution
        if "train_loss" in self.step_metrics:
            train_losses = list(self.step_metrics["train_loss"].buffer)
            ax2.hist(train_losses, bins=50, alpha=0.7, label='Training Loss', density=True)
        
        ax2.set_xlabel('Loss Value')
        ax2.set_ylabel('Density')
        ax2.set_title('Loss Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def export_metrics(self, filepath: str):
        """
        Export metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        export_data = {
            'step_metrics': {name: list(buffer.buffer) for name, buffer in self.step_metrics.items()},
            'epoch_metrics': dict(self.epoch_metrics),
            'custom_metrics': dict(self.custom_metrics),
            'best_metrics': self.best_metrics,
            'best_epochs': self.best_epochs,
            'anomalies': self.anomalies,
            'training_info': {
                'current_step': self.current_step,
                'current_epoch': self.current_epoch,
                'total_time': time.time() - self.start_time
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _auto_save(self):
        """Automatically save metrics periodically."""
        try:
            save_dir = Path("metrics_autosave")
            save_dir.mkdir(exist_ok=True)
            
            timestamp = int(time.time())
            filepath = save_dir / f"metrics_{timestamp}.json"
            
            self.export_metrics(str(filepath))
            
            # Keep only last 5 autosaves
            autosave_files = sorted(save_dir.glob("metrics_*.json"))
            if len(autosave_files) > 5:
                for old_file in autosave_files[:-5]:
                    old_file.unlink()
                    
        except Exception as e:
            warnings.warn(f"Auto-save failed: {e}")
    
    def get_summary_report(self) -> str:
        """
        Generate a comprehensive training summary report.
        
        Returns:
            Formatted summary report
        """
        report = []
        report.append("=" * 60)
        report.append("TRAINING SUMMARY REPORT")
        report.append("=" * 60)
        
        # Basic info
        total_time = time.time() - self.start_time
        report.append(f"Total Training Time: {total_time/3600:.2f} hours")
        report.append(f"Current Epoch: {self.current_epoch}")
        report.append(f"Current Step: {self.current_step}")
        report.append("")
        
        # Best metrics
        report.append("BEST METRICS:")
        report.append("-" * 20)
        for metric, value in self.best_metrics.items():
            epoch = self.best_epochs[metric]
            report.append(f"{metric}: {value:.6f} (epoch {epoch})")
        report.append("")
        
        # Convergence analysis
        report.append("CONVERGENCE ANALYSIS:")
        report.append("-" * 25)
        for metric in ["train_loss", "val_loss"]:
            if metric in self.epoch_metrics:
                conv_info = self.get_convergence_info(metric)
                report.append(f"{metric}:")
                report.append(f"  Converged: {conv_info['converged']}")
                report.append(f"  Trend: {conv_info['trend']}")
                report.append(f"  Recent Mean: {conv_info['recent_mean']:.6f}")
        report.append("")
        
        # Training issues
        issues = self.detect_training_issues()
        if issues:
            report.append("DETECTED ISSUES:")
            report.append("-" * 20)
            for issue_type, descriptions in issues.items():
                report.append(f"{issue_type.upper()}:")
                for desc in descriptions:
                    report.append(f"  - {desc}")
        else:
            report.append("No training issues detected.")
        report.append("")
        
        # Anomalies
        if self.anomalies:
            recent_anomalies = [a for a in self.anomalies if a['step'] > self.current_step - 1000]
            report.append(f"ANOMALIES: {len(recent_anomalies)} in last 1000 steps")
        
        return "\n".join(report)


# ---------------------------------------------------------------------------
# Volumetric eval bundle (ported: 3DStyleGAN + vnet.pytorch + 3dgan-release)
# ---------------------------------------------------------------------------
#
# NEEDS-GPU-VALIDATION: these are shape/numerics-verified helpers only; their
# usefulness as a volumetric-FID stand-in / reconstruction signal on the real
# ds-improve eval loop has not been validated against a trained model.


def msssim_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    levels: int = 3,
    window: int = 7,
    sigma: float = 1.5,
) -> torch.Tensor:
    """3D multi-scale SSIM between two occupancy/feature volumes.

    Port of ``msssim_3d`` from sh4174/3DStyleGAN (``metrics/ms_ssim.py:46``):
    a Gaussian-windowed SSIM computed at ``levels`` scales via an
    ``avg_pool3d`` pyramid, geometric-mean combined. Serves as a
    volumetric-FID stand-in / sample-similarity metric for the ds-improve loop.

    Inputs are ``(B, C, D, H, W)`` (any C); returns a scalar mean SSIM in
    roughly [0, 1] (1 == identical).
    """
    if x.shape != y.shape:
        raise ValueError(f"msssim_3d: shape mismatch {tuple(x.shape)} vs {tuple(y.shape)}")
    x = x.float()
    y = y.float()
    channels = x.shape[1]

    # Separable 3D Gaussian kernel.
    coords = torch.arange(window, dtype=torch.float32, device=x.device) - (window - 1) / 2.0
    g1 = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g1 = g1 / g1.sum()
    kernel = g1[:, None, None] * g1[None, :, None] * g1[None, None, :]
    kernel = kernel.expand(channels, 1, window, window, window).contiguous()
    pad = window // 2
    c1, c2 = 0.01 ** 2, 0.03 ** 2

    def _ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        mu_a = F.conv3d(a, kernel, padding=pad, groups=channels)
        mu_b = F.conv3d(b, kernel, padding=pad, groups=channels)
        mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
        var_a = F.conv3d(a * a, kernel, padding=pad, groups=channels) - mu_a2
        var_b = F.conv3d(b * b, kernel, padding=pad, groups=channels) - mu_b2
        cov = F.conv3d(a * b, kernel, padding=pad, groups=channels) - mu_ab
        ssim_map = ((2 * mu_ab + c1) * (2 * cov + c2)) / (
            (mu_a2 + mu_b2 + c1) * (var_a + var_b + c2)
        )
        return ssim_map.mean()

    scores = []
    a, b = x, y
    for level in range(levels):
        scores.append(_ssim(a, b).clamp_min(1e-8))
        if level < levels - 1:
            # Downsample the pyramid; guard against volumes too small to pool.
            if min(a.shape[2:]) < 2:
                break
            a = F.avg_pool3d(a, 2)
            b = F.avg_pool3d(b, 2)
    stacked = torch.stack(scores)
    return stacked.log().mean().exp()  # geometric mean of per-scale SSIM


def soft_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice coefficient for sparse voxel occupancy.

    Port of the Dice objective from mattmacy/vnet.pytorch
    (``train.py:320`` / ``bioloss.dice_loss``). Sparse sculptures have extreme
    foreground/background imbalance (occupancy a few %), where Dice is a far
    better reconstruction/eval signal than raw BCE. Returns Dice in [0, 1]
    (1 == perfect overlap); ``1 - soft_dice`` is the loss form.

    Both tensors are occupancy fields (any matching shape); values are treated
    as soft probabilities so this is differentiable.
    """
    if pred.shape != target.shape:
        raise ValueError(f"soft_dice: shape mismatch {tuple(pred.shape)} vs {tuple(target.shape)}")
    p = pred.float().reshape(pred.shape[0], -1)
    t = target.float().reshape(target.shape[0], -1)
    intersection = (p * t).sum(dim=1)
    denom = p.sum(dim=1) + t.sum(dim=1)
    dice = (2 * intersection + eps) / (denom + eps)
    return dice.mean()


def occupancy_class_weight(target_mean: float) -> float:
    """Occupancy-derived background weight for imbalanced voxel CE/BCE.

    One-liner recipe from vnet.pytorch (``train.py:225``):
    ``bg_weight = target_mean / (1 + target_mean)``. Weights the abundant
    background class down in proportion to dataset occupancy.
    """
    return target_mean / (1.0 + target_mean)


def max_connected_component(
    volume,
    threshold: float = 0.5,
    connectivity: int = 1,
):
    """Keep only the largest connected occupied component (debris cleanup).

    Port of ``max_connected`` from zck119/3dgan-release
    (``visualization/python/util.py:110``), rewritten with
    ``scipy.ndimage.label`` instead of the original O(n³) Python flood fill.
    GAN outputs frequently contain floating single-voxel debris; this keeps the
    largest 6/18/26-connected blob and zeros the rest — an artifact-free export
    post-process for ``volume_export``.

    Accepts a numpy array or a torch tensor ``(D, H, W)`` (or ``(1, D, H, W)``);
    returns the same type/shape, binarized to {0, 1}.
    """
    from scipy import ndimage

    is_torch = isinstance(volume, torch.Tensor)
    arr = volume.detach().cpu().numpy() if is_torch else np.asarray(volume)
    squeezed = arr.ndim == 4 and arr.shape[0] == 1
    work = arr[0] if squeezed else arr

    occupied = work > threshold
    structure = ndimage.generate_binary_structure(3, connectivity)
    labels, n = ndimage.label(occupied, structure=structure)
    if n > 0:
        # Component sizes (index 0 is background — ignore it).
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = int(sizes.argmax())
        cleaned = (labels == keep).astype(arr.dtype)
    else:
        cleaned = np.zeros_like(work, dtype=arr.dtype)

    if squeezed:
        cleaned = cleaned[None, ...]
    if is_torch:
        return torch.from_numpy(cleaned).to(volume.device)
    return cleaned