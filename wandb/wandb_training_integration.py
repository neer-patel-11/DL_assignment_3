"""
wandb_training_integration.py — Integration for Actual Training Loops
DA6401 Assignment 3: Transformer NMT

This module provides helper functions to integrate W&B logging into your actual
training code. Use these functions in your train.py to log metrics during training.

Features:
- Automatic W&B initialization with experiment configuration
- Batch-level and epoch-level logging
- Attention visualization during training
- Gradient analysis for experiments
- Checkpoint management with W&B artifacts
- Integration with LabelSmoothingLoss and NoamScheduler
"""

import os
import torch
import numpy as np
from typing import Dict, Optional, Any, Tuple
from pathlib import Path
import json

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

from wandb_config import RunGroupManager, WandBMetricsLogger


# ═══════════════════════════════════════════════════════════════════════════════
#                     TRAINING INTEGRATION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

class WandBTrainingHelper:
    """Helper class to integrate W&B logging into training loops."""
    
    def __init__(self, 
                 group_key: str = None,
                 variant_key: str = None,
                 use_wandb: bool = True,
                 run_name: Optional[str] = None,
                 additional_config: Optional[Dict] = None):
        """
        Initialize W&B training helper.
        
        Args:
            group_key: Experiment group key (e.g., 'exp1_noam_scheduler')
            variant_key: Variant key (e.g., 'noam', 'fixed')
            use_wandb: Whether to use W&B (default True)
            run_name: Custom run name
            additional_config: Additional config to merge
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.run = None
        self.config = None
        self.logger = None
        self.metrics_buffer = {}
        self.step = 0
        self.epoch = 0
        
        if self.use_wandb and group_key and variant_key:
            self.initialize_run(group_key, variant_key, run_name, additional_config)
    
    def initialize_run(self, 
                       group_key: str,
                       variant_key: str,
                       run_name: Optional[str] = None,
                       additional_config: Optional[Dict] = None):
        """Initialize W&B run."""
        if not self.use_wandb:
            return
        
        manager = RunGroupManager()
        self.run, self.config = manager.initialize_wandb_run(
            group_key, 
            variant_key,
            run_name,
            additional_config
        )
        self.logger = WandBMetricsLogger(self.run)
    
    def log_batch(self, 
                  loss: float,
                  lr: float,
                  metrics: Optional[Dict] = None,
                  log_immediately: bool = True):
        """
        Log training batch metrics.
        
        Args:
            loss: Training loss
            lr: Current learning rate
            metrics: Additional metrics dictionary
            log_immediately: If True, log immediately; else buffer
        """
        if not self.use_wandb:
            return
        
        self.metrics_buffer.update({
            "loss": loss,
            "lr": lr,
            **(metrics or {})
        })
        
        if log_immediately:
            self.log_buffer()
    
    def log_buffer(self):
        """Log buffered metrics to W&B."""
        if not self.use_wandb or not self.metrics_buffer:
            return
        
        if self.logger:
            self.logger.log_training_batch(
                epoch=self.epoch,
                batch_idx=self.step,
                loss=self.metrics_buffer.get("loss", 0),
                lr=self.metrics_buffer.get("lr", 0),
                metrics={k: v for k, v in self.metrics_buffer.items() 
                        if k not in ["loss", "lr"]}
            )
        self.metrics_buffer = {}
        self.step += 1
    
    def log_validation(self,
                      val_loss: float,
                      bleu_score: float,
                      metrics: Optional[Dict] = None):
        """
        Log validation metrics.
        
        Args:
            val_loss: Validation loss
            bleu_score: BLEU score
            metrics: Additional metrics
        """
        if not self.use_wandb or not self.logger:
            return
        
        self.logger.log_validation(
            epoch=self.epoch,
            val_loss=val_loss,
            bleu_score=bleu_score,
            metrics=metrics
        )
    
    def log_attention_weights(self,
                             attention_weights: torch.Tensor,
                             layer: int,
                             head: int,
                             num_heads: int = 8):
        """
        Log attention weights for visualization.
        
        Args:
            attention_weights: Shape [seq_len, seq_len]
            layer: Layer index
            head: Head index
            num_heads: Total number of heads
        """
        if not self.use_wandb or not self.logger:
            return
        
        if isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu()
        
        self.logger.log_attention_heatmap(
            attention_weights.float() if isinstance(attention_weights, torch.Tensor) else attention_weights,
            layer=layer,
            head=head,
            step=self.step
        )
    
    def log_gradient_analysis(self,
                             model: torch.nn.Module,
                             analyze_layers: Optional[list] = None):
        """
        Log gradient norms for specific layers.
        
        Args:
            model: Model to analyze
            analyze_layers: List of layer names to analyze (e.g., ['attention', 'feed_forward'])
        """
        if not self.use_wandb or not self.logger:
            return
        
        analyze_layers = analyze_layers or ['attention', 'feed_forward']
        
        for name, param in model.named_parameters():
            # Filter by layer names if specified
            if analyze_layers and not any(layer in name for layer in analyze_layers):
                continue
            
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                # Log only significant gradients
                if grad_norm > 1e-8:
                    self.logger.run.log({
                        f"gradient/{name}": grad_norm,
                        "step": self.step
                    })
    
    def log_prediction_analysis(self,
                               logits: torch.Tensor,
                               targets: torch.Tensor):
        """
        Log prediction confidence and accuracy.
        
        Args:
            logits: Model output logits [batch, seq_len, vocab_size]
            targets: Ground truth tokens [batch, seq_len]
        """
        if not self.use_wandb or not self.logger:
            return
        
        self.logger.log_prediction_confidence(logits, targets, self.step)
    
    def save_checkpoint(self,
                       checkpoint_path: str,
                       metric_name: str = "latest"):
        """
        Save checkpoint and upload to W&B.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metric_name: Name for artifact (e.g., 'best', 'latest')
        """
        if not self.use_wandb or not self.logger:
            return
        
        if os.path.exists(checkpoint_path):
            self.logger.save_checkpoint_artifact(checkpoint_path, self.epoch, metric_name)
    
    def log_config_summary(self):
        """Log configuration summary."""
        if not self.use_wandb or not self.config:
            return
        
        # Log config to W&B
        if self.run:
            self.run.config.update({
                "model": self.config.get("model", {}),
                "training": self.config.get("training", {}),
                "dataset": self.config.get("dataset", {}),
            })
    
    def increment_epoch(self):
        """Increment epoch counter."""
        self.epoch += 1
    
    def increment_step(self):
        """Increment step counter."""
        self.step += 1
    
    def finish(self):
        """Finish W&B run."""
        if self.use_wandb and self.logger:
            self.logger.finish()
    
    def get_config(self) -> Optional[Dict]:
        """Get experiment configuration."""
        return self.config


# ═══════════════════════════════════════════════════════════════════════════════
#                      TRAINING LOOP INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

class WandBTrainingHook:
    """
    Context manager for W&B integration in training loops.
    
    Usage:
        with WandBTrainingHook("exp1_noam_scheduler", "noam") as hook:
            for epoch in range(num_epochs):
                for batch in dataloader:
                    loss = train_step(batch)
                    hook.log_batch(loss, lr)
                
                val_loss, bleu = evaluate()
                hook.log_validation(val_loss, bleu)
                hook.increment_epoch()
    """
    
    def __init__(self,
                 group_key: Optional[str] = None,
                 variant_key: Optional[str] = None,
                 use_wandb: bool = True,
                 run_name: Optional[str] = None):
        self.helper = WandBTrainingHelper(
            group_key=group_key,
            variant_key=variant_key,
            use_wandb=use_wandb,
            run_name=run_name
        )
    
    def __enter__(self):
        self.helper.log_config_summary()
        return self.helper
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.helper.finish()
        if exc_type is not None:
            print(f"Error during training: {exc_val}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#                      EXPERIMENT-SPECIFIC HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_experiment_1_noam_scheduler(use_wandb: bool = True) -> WandBTrainingHelper:
    """Setup W&B for Experiment 1: Noam Scheduler vs Fixed LR."""
    return WandBTrainingHelper(
        group_key="exp1_noam_scheduler",
        variant_key="noam",  # Change to "fixed" for fixed LR variant
        use_wandb=use_wandb,
        run_name="exp1_noam_scheduler_full_training"
    )


def setup_experiment_2_scaling_factor(use_wandb: bool = True) -> WandBTrainingHelper:
    """Setup W&B for Experiment 2: Scaling Factor."""
    return WandBTrainingHelper(
        group_key="exp2_scaling_factor",
        variant_key="with_scaling",  # Change to "without_scaling" for ablation
        use_wandb=use_wandb,
        run_name="exp2_scaling_factor_full_training"
    )


def setup_experiment_3_attention(use_wandb: bool = True) -> WandBTrainingHelper:
    """Setup W&B for Experiment 3: Attention Rollout."""
    return WandBTrainingHelper(
        group_key="exp3_attention_rollout",
        variant_key="baseline",
        use_wandb=use_wandb,
        run_name="exp3_attention_rollout_full_training"
    )


def setup_experiment_4_positional(use_wandb: bool = True) -> WandBTrainingHelper:
    """Setup W&B for Experiment 4: Positional Encoding."""
    return WandBTrainingHelper(
        group_key="exp4_positional_encoding",
        variant_key="sinusoidal",  # Change to "learned" for learned embeddings
        use_wandb=use_wandb,
        run_name="exp4_positional_encoding_full_training"
    )


def setup_experiment_5_label_smoothing(use_wandb: bool = True) -> WandBTrainingHelper:
    """Setup W&B for Experiment 5: Label Smoothing."""
    return WandBTrainingHelper(
        group_key="exp5_label_smoothing",
        variant_key="with_smoothing",  # Change to "without_smoothing" for ablation
        use_wandb=use_wandb,
        run_name="exp5_label_smoothing_full_training"
    )


# ═══════════════════════════════════════════════════════════════════════════════
#                      DECORATOR FOR TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def wandb_training(group_key: Optional[str] = None,
                   variant_key: Optional[str] = None,
                   use_wandb: bool = True):
    """
    Decorator to automatically setup W&B for training functions.
    
    Usage:
        @wandb_training("exp1_noam_scheduler", "noam")
        def train_model(helper, model, train_loader, val_loader, num_epochs):
            # Training code here
            # Use helper.log_batch(), helper.log_validation(), etc.
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            helper = WandBTrainingHelper(
                group_key=group_key,
                variant_key=variant_key,
                use_wandb=use_wandb
            )
            helper.log_config_summary()
            
            try:
                # Call original function with helper as first argument
                result = func(helper, *args, **kwargs)
            finally:
                helper.finish()
            
            return result
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
#                      EXAMPLE TRAINING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

def example_training_loop():
    """
    Example showing how to integrate W&B into your training loop.
    
    This is a template - replace with your actual training code.
    """
    
    # Initialize W&B for specific experiment
    with WandBTrainingHook("exp1_noam_scheduler", "noam") as hook:
        num_epochs = 10
        num_batches = 100
        
        # Get configuration (optional)
        config = hook.get_config()
        print(f"Training with config: {config}")
        
        for epoch in range(num_epochs):
            hook.increment_epoch()
            
            for batch_idx in range(num_batches):
                # Simulate training step
                loss = 5.0 * np.exp(-(epoch * num_batches + batch_idx) / 500)
                lr = 0.0001
                
                # Log batch metrics
                hook.log_batch(
                    loss=loss,
                    lr=lr,
                    metrics={
                        "gradient_norm": 0.1,
                        "batch_size": 32
                    }
                )
                
                if batch_idx % 10 == 0:
                    hook.increment_step()
            
            # Simulate validation
            val_loss = loss * 1.05
            bleu_score = 20.0 + (epoch / num_epochs) * 25
            
            hook.log_validation(
                val_loss=val_loss,
                bleu_score=bleu_score,
                metrics={"perplexity": np.exp(val_loss)}
            )


if __name__ == "__main__":
    # Test the integration
    print("W&B Training Integration Test")
    print("=" * 80)
    
    # Show available setup functions
    print("\nAvailable experiment setup functions:")
    print("  - setup_experiment_1_noam_scheduler()")
    print("  - setup_experiment_2_scaling_factor()")
    print("  - setup_experiment_3_attention()")
    print("  - setup_experiment_4_positional()")
    print("  - setup_experiment_5_label_smoothing()")
    
    # Run example (with W&B disabled for testing)
    print("\nRunning example training loop (W&B disabled)...")
    example_training_loop()
    print("✓ Example completed")