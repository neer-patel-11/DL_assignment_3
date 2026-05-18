"""
wandb_config.py — Weights & Biases Configuration & Project Setup
DA6401 Assignment 3: Transformer NMT with Organized Run Groups

This module centralizes all W&B configurations, run groups, and logging setup.
All experiments are organized under a single project with structured group names.

Run Groups (Required Experiments):
  1. Experiment_1_Noam_vs_FixedLR      - Noam Scheduler vs Fixed Learning Rate
  2. Experiment_2_Scaling_Factor       - Impact of scaling factor in attention
  3. Experiment_3_Attention_Rollout    - Head specialization and attention analysis
  4. Experiment_4_PositionalEncoding   - Sinusoidal vs Learned embeddings
  5. Experiment_5_LabelSmoothing       - Effect of label smoothing (eps=0.1 vs 0.0)
"""

import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════════════════
#                          W&B PROJECT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class WandBConfig:
    """Central W&B configuration manager for all experiments."""
    
    # Global project settings
    PROJECT_NAME = "da6401-assignment3-transformer-nmt"
    ENTITY = None  # Set your entity/team name here, or leave None for default
    
    # Run group definitions for the 5 required experiments
    RUN_GROUPS = {
        "exp1_noam_scheduler": {
            "display_name": "Experiment 1: Noam Scheduler vs Fixed LR",
            "description": "Ablation study comparing Noam scheduler with warmup vs fixed learning rate",
            "task": "Necessity of the Noam Scheduler",
            "config_variants": {
                "noam": {
                    "scheduler_type": "noam",
                    "learning_rate": 1e-4,  # base lr (scaled by Noam)
                    "warmup_steps": 4000,
                    "d_model": 512,
                    "description": "Noam scheduler with linear warmup and inverse sqrt decay"
                },
                "fixed": {
                    "scheduler_type": "fixed",
                    "learning_rate": 1e-4,
                    "warmup_steps": 0,
                    "d_model": 512,
                    "description": "Fixed learning rate with no warmup"
                }
            }
        },
        
        "exp2_scaling_factor": {
            "display_name": "Experiment 2: Scaling Factor √(1/dk)",
            "description": "Ablation on the scaling factor in scaled dot-product attention",
            "task": "Ablation: The Scaling Factor √(1/dk)",
            "config_variants": {
                "with_scaling": {
                    "use_attention_scaling": True,
                    "scaling_factor": "sqrt_dk",
                    "description": "With 1/sqrt(dk) scaling in attention"
                },
                "without_scaling": {
                    "use_attention_scaling": False,
                    "scaling_factor": None,
                    "description": "Without scaling factor in attention"
                }
            }
        },
        
        "exp3_attention_rollout": {
            "display_name": "Experiment 3: Attention Rollout & Head Specialization",
            "description": "Analysis of attention weights and head-wise task specialization",
            "task": "Attention Rollout & Head Specialization",
            "config_variants": {
                "baseline": {
                    "num_heads": 8,
                    "d_model": 512,
                    "visualize_attention": True,
                    "description": "Standard 8-head attention with visualization"
                }
            }
        },
        
        "exp4_positional_encoding": {
            "display_name": "Experiment 4: Positional Encoding Methods",
            "description": "Comparison of sinusoidal positional encoding vs learned embeddings",
            "task": "Positional Encoding vs Learned Embeddings",
            "config_variants": {
                "sinusoidal": {
                    "positional_encoding_type": "sinusoidal",
                    "d_model": 512,
                    "description": "Sinusoidal positional encoding (paper default)"
                },
                "learned": {
                    "positional_encoding_type": "learned",
                    "d_model": 512,
                    "description": "Learned positional embeddings"
                }
            }
        },
        
        "exp5_label_smoothing": {
            "display_name": "Experiment 5: Label Smoothing Effect",
            "description": "Ablation on label smoothing regularization",
            "task": "Decoder Sensitivity: Label Smoothing",
            "config_variants": {
                "with_smoothing": {
                    "label_smoothing_eps": 0.1,
                    "description": "Label smoothing with epsilon=0.1"
                },
                "without_smoothing": {
                    "label_smoothing_eps": 0.0,
                    "description": "Standard cross-entropy (no smoothing)"
                }
            }
        }
    }
    
    # Model hyperparameters (shared across experiments)
    @staticmethod
    def get_default_model_config() -> Dict[str, Any]:
        """Returns default model hyperparameters."""
        return {
            "d_model": 512,
            "num_heads": 8,
            "d_ff": 2048,
            "num_layers": 6,
            "dropout": 0.1,
            "max_seq_len": 100,
            "vocab_size_src": 10000,  # Will be updated from dataset
            "vocab_size_tgt": 10000,  # Will be updated from dataset
        }
    
    # Training hyperparameters (shared across experiments)
    @staticmethod
    def get_default_training_config() -> Dict[str, Any]:
        """Returns default training hyperparameters."""
        return {
            "batch_size": 32,
            "num_epochs": 20,
            "learning_rate": 1e-4,
            "warmup_steps": 4000,
            "optimizer": "adam",
            "optimizer_betas": (0.9, 0.98),
            "optimizer_eps": 1e-9,
            "weight_decay": 0.0,
            "gradient_clip_value": 1.0,
            "label_smoothing_eps": 0.1,
            "device": "cuda",
            "seed": 42,
        }
    
    # Dataset configuration
    @staticmethod
    def get_dataset_config() -> Dict[str, Any]:
        """Returns dataset configuration."""
        return {
            "dataset_name": "multi30k",
            "split": "train",
            "src_lang": "de",
            "tgt_lang": "en",
            "max_src_len": 100,
            "max_tgt_len": 100,
        }
    
    # Logging and evaluation configuration
    @staticmethod
    def get_logging_config() -> Dict[str, Any]:
        """Returns logging configuration."""
        return {
            "log_frequency": 100,  # Log every N batches
            "eval_frequency": 500,  # Evaluate every N batches
            "checkpoint_frequency": 1,  # Save checkpoint every N epochs
            "compute_bleu_on_val": True,
            "compute_bleu_on_test": True,
            "max_predictions_for_bleu": 1000,  # Sample for efficiency
        }


# ═══════════════════════════════════════════════════════════════════════════════
#                           RUN GROUP MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class RunGroupManager:
    """Manages W&B run initialization for different experiment groups."""
    
    def __init__(self, project_name: str = WandBConfig.PROJECT_NAME, 
                 entity: Optional[str] = WandBConfig.ENTITY):
        self.project_name = project_name
        self.entity = entity
        self.run_group_configs = WandBConfig.RUN_GROUPS
    
    def get_run_config(self, group_key: str, variant_key: str) -> Dict[str, Any]:
        """
        Get configuration for a specific run variant.
        
        Args:
            group_key: Key from RUN_GROUPS (e.g., 'exp1_noam_scheduler')
            variant_key: Key from config_variants (e.g., 'noam', 'fixed')
        
        Returns:
            Complete configuration dictionary for the run
        """
        if group_key not in self.run_group_configs:
            raise ValueError(f"Unknown group: {group_key}")
        
        group = self.run_group_configs[group_key]
        if variant_key not in group["config_variants"]:
            raise ValueError(f"Unknown variant {variant_key} in group {group_key}")
        
        variant_config = group["config_variants"][variant_key]
        
        # Merge all configurations
        full_config = {
            "model": WandBConfig.get_default_model_config(),
            "training": WandBConfig.get_default_training_config(),
            "dataset": WandBConfig.get_dataset_config(),
            "logging": WandBConfig.get_logging_config(),
            "group_info": {
                "group_key": group_key,
                "group_name": group["display_name"],
                "task": group["task"],
                "variant_key": variant_key,
                "description": variant_config["description"],
            }
        }
        
        # Apply variant-specific overrides
        for key, value in variant_config.items():
            if key not in ["description"]:
                # Try to place in appropriate section
                if key in full_config["training"]:
                    full_config["training"][key] = value
                elif key in full_config["model"]:
                    full_config["model"][key] = value
                else:
                    # Add to group info if not found elsewhere
                    full_config["group_info"][key] = value
        
        return full_config
    
    def initialize_wandb_run(self, group_key: str, variant_key: str, 
                            run_name: Optional[str] = None,
                            additional_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Initialize a W&B run for a specific experiment variant.
        
        Args:
            group_key: Experiment group identifier
            variant_key: Variant within the group
            run_name: Optional custom run name
            additional_config: Additional config to merge
        
        Returns:
            W&B run object and config dictionary
        """
        import wandb
        
        config = self.get_run_config(group_key, variant_key)
        
        # Merge additional config if provided
        if additional_config:
            for section, values in additional_config.items():
                if section in config:
                    config[section].update(values)
                else:
                    config[section] = values
        
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{group_key}_{variant_key}_{timestamp}"
        
        # Initialize W&B run
        group_info = config.pop("group_info")
        
        run = wandb.init(
            project=self.project_name,
            entity=self.entity,
            name=run_name,
            group=group_info["group_name"],
            tags=[group_info["group_key"], group_info["variant_key"], group_info["task"]],
            config={**config["model"], **config["training"], **config["dataset"]},
            notes=group_info["description"],
        )
        
        return run, {**config, "group_info": group_info}
    
    def get_all_experiments(self) -> List[tuple]:
        """
        Get all experiment combinations as (group_key, variant_key) tuples.
        
        Returns:
            List of tuples for all experiments
        """
        experiments = []
        for group_key, group_config in self.run_group_configs.items():
            for variant_key in group_config["config_variants"].keys():
                experiments.append((group_key, variant_key))
        return experiments


# ═══════════════════════════════════════════════════════════════════════════════
#                           METRICS LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

class WandBMetricsLogger:
    """Handles all W&B logging for training metrics and visualizations."""
    
    def __init__(self, run=None):
        self.run = run
        import wandb
        self.wandb = wandb
    
    def log_training_batch(self, epoch: int, batch_idx: int, 
                          loss: float, lr: float, metrics: Optional[Dict] = None):
        """Log training batch metrics."""
        log_dict = {
            "epoch": epoch,
            "batch": batch_idx,
            "train/loss": loss,
            "train/learning_rate": lr,
        }
        
        if metrics:
            for key, value in metrics.items():
                log_dict[f"train/{key}"] = value
        
        if self.run:
            self.run.log(log_dict)
    
    def log_validation(self, epoch: int, val_loss: float, 
                      bleu_score: float, metrics: Optional[Dict] = None):
        """Log validation metrics."""
        log_dict = {
            "val/loss": val_loss,
            "val/bleu": bleu_score,
            "epoch": epoch,
        }
        
        if metrics:
            for key, value in metrics.items():
                log_dict[f"val/{key}"] = value
        
        if self.run:
            self.run.log(log_dict)
    
    def log_attention_heatmap(self, attention_weights: Any, 
                             layer: int, head: int, step: int):
        """
        Log attention heatmap for visualization.
        
        Args:
            attention_weights: Tensor of shape [seq_len, seq_len]
            layer: Encoder/decoder layer number
            head: Attention head number
            step: Training step
        """
        if self.run is None:
            return
        
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attention_weights.detach().cpu().numpy(), cmap='viridis')
        ax.set_title(f"Attention Head {head} - Layer {layer} - Step {step}")
        ax.set_xlabel("Key Position")
        ax.set_ylabel("Query Position")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        
        self.run.log({
            f"attention/layer_{layer}_head_{head}": self.wandb.Image(fig)
        })
        plt.close(fig)
    
    def log_gradient_norms(self, named_parameters, step: int):
        """Log gradient norms for analysis."""
        if self.run is None:
            return
        
        log_dict = {}
        for name, param in named_parameters:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                log_dict[f"grad_norm/{name}"] = grad_norm
        
        if log_dict:
            log_dict["step"] = step
            self.run.log(log_dict)
    
    def log_prediction_confidence(self, logits: Any, targets: Any, step: int):
        """
        Log prediction confidence (softmax probability of correct token).
        
        Args:
            logits: Model output logits
            targets: Ground truth token indices
            step: Training step
        """
        if self.run is None:
            return
        
        import torch
        
        probs = torch.softmax(logits, dim=-1)
        batch_size = targets.shape[0]
        
        confidences = []
        for i in range(batch_size):
            for j in range(targets.shape[1]):
                if targets[i, j] > 0:  # Not padding
                    conf = probs[i, j, targets[i, j]].item()
                    confidences.append(conf)
        
        if confidences:
            avg_confidence = np.mean(confidences)
            self.run.log({
                "train/prediction_confidence": avg_confidence,
                "step": step
            })
    
    def save_checkpoint_artifact(self, checkpoint_path: str, 
                                epoch: int, metric_name: str = "best"):
        """Save model checkpoint as W&B artifact."""
        if self.run is None:
            return
        
        artifact = self.wandb.Artifact(
            name=f"model-checkpoint-{metric_name}-ep{epoch}",
            type="model"
        )
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
    
    def finish(self):
        """Finish W&B run."""
        if self.run:
            self.run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
#                           UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_experiment_summary():
    """Print summary of all experiments."""
    print("\n" + "="*80)
    print("DA6401 Assignment 3 - Transformer NMT Experiments Summary")
    print("="*80 + "\n")
    
    for i, (group_key, group_config) in enumerate(WandBConfig.RUN_GROUPS.items(), 1):
        print(f"\n{i}. {group_config['display_name']}")
        print(f"   Task: {group_config['task']}")
        print(f"   Variants:")
        for var_key, var_config in group_config['config_variants'].items():
            print(f"     - {var_key}: {var_config['description']}")
    
    print("\n" + "="*80)
    print("Total Experiments:", len(RunGroupManager().get_all_experiments()))
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print experiment summary
    print_experiment_summary()
    
    # Example: Get configuration for specific variant
    manager = RunGroupManager()
    config = manager.get_run_config("exp1_noam_scheduler", "noam")
    print("\nExample Config (Noam Scheduler):")
    print(json.dumps({
        "model": config["model"],
        "training": config["training"]
    }, indent=2))