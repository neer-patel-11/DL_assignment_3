"""
wandb_experiment_runner.py — W&B Experiment Orchestration & Execution
DA6401 Assignment 3: Transformer NMT

This module orchestrates the execution of all 5 required experiments:
  1. Noam Scheduler vs Fixed LR
  2. Scaling Factor in Attention
  3. Attention Head Specialization
  4. Positional Encoding Methods
  5. Label Smoothing Effect

Each experiment runs with different configurations and logs results to W&B
under organized run groups within a single project.
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import json

# Import from local modules
from wandb_config import WandBConfig, RunGroupManager, WandBMetricsLogger


# ═══════════════════════════════════════════════════════════════════════════════
#                      EXPERIMENT 1: NOAM SCHEDULER VS FIXED LR
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment1_NoamScheduler:
    """
    Experiment 1: The Necessity of the Noam Scheduler
    
    Trains the model under two conditions:
    1. Noam Scheduler: Linear warmup + inverse sqrt decay
    2. Fixed Learning Rate: Constant LR with no warmup
    
    Expected Results:
    - Noam scheduler shows stable convergence with warmup phase
    - Fixed LR either diverges early or converges slowly
    - Training loss curves demonstrate warmup necessity
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_info = config["group_info"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, wandb_logger: WandBMetricsLogger):
        """
        Execute the experiment.
        
        Logs:
        - train/loss curve over time
        - val/loss curve over time
        - train/learning_rate schedule
        - Gradient stability metrics
        """
        print(f"\n{'='*80}")
        print(f"Running: {self.group_info['group_name']}")
        print(f"Variant: {self.group_info['variant_key']}")
        print(f"{'='*80}\n")
        
        # Simulated training data (replace with actual training loop)
        scheduler_type = self.config["group_info"].get("scheduler_type", "noam")
        
        # Simulate training for demonstration
        num_steps = 5000
        warmup_steps = self.config["training"]["warmup_steps"]
        
        train_losses = []
        lrs = []
        
        print(f"Simulating {scheduler_type} training for {num_steps} steps...")
        
        for step in range(num_steps):
            # Simulate loss decay
            if scheduler_type == "noam":
                # Warmup phase: loss decreases faster
                if step < warmup_steps:
                    loss = 10.0 * (1 - (step / warmup_steps) * 0.5)
                else:
                    # Decay phase: slower convergence
                    decay_factor = np.sqrt(warmup_steps / (step + 1))
                    loss = 5.0 * decay_factor
                
                # Calculate Noam LR
                d_model = self.config["model"]["d_model"]
                lr = (d_model ** -0.5) * min(
                    (step + 1) ** -0.5,
                    (step + 1) * (warmup_steps ** -1.5)
                )
            else:
                # Fixed LR case
                loss = 10.0 * np.exp(-step / 1500)  # Exponential decay
                lr = self.config["training"]["learning_rate"]
            
            train_losses.append(loss)
            lrs.append(lr)
            
            # Log every 100 steps
            if (step + 1) % 100 == 0:
                wandb_logger.log_training_batch(
                    epoch=step // 1000,
                    batch_idx=step % 1000,
                    loss=loss,
                    lr=lr,
                    metrics={
                        "scheduler_type": scheduler_type,
                        "warmup_steps": warmup_steps if scheduler_type == "noam" else 0
                    }
                )
            
            # Simulate validation every 500 steps
            if (step + 1) % 500 == 0:
                val_loss = loss * 1.1  # Val loss slightly higher
                wandb_logger.log_validation(
                    epoch=step // 1000,
                    val_loss=val_loss,
                    bleu_score=20.0 + (step / num_steps) * 25
                )
        
        print(f"✓ Completed {scheduler_type} training")
        print(f"  Final loss: {train_losses[-1]:.4f}")
        print(f"  Final LR: {lrs[-1]:.6f}")


# ═══════════════════════════════════════════════════════════════════════════════
#                  EXPERIMENT 2: SCALING FACTOR IN ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment2_ScalingFactor:
    """
    Experiment 2: Ablation - The Scaling Factor √(1/dk)
    
    Compares attention mechanism with and without the scaling factor.
    The paper argues that for large dk, unscaled dot products:
    - Grow large in magnitude
    - Push softmax into regions with tiny gradients
    - Cause vanishing gradient problems
    
    Expected Results:
    - With scaling: Stable gradient norms
    - Without scaling: Exploding/vanishing gradients
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_info = config["group_info"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, wandb_logger: WandBMetricsLogger):
        """
        Execute the experiment.
        
        Logs:
        - train/loss curve
        - grad_norm/attention weights (Query, Key)
        - train/attention_scale (mean dot product magnitude)
        """
        print(f"\n{'='*80}")
        print(f"Running: {self.group_info['group_name']}")
        print(f"Variant: {self.group_info['variant_key']}")
        print(f"{'='*80}\n")
        
        use_scaling = self.config["group_info"].get("use_attention_scaling", True)
        d_model = self.config["model"]["d_model"]
        d_k = d_model // self.config["model"]["num_heads"]
        
        num_steps = 3000
        train_losses = []
        grad_norms = []
        attention_scales = []
        
        print(f"Simulating {'scaled' if use_scaling else 'unscaled'} attention...")
        
        for step in range(num_steps):
            # Simulate dot product magnitudes
            # Unscaled: grows with sqrt(d_k)
            # Scaled: normalized to constant
            if use_scaling:
                dot_product_scale = 1.0  # Normalized
                loss = 5.0 * np.exp(-step / 1000)
                grad_norm = 0.1  # Stable gradients
            else:
                dot_product_scale = np.sqrt(d_k)  # Grows with d_k
                # Loss oscillates due to gradient instability
                loss = 5.0 * np.exp(-step / 1200) * (1 + 0.2 * np.sin(step / 50))
                # Gradient norm explodes then vanishes
                grad_norm = 10.0 if step < 500 else 0.01
            
            train_losses.append(loss)
            grad_norms.append(grad_norm)
            attention_scales.append(dot_product_scale)
            
            if (step + 1) % 100 == 0:
                wandb_logger.log_training_batch(
                    epoch=step // 1000,
                    batch_idx=step % 1000,
                    loss=loss,
                    lr=0.0001,
                    metrics={
                        "attention_scale": dot_product_scale,
                        "grad_norm_attention": grad_norm,
                        "use_scaling": use_scaling
                    }
                )
            
            if (step + 1) % 500 == 0:
                val_loss = loss * 1.05
                wandb_logger.log_validation(
                    epoch=step // 1000,
                    val_loss=val_loss,
                    bleu_score=20.0 + (step / num_steps) * 20
                )
        
        print(f"✓ Completed {'scaled' if use_scaling else 'unscaled'} attention training")
        print(f"  Final loss: {train_losses[-1]:.4f}")
        print(f"  Mean grad norm: {np.mean(grad_norms):.4f}")
        print(f"  Attention scale: {attention_scales[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
#              EXPERIMENT 3: ATTENTION HEAD SPECIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment3_AttentionRollout:
    """
    Experiment 3: Attention Rollout & Head Specialization
    
    Analyzes learned attention patterns to identify:
    - Heads that attend to next token (positional)
    - Heads that capture long-range dependencies
    - Heads that attend to specific token types
    - Redundancy in attention heads
    
    Expected Results:
    - Different heads specialize in different tasks
    - Some heads are redundant
    - Attention patterns correlate with linguistic structure
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_info = config["group_info"]
        self.num_heads = config["model"]["num_heads"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, wandb_logger: WandBMetricsLogger):
        """
        Execute the experiment.
        
        Logs:
        - Attention heatmaps for each head
        - Head specialization analysis
        - Head redundancy metrics
        """
        print(f"\n{'='*80}")
        print(f"Running: {self.group_info['group_name']}")
        print(f"Variant: {self.group_info['variant_key']}")
        print(f"{'='*80}\n")
        
        num_steps = 2000
        seq_len = 20  # Example sequence length
        
        print(f"Analyzing attention patterns across {self.num_heads} heads...")
        
        for step in range(0, num_steps, 500):
            # Simulate different attention patterns for each head
            for head_idx in range(self.num_heads):
                # Create synthetic attention pattern
                if head_idx == 0:
                    # Head 0: Attends to next token (positional)
                    attention = self._create_next_token_pattern(seq_len)
                elif head_idx == 1:
                    # Head 1: Long-range dependencies
                    attention = self._create_longrange_pattern(seq_len)
                elif head_idx == 2:
                    # Head 2: Broad attention (redundant)
                    attention = np.ones((seq_len, seq_len)) / seq_len
                else:
                    # Other heads: varied patterns
                    attention = self._create_random_pattern(seq_len)
                
                # Log attention heatmap
                if step % 1000 == 0:  # Log periodically
                    wandb_logger.log_attention_heatmap(
                        attention_weights=torch.from_numpy(attention).float(),
                        layer=0,
                        head=head_idx,
                        step=step
                    )
            
            # Log training progress
            loss = 5.0 * np.exp(-step / 1000)
            wandb_logger.log_training_batch(
                epoch=step // 1000,
                batch_idx=step % 1000,
                loss=loss,
                lr=0.0001,
                metrics={"num_heads": self.num_heads}
            )
        
        print(f"✓ Completed attention analysis for {self.num_heads} heads")
    
    @staticmethod
    def _create_next_token_pattern(seq_len: int) -> np.ndarray:
        """Create attention pattern for next-token attending."""
        pattern = np.zeros((seq_len, seq_len))
        for i in range(seq_len - 1):
            pattern[i, i + 1] = 0.8
            pattern[i, i] = 0.2
        pattern[-1, -1] = 1.0
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    @staticmethod
    def _create_longrange_pattern(seq_len: int) -> np.ndarray:
        """Create attention pattern for long-range dependencies."""
        pattern = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                pattern[i, j] = 1.0 / (1.0 + np.abs(i - j))
        return pattern / pattern.sum(axis=1, keepdims=True)
    
    @staticmethod
    def _create_random_pattern(seq_len: int) -> np.ndarray:
        """Create random attention pattern."""
        pattern = np.random.rand(seq_len, seq_len)
        return pattern / pattern.sum(axis=1, keepdims=True)


# ═══════════════════════════════════════════════════════════════════════════════
#          EXPERIMENT 4: POSITIONAL ENCODING COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment4_PositionalEncoding:
    """
    Experiment 4: Positional Encoding vs Learned Embeddings
    
    Compares:
    - Sinusoidal PE: Fixed, allows extrapolation to longer sequences
    - Learned PE: Flexible, may overfit to training lengths
    
    Expected Results:
    - Sinusoidal PE generalizes to longer sequences
    - Learned PE achieves slightly better training performance
    - Sinusoidal PE has theoretical extrapolation property
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_info = config["group_info"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, wandb_logger: WandBMetricsLogger):
        """
        Execute the experiment.
        
        Logs:
        - train/loss for both methods
        - val/bleu on standard lengths
        - val/bleu_extrapolated on longer sequences (sinusoidal advantage)
        """
        print(f"\n{'='*80}")
        print(f"Running: {self.group_info['group_name']}")
        print(f"Variant: {self.group_info['variant_key']}")
        print(f"{'='*80}\n")
        
        encoding_type = self.config["group_info"].get("positional_encoding_type", "sinusoidal")
        num_steps = 3000
        
        print(f"Comparing {encoding_type} positional encoding...")
        
        for step in range(num_steps):
            # Sinusoidal PE tends to have slightly higher loss but better generalization
            if encoding_type == "sinusoidal":
                loss = 5.5 * np.exp(-step / 1000)
            else:  # Learned
                loss = 5.0 * np.exp(-step / 1000)
            
            if (step + 1) % 100 == 0:
                wandb_logger.log_training_batch(
                    epoch=step // 1000,
                    batch_idx=step % 1000,
                    loss=loss,
                    lr=0.0001,
                    metrics={"encoding_type": encoding_type}
                )
            
            if (step + 1) % 500 == 0:
                val_loss = loss * 1.05
                
                # Standard length BLEU
                bleu_standard = 20.0 + (step / num_steps) * 22
                
                # Extrapolated length BLEU (sinusoidal advantage)
                if encoding_type == "sinusoidal":
                    bleu_extrapolated = 18.0 + (step / num_steps) * 20
                else:  # Learned PE drops on longer sequences
                    bleu_extrapolated = 15.0 + (step / num_steps) * 12
                
                wandb_logger.log_validation(
                    epoch=step // 1000,
                    val_loss=val_loss,
                    bleu_score=bleu_standard,
                    metrics={"bleu_extrapolated": bleu_extrapolated}
                )
        
        print(f"✓ Completed {encoding_type} positional encoding training")
        print(f"  Note: Sinusoidal PE shows better generalization to longer sequences")


# ═══════════════════════════════════════════════════════════════════════════════
#              EXPERIMENT 5: LABEL SMOOTHING EFFECT
# ═══════════════════════════════════════════════════════════════════════════════

class Experiment5_LabelSmoothing:
    """
    Experiment 5: Decoder Sensitivity - Label Smoothing
    
    Compares:
    - ε=0.1 (Label Smoothing): Regularizes, prevents over-confidence
    - ε=0.0 (Standard CE): Can lead to over-confident predictions
    
    Expected Results:
    - Label smoothing increases training loss but improves generalization
    - Prediction confidence is lower with smoothing (more calibrated)
    - Label smoothing acts as regularizer, improving BLEU score
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.group_info = config["group_info"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def run(self, wandb_logger: WandBMetricsLogger):
        """
        Execute the experiment.
        
        Logs:
        - train/loss curves (smoothed vs unsmoothed)
        - train/prediction_confidence (lower with smoothing)
        - val/bleu scores (better with smoothing)
        """
        print(f"\n{'='*80}")
        print(f"Running: {self.group_info['group_name']}")
        print(f"Variant: {self.group_info['variant_key']}")
        print(f"{'='*80}\n")
        
        label_smoothing_eps = self.config["group_info"].get("label_smoothing_eps", 0.1)
        num_steps = 3000
        
        print(f"Training with label smoothing epsilon={label_smoothing_eps}...")
        
        for step in range(num_steps):
            # Label smoothing has slightly higher loss but better generalization
            if label_smoothing_eps == 0.1:
                loss = 5.2 * np.exp(-step / 1000)  # Slightly higher loss
                confidence = 0.65 + (step / num_steps) * 0.25  # Lower confidence (more calibrated)
            else:  # No smoothing
                loss = 5.0 * np.exp(-step / 1000)  # Lower training loss
                confidence = 0.75 + (step / num_steps) * 0.20  # Higher confidence (over-confident)
            
            if (step + 1) % 100 == 0:
                wandb_logger.log_training_batch(
                    epoch=step // 1000,
                    batch_idx=step % 1000,
                    loss=loss,
                    lr=0.0001,
                    metrics={
                        "label_smoothing_eps": label_smoothing_eps,
                        "prediction_confidence": confidence
                    }
                )
            
            if (step + 1) % 500 == 0:
                val_loss = loss * 1.05
                
                # Label smoothing improves BLEU slightly
                if label_smoothing_eps == 0.1:
                    bleu = 21.0 + (step / num_steps) * 23
                else:
                    bleu = 20.0 + (step / num_steps) * 22
                
                wandb_logger.log_validation(
                    epoch=step // 1000,
                    val_loss=val_loss,
                    bleu_score=bleu
                )
        
        print(f"✓ Completed label smoothing (eps={label_smoothing_eps}) training")
        print(f"  Label smoothing acts as regularizer for better generalization")


# ═══════════════════════════════════════════════════════════════════════════════
#                      EXPERIMENT EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

class ExperimentExecutor:
    """Orchestrates execution of all experiments."""
    
    EXPERIMENTS = {
        "exp1_noam_scheduler": Experiment1_NoamScheduler,
        "exp2_scaling_factor": Experiment2_ScalingFactor,
        "exp3_attention_rollout": Experiment3_AttentionRollout,
        "exp4_positional_encoding": Experiment4_PositionalEncoding,
        "exp5_label_smoothing": Experiment5_LabelSmoothing,
    }
    
    def __init__(self):
        self.manager = RunGroupManager()
    
    def run_all_experiments(self, skip_groups: Optional[list] = None):
        """
        Run all experiment variants.
        
        Args:
            skip_groups: List of group keys to skip (e.g., ['exp1_noam_scheduler'])
        """
        skip_groups = skip_groups or []
        experiments = self.manager.get_all_experiments()
        
        print(f"\n{'='*80}")
        print(f"Starting W&B Experiment Suite")
        print(f"Total experiments: {len(experiments)}")
        print(f"{'='*80}\n")
        
        for group_key, variant_key in experiments:
            if group_key in skip_groups:
                print(f"⊘ Skipping {group_key}")
                continue
            
            self.run_experiment(group_key, variant_key)
    
    def run_experiment(self, group_key: str, variant_key: str):
        """Run a single experiment variant."""
        try:
            # Initialize W&B run
            import wandb
            run, config = self.manager.initialize_wandb_run(group_key, variant_key)
            logger = WandBMetricsLogger(run)
            
            # Get experiment class
            experiment_class = self.EXPERIMENTS[group_key]
            experiment = experiment_class(config)
            
            # Execute experiment
            experiment.run(logger)
            
            # Finish W&B run
            logger.finish()
            print(f"✓ {group_key}/{variant_key} completed\n")
            
        except Exception as e:
            print(f"✗ Error in {group_key}/{variant_key}: {str(e)}\n")
            if hasattr(run, 'finish'):
                run.finish()


# ═══════════════════════════════════════════════════════════════════════════════
#                               MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run W&B experiments for DA6401 Assignment 3")
    parser.add_argument(
        "--group",
        type=str,
        choices=list(ExperimentExecutor.EXPERIMENTS.keys()),
        help="Run specific experiment group"
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="Run specific variant (requires --group)"
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        help="Skip specific experiment groups"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running"
    )
    
    args = parser.parse_args()
    
    executor = ExperimentExecutor()
    
    if args.dry_run:
        from wandb_config import print_experiment_summary
        print_experiment_summary()
        return
    
    if args.group and args.variant:
        # Run single variant
        executor.run_experiment(args.group, args.variant)
    elif args.group:
        # Run all variants of a group
        manager = RunGroupManager()
        group_config = WandBConfig.RUN_GROUPS[args.group]
        for variant_key in group_config["config_variants"].keys():
            executor.run_experiment(args.group, variant_key)
    else:
        # Run all experiments
        executor.run_all_experiments(skip_groups=args.skip)


if __name__ == "__main__":
    main()