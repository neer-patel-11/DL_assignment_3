"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Convert logits to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Create smoothed target distribution
        # Start with uniform distribution over all classes
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))  # Smooth
            
            # Set true class with confidence
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Zero out padding positions
            true_dist[:, self.pad_idx] = 0
            
            # Renormalize to ensure it sums to 1
            # mask = torch.nonzero(target == self.pad_idx)
            # if mask.numel() > 0:
            #     true_dist[mask] = 0
            # Renormalize where target is NOT padding
            mask = (target != self.pad_idx).unsqueeze(1)
            true_dist = true_dist * mask
        
        # Compute KL divergence (cross entropy with smoothed labels)
        loss = torch.sum(-true_dist * log_probs, dim=-1)
        
        # Mask out padding tokens
        mask = (target != self.pad_idx).float()
        loss = (loss * mask).sum() / mask.sum()
        
        return loss


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION DATASET  
# ══════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """PyTorch Dataset for translation pairs."""
    
    def __init__(self, src_data, tgt_data):
        """
        Args:
            src_data: List of source token index lists
            tgt_data: List of target token index lists
        """
        self.src_data = src_data
        self.tgt_data = tgt_data
    
    def __len__(self):
        return len(self.src_data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.src_data[idx], dtype=torch.long), \
               torch.tensor(self.tgt_data[idx], dtype=torch.long)


def collate_fn(batch, pad_idx=1, max_len=100):
    """
    Collate function to pad sequences to the same length in a batch.
    
    Args:
        batch: List of (src, tgt) tensors
        pad_idx: Index of padding token
        max_len: Maximum sequence length
    
    Returns:
        Padded src and tgt tensors
    """
    src_batch, tgt_batch = zip(*batch)
    
    # Pad sequences
    src_padded = nn.utils.rnn.pad_sequence(
        [s[:max_len] for s in src_batch],
        batch_first=True,
        padding_value=pad_idx
    )
    tgt_padded = nn.utils.rnn.pad_sequence(
        [t[:max_len] for t in tgt_batch],
        batch_first=True,
        padding_value=pad_idx
    )
    
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    model.train() if is_train else model.eval()
    
    total_loss = 0
    total_tokens = 0
    
    pbar = tqdm(data_iter, disable=False)
    
    for src, tgt in pbar:
        src = src.to(device)
        tgt = tgt.to(device)
        
        # Create masks
        src_mask = make_src_mask(src, pad_idx=1)
        tgt_mask = make_tgt_mask(tgt, pad_idx=1)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        # Target input is everything except the last token
        tgt_input = tgt[:, :-1]
        # Target output is everything except the first token (for loss computation)
        tgt_output = tgt[:, 1:]
        
        # Create adjusted masks for the decoder input
        src_mask_decoder = src_mask
        tgt_mask_decoder = make_tgt_mask(tgt_input, pad_idx=1).to(device)
        
        # Forward pass
        with torch.set_grad_enabled(is_train):
            logits = model(src, tgt_input, src_mask_decoder, tgt_mask_decoder)
            
            # Reshape for loss computation
            logits_flat = logits.reshape(-1, model.tgt_vocab_size)
            tgt_flat = tgt_output.reshape(-1)
            
            # Compute loss
            loss = loss_fn(logits_flat, tgt_flat)
        
        if is_train:
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        
        # Accumulate loss
        num_tokens = (tgt_output != 1).sum().item()  # Count non-padding tokens
        total_loss += loss.item() * num_tokens
        total_tokens += num_tokens
        
        pbar.set_description(
            f"Epoch {epoch_num} | {'Train' if is_train else 'Val'} | "
            f"Loss: {loss.item():.4f}"
        )
    
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = 3,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    model.eval()
    
    with torch.no_grad():
        # Encode source
        memory = model.encode(src, src_mask)
        
        # Initialize target with start symbol
        ys = torch.ones(1, 1, dtype=torch.long, device=device) * start_symbol
        
        for _ in range(max_len - 1):
            # Create target mask
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            
            # Decode
            logits = model.decode(memory, src_mask, ys, tgt_mask)
            
            # Get the next token (greedy)
            next_token = logits[0, -1, :].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            
            # Append to sequence
            ys = torch.cat([ys, next_token], dim=1)
            
            # Stop if end symbol
            if next_token.item() == end_symbol:
                break
    
    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════
def calculate_bleu_score(reference, hypothesis, max_n=4):
    from collections import Counter
    import math

    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    weights = [0.25, 0.25, 0.25, 0.25]
    log_score = 0.0

    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n:
            return 0.0  # can't compute this n-gram, BLEU is 0

        ref_ngrams = Counter()
        hyp_ngrams = Counter()

        for i in range(len(ref_tokens) - n + 1):
            ref_ngrams[tuple(ref_tokens[i:i+n])] += 1

        for i in range(len(hyp_tokens) - n + 1):
            hyp_ngrams[tuple(hyp_tokens[i:i+n])] += 1

        matches = sum(min(count, ref_ngrams.get(ngram, 0))
                      for ngram, count in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())

        if matches == 0:
            return 0.0  # geometric mean collapses to 0

        log_score += weights[n-1] * math.log(matches / total)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens)))

    return bp * math.exp(log_score) * 100



def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    model.eval()
    
    total_bleu = 0
    count = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="Evaluating BLEU"):
            src = src.to(device)
            tgt = tgt.to(device)
            
            for i in range(src.shape[0]):
                src_seq = src[i:i+1]
                tgt_seq = tgt[i:i+1]
                
                # Create mask
                src_mask = make_src_mask(src_seq, pad_idx=1).to(device)
                
                # Decode
                ys = greedy_decode(
                    model, src_seq, src_mask, max_len,
                    start_symbol=2,  # <sos>
                    end_symbol=3,    # <eos>
                    device=device
                )
                
                # Convert to text
                hyp_tokens = []
                for idx in ys[0].cpu().numpy():
                    if idx in tgt_vocab.itos:
                        token = tgt_vocab.itos[idx]
                        if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                            hyp_tokens.append(token)
                hypothesis = ' '.join(hyp_tokens)
                
                ref_tokens = []
                for idx in tgt_seq[0].cpu().numpy():
                    if idx in tgt_vocab.itos:
                        token = tgt_vocab.itos[idx]
                        if token not in ['<sos>', '<eos>', '<pad>', '<unk>']:
                            ref_tokens.append(token)
                reference = ' '.join(ref_tokens)
                
                # Calculate BLEU for this pair
                bleu = calculate_bleu_score(reference, hypothesis)
                total_bleu += bleu
                count += 1
    
    corpus_bleu = total_bleu / max(count, 1)
    return corpus_bleu


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_vocab_size,
            'tgt_vocab_size': model.tgt_vocab_size,
            'd_model': model.d_model,
            'N': model.N,
            'num_heads': model.num_heads,
            'd_ff': model.d_ff,
            'dropout': model.dropout_rate,
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    checkpoint = torch.load(path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path}, epoch {epoch}")
    
    return epoch


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # Configuration
    config = {
        'batch_size': 16,
        'num_epochs': 40,
        'd_model': 512,
        'N': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'warmup_steps': 4000,
        'label_smoothing': 0.1,
    
    }
    
    # Initialize W&B
    # wandb.init(project="da6401-a3", config=config)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load datasets
    print("Loading datasets...")

    train_dataset = Multi30kDataset(split='train')

    val_dataset = Multi30kDataset(split='validation')
    test_dataset = Multi30kDataset(split='test')

    # SHARE VOCAB
    val_dataset.src_vocab = train_dataset.src_vocab
    val_dataset.tgt_vocab = train_dataset.tgt_vocab

    test_dataset.src_vocab = train_dataset.src_vocab
    test_dataset.tgt_vocab = train_dataset.tgt_vocab
    
    # Process data
    print("Processing training data...")
    train_src, train_tgt = train_dataset.process_data()
    
    print("Processing validation data...")
    val_dataset.process_data()
    val_src, val_tgt = val_dataset.src_data, val_dataset.tgt_data
    
    print("Processing test data...")
    test_dataset.process_data()
    test_src, test_tgt = test_dataset.src_data, test_dataset.tgt_data
    
    # Create datasets and dataloaders
    train_dataset_torch = TranslationDataset(train_src, train_tgt)
    val_dataset_torch = TranslationDataset(val_src, val_tgt)
    test_dataset_torch = TranslationDataset(test_src, test_tgt)
    
    pad_idx = 1
    train_loader = DataLoader(
        train_dataset_torch,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx)
    )
    val_loader = DataLoader(
        val_dataset_torch,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx)
    )
    test_loader = DataLoader(
        test_dataset_torch,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_idx=pad_idx)
    )
    
    # Build model
    print("Building model...")
    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=config['d_model'],
        N=config['N'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        
    ).to(device)
    
    # Optimizer with specified hyperparameters
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Will be scaled by NoamScheduler
        betas=(0.9, 0.98),
        eps=1e-9
    )
    
    # Scheduler
    scheduler = NoamScheduler(
        optimizer,
        d_model=config['d_model'],
        warmup_steps=config['warmup_steps']
    )
    
    # Loss function
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=pad_idx,
        smoothing=config['label_smoothing']
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Training
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch+1, is_train=True, device=device
        )
        
        # Validation
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch+1, is_train=False, device=device
        )
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log to W&B
        # wandb.log({
        #     'epoch': epoch+1,
        #     'train_loss': train_loss,
        #     'val_loss': val_loss,
        # })
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch+1, f'checkpoint_epoch_{epoch+1}.pt')
        
        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch+1, 'best_checkpoint.pt')
    
    # Final BLEU evaluation on test set
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60)
    
    test_bleu = evaluate_bleu(
        model, test_loader, train_dataset.tgt_vocab,
        device=device, max_len=100
    )
    
    print(f"\nTest BLEU Score: {test_bleu:.4f}")
    
    # wandb.log({'test_bleu': test_bleu})
    # wandb.finish()


if __name__ == "__main__":
    run_training_experiment()