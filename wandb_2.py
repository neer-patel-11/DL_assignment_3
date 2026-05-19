"""
Experiment 2 — Ablation: Scaling Factor 1/√dₖ in Attention
DA6401 Assignment 3 | W&B Report Section 2.2

Trains two models:
  A) WITH    the 1/√dₖ scaling factor  (standard)
  B) WITHOUT the 1/√dₖ scaling factor  (ablated)

Also logs gradient norms of Q and K weight matrices for the
first 1 000 optimiser steps to illustrate vanishing-gradient risk.

Both runs are grouped under "exp2_scaling_factor" in W&B.
Epochs = 15.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial
from typing import Optional, Tuple
import copy
import wandb

from model import (
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    EncoderLayer,
    DecoderLayer,
    Encoder,
    Decoder,
    Transformer,
    make_src_mask,
    make_tgt_mask,
)
from train import (
    run_epoch,
    LabelSmoothingLoss,
    TranslationDataset,
    collate_fn,
    evaluate_bleu,
    save_checkpoint,
)
from lr_scheduler import NoamScheduler
from dataset import Multi30kDataset

PROJECT = "da6401-a3"
GROUP   = "exp2_scaling_factor"
PAD_IDX = 1

BASE_CONFIG = dict(
    batch_size     = 32,
    num_epochs     = 15,
    d_model        = 256,
    N              = 3,
    num_heads      = 8,
    d_ff           = 512,
    dropout        = 0.1,
    warmup_steps   = 4000,
    label_smoothing= 0.1,
    grad_log_steps = 1000,   # log grad norms for first N steps
)


# ─────────────────────────────────────────────
# PATCHED ATTENTION (no scaling)
# ─────────────────────────────────────────────

def scaled_dot_product_attention_no_scale(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention WITHOUT the 1/√dₖ denominator."""
    scores = torch.matmul(Q, K.transpose(-2, -1))   # no sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = torch.nan_to_num(attn_weights, 0.0)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class MultiHeadAttentionNoScale(MultiHeadAttention):
    """MultiHeadAttention with the √dₖ scaling removed."""

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, _ = scaled_dot_product_attention_no_scale(Q, K, V, mask)
        attn_output = self.dropout(attn_output)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)


class EncoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn   = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.feed_forward= PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.dropout     = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        return x


class DecoderLayerNoScale(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttentionNoScale(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.dropout(self.cross_attn(x, memory, memory, src_mask)))
        x = self.norm3(x + self.dropout(self.feed_forward(x)))
        return x


def build_model_no_scale(src_vocab_size, tgt_vocab_size, cfg, device):
    """Build a Transformer where every attention layer skips the √dₖ scaling."""
    d, N, H, d_ff, dr = cfg["d_model"], cfg["N"], cfg["num_heads"], cfg["d_ff"], cfg["dropout"]

    enc_layer = EncoderLayerNoScale(d, H, d_ff, dr)
    dec_layer = DecoderLayerNoScale(d, H, d_ff, dr)

    model = Transformer.__new__(Transformer)
    nn.Module.__init__(model)

    model.src_vocab_size = src_vocab_size
    model.tgt_vocab_size = tgt_vocab_size
    model.d_model        = d
    model.N              = N
    model.num_heads      = H
    model.d_ff           = d_ff
    model.dropout_rate   = dr

    model.src_embed  = nn.Embedding(src_vocab_size, d)
    model.tgt_embed  = nn.Embedding(tgt_vocab_size, d)
    model.pos_encoding = PositionalEncoding(d, dr)
    model.encoder    = Encoder(enc_layer, N)
    model.decoder    = Decoder(dec_layer, N)
    model.fc_out     = nn.Linear(d, tgt_vocab_size)

    # Bind the standard encode / decode / forward / infer methods
    import types
    model.encode  = types.MethodType(Transformer.encode,  model)
    model.decode  = types.MethodType(Transformer.decode,  model)
    model.forward = types.MethodType(Transformer.forward, model)

    return model.to(device)


# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
def load_data(cfg):
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")
    val_ds.src_vocab  = val_ds.tgt_vocab  = None
    val_ds.src_vocab  = train_ds.src_vocab
    val_ds.tgt_vocab  = train_ds.tgt_vocab
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab

    train_src, train_tgt = train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    _col = partial(collate_fn, pad_idx=PAD_IDX)
    train_loader = DataLoader(TranslationDataset(train_src, train_ds.tgt_data),
                              batch_size=cfg["batch_size"], shuffle=True, collate_fn=_col)
    val_loader   = DataLoader(TranslationDataset(val_ds.src_data,  val_ds.tgt_data),
                              batch_size=cfg["batch_size"], shuffle=False, collate_fn=_col)
    test_loader  = DataLoader(TranslationDataset(test_ds.src_data, test_ds.tgt_data),
                              batch_size=cfg["batch_size"], shuffle=False, collate_fn=_col)
    return train_loader, val_loader, test_loader, train_ds


# ─────────────────────────────────────────────
# TRAINING WITH GRAD-NORM LOGGING
# ─────────────────────────────────────────────
def train_run_with_grad_logging(run_name, model, cfg, train_loader, val_loader,
                                test_loader, train_ds, device):
    wandb.init(project=PROJECT, group=GROUP, name=run_name,
               config={**cfg, "run_name": run_name}, reinit=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    loss_fn   = LabelSmoothingLoss(len(train_ds.tgt_vocab), PAD_IDX, cfg["label_smoothing"])

    global_step   = 0
    log_grad_until = cfg["grad_log_steps"]
    best_val_loss  = float("inf")

    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()
        for src, tgt in train_loader:
            src, tgt = src.to(device), tgt.to(device)
            src_mask = make_src_mask(src).to(device)
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask   = make_tgt_mask(tgt_input).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss   = loss_fn(logits.reshape(-1, model.tgt_vocab_size), tgt_output.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            # Log Q/K grad norms for first N steps
            if global_step < log_grad_until:
                q_norms, k_norms = [], []
                for module in model.modules():
                    if isinstance(module, (MultiHeadAttention, MultiHeadAttentionNoScale)):
                        if module.W_q.weight.grad is not None:
                            q_norms.append(module.W_q.weight.grad.norm().item())
                        if module.W_k.weight.grad is not None:
                            k_norms.append(module.W_k.weight.grad.norm().item())
                if q_norms:
                    wandb.log({
                        "step": global_step,
                        "grad_norm_Q_mean": sum(q_norms) / len(q_norms),
                        "grad_norm_K_mean": sum(k_norms) / len(k_norms),
                    })

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1

        # Epoch-level validation
        val_loss = run_epoch(val_loader, model, loss_fn, None, None,
                             epoch_num=epoch, is_train=False, device=device)
        # Compute train loss for logging (re-use run_epoch in eval mode for simplicity)
        train_loss_log = run_epoch(train_loader, model, loss_fn, None, None,
                                   epoch_num=epoch, is_train=False, device=device)
        print(f"[{run_name}] Epoch {epoch} | Val Loss {val_loss:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss_log, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, f"best_{run_name}.pt")

    test_bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    wandb.log({"test_bleu": test_bleu})
    print(f"[{run_name}] Test BLEU: {test_bleu:.2f}")
    wandb.finish()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_ds = load_data(BASE_CONFIG)
    src_vs = len(train_ds.src_vocab)
    tgt_vs = len(train_ds.tgt_vocab)

    # Run A — WITH scaling (standard Transformer from model.py)
    model_with = Transformer(
        src_vocab_size=src_vs, tgt_vocab_size=tgt_vs,
        d_model=BASE_CONFIG["d_model"], N=BASE_CONFIG["N"],
        num_heads=BASE_CONFIG["num_heads"], d_ff=BASE_CONFIG["d_ff"],
        dropout=BASE_CONFIG["dropout"],
    ).to(device)
    train_run_with_grad_logging(
        "with_scaling", model_with, BASE_CONFIG,
        train_loader, val_loader, test_loader, train_ds, device,
    )

    # Run B — WITHOUT scaling
    model_no = build_model_no_scale(src_vs, tgt_vs, BASE_CONFIG, device)
    train_run_with_grad_logging(
        "without_scaling", model_no, BASE_CONFIG,
        train_loader, val_loader, test_loader, train_ds, device,
    )