"""
Experiment 1 — Noam Scheduler vs. Fixed Learning Rate
DA6401 Assignment 3 | W&B Report Section 2.1

Trains the Transformer under two conditions:
  A) Noam Scheduler  (linear warm-up + inverse-sqrt decay)
  B) Fixed LR = 1e-4

Both runs are grouped under "exp1_lr_schedule" in W&B.
Epochs = 15 for each run.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
import wandb

from model import Transformer
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

# ─────────────────────────────────────────────
# SHARED CONFIG
# ─────────────────────────────────────────────
BASE_CONFIG = dict(
    batch_size    = 32,
    num_epochs    = 15,
    d_model       = 256,
    N             = 3,
    num_heads     = 8,
    d_ff          = 512,
    dropout       = 0.1,
    warmup_steps  = 4000,
    label_smoothing = 0.1,
)

PROJECT   = "da6401-a3"
GROUP     = "exp1_lr_schedule"
PAD_IDX   = 1


# ─────────────────────────────────────────────
# DATA LOADING (done once, shared across runs)
# ─────────────────────────────────────────────
def load_data(cfg):
    print("Loading datasets …")
    train_ds = Multi30kDataset(split="train")
    val_ds   = Multi30kDataset(split="validation")
    test_ds  = Multi30kDataset(split="test")

    # Share vocab built from training split
    val_ds.src_vocab  = train_ds.src_vocab
    val_ds.tgt_vocab  = train_ds.tgt_vocab
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab

    train_src, train_tgt = train_ds.process_data()
    val_ds.process_data()
    test_ds.process_data()

    _collate = partial(collate_fn, pad_idx=PAD_IDX)

    train_loader = DataLoader(
        TranslationDataset(train_src, train_ds.tgt_data),
        batch_size=cfg["batch_size"], shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        TranslationDataset(val_ds.src_data, val_ds.tgt_data),
        batch_size=cfg["batch_size"], shuffle=False, collate_fn=_collate,
    )
    test_loader = DataLoader(
        TranslationDataset(test_ds.src_data, test_ds.tgt_data),
        batch_size=cfg["batch_size"], shuffle=False, collate_fn=_collate,
    )
    return train_loader, val_loader, test_loader, train_ds


# ─────────────────────────────────────────────
# SINGLE TRAINING RUN
# ─────────────────────────────────────────────
def train_run(run_name, use_noam, cfg, train_loader, val_loader, test_loader, train_ds, device):
    wandb.init(
        project = PROJECT,
        group   = GROUP,
        name    = run_name,
        config  = {**cfg, "use_noam": use_noam},
        reinit  = True,
    )

    model = Transformer(
        src_vocab_size = len(train_ds.src_vocab),
        tgt_vocab_size = len(train_ds.tgt_vocab),
        d_model  = cfg["d_model"],
        N        = cfg["N"],
        num_heads= cfg["num_heads"],
        d_ff     = cfg["d_ff"],
        dropout  = cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    else:
        # Fixed LR — use a constant scheduler that keeps lr=1e-4 always
        for pg in optimizer.param_groups:
            pg["lr"] = 1e-4
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    loss_fn = LabelSmoothingLoss(
        vocab_size = len(train_ds.tgt_vocab),
        pad_idx    = PAD_IDX,
        smoothing  = cfg["label_smoothing"],
    )

    best_val_loss = float("inf")
    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{run_name}] Epoch {epoch} | Train {train_loss:.4f} | Val {val_loss:.4f} | LR {current_lr:.2e}")
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, f"best_{run_name}.pt")

    test_bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"[{run_name}] Test BLEU: {test_bleu:.2f}")
    wandb.log({"test_bleu": test_bleu})
    wandb.finish()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader, train_ds = load_data(BASE_CONFIG)

    # Run A — Noam Scheduler
    train_run("noam_scheduler", use_noam=True,
              cfg=BASE_CONFIG, train_loader=train_loader,
              val_loader=val_loader, test_loader=test_loader,
              train_ds=train_ds, device=device)

    # Run B — Fixed LR 1e-4
    train_run("fixed_lr_1e4", use_noam=False,
              cfg=BASE_CONFIG, train_loader=train_loader,
              val_loader=val_loader, test_loader=test_loader,
              train_ds=train_ds, device=device)