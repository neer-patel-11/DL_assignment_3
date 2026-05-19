# ════════════════════════════════════════════════════════════════════
# train.py
# ════════════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from tqdm import tqdm
import sacrebleu


# ══════════════════════════════════════════════════════════════════
# 1. LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    KL-divergence based label smoothing.
    smoothed_target = (1-eps)*one_hot + eps/(V-1)  (pad gets 0)
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [N, V]   target: [N]
        """
        V   = self.vocab_size
        eps = self.smoothing

        log_probs = F.log_softmax(logits, dim=-1)                    # [N, V]

        # Build smoothed distribution
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, eps / (V - 2))  # exclude pad & true
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            smooth_dist[:, self.pad_idx] = 0.0

        # Mask PAD positions
        non_pad = (target != self.pad_idx)
        loss = -(smooth_dist * log_probs).sum(dim=-1)                # [N]
        loss = loss[non_pad].mean()
        return loss


# ══════════════════════════════════════════════════════════════════
# 2. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = 'cpu',
) -> float:
    model.train() if is_train else model.eval()
    total_loss, total_tokens = 0.0, 0

    ctx = torch.enable_grad if is_train else torch.no_grad
    phase = 'Train' if is_train else 'Val'

    with ctx():
        pbar = tqdm(data_iter, desc=f'Epoch {epoch_num} [{phase}]', leave=False)
        for src, tgt in pbar:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: feed tgt[:-1], predict tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx=1).to(device)
            tgt_mask = make_tgt_mask(tgt_in, pad_idx=1).to(device)

            logits  = model(src, tgt_in, src_mask, tgt_mask)         # [B, T, V]
            B, T, V = logits.shape
            loss    = loss_fn(logits.reshape(B*T, V), tgt_out.reshape(B*T))

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens     = (tgt_out != 1).sum().item()
            total_loss  += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix({'loss': f'{loss.item():.3f}'})

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════
# 3. GREEDY DECODE
# ══════════════════════════════════════════════════════════════════

def greedy_decode(
    model,
    src:          torch.Tensor,
    src_mask:     torch.Tensor,
    max_len:      int,
    start_symbol: int,
    end_symbol:   int,
    device:       str = 'cpu',
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        memory = model.encode(src, src_mask)
        ys     = torch.tensor([[start_symbol]], dtype=torch.long, device=device)
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx=1).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, t, V]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys       = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break
    return ys


# ══════════════════════════════════════════════════════════════════
# 4. BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = 'cpu',
    max_len: int = 100,
) -> float:
    """
    Corpus-level BLEU (sacrebleu), range 0-100.
    tgt_vocab: object with .lookup_token(idx) OR dict itos
    """
    # from dataset import SOS_IDX, EOS_IDX, PAD_IDX

    model.eval()
    hypotheses, references = [], []

    def idx_to_str(ids):
        words = []
        for i in ids:
            if i in (SOS_IDX, EOS_IDX, PAD_IDX):
                continue
            if hasattr(tgt_vocab, 'lookup_token'):
                words.append(tgt_vocab.lookup_token(i))
            else:
                words.append(tgt_vocab.itos.get(i, '<unk>'))
        return ' '.join(words)

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc='BLEU eval', leave=False):
            src      = src.to(device)
            src_mask = make_src_mask(src, pad_idx=PAD_IDX).to(device)
            pred_ids = greedy_decode(
                model, src, src_mask, max_len, SOS_IDX, EOS_IDX, device
            )
            hypotheses.append(idx_to_str(pred_ids[0].tolist()))
            references.append(idx_to_str(tgt[0].tolist()))

    result = sacrebleu.corpus_bleu(hypotheses, [references])
    return result.score


# ══════════════════════════════════════════════════════════════════
# 5. CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, epoch: int, path: str = 'checkpoint.pt'):
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config': {
            'src_vocab_size': model.src_embed.num_embeddings,
            'tgt_vocab_size': model.tgt_embed.num_embeddings,
            'd_model':        model.d_model,
            'N':              len(model.encoder.layers),
            'num_heads':      model.encoder.layers[0].self_attn.num_heads,
            'd_ff':           model.encoder.layers[0].ffn.linear1.out_features,
            'dropout':        model.encoder.layers[0].dropout.p,
        },
    }, path)
    print(f'✅ Saved checkpoint → {path}')


def load_checkpoint(path: str, model, optimizer=None, scheduler=None) -> int:
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer  and 'optimizer_state_dict'  in ckpt: optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler  and 'scheduler_state_dict'  in ckpt: scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    print(f'Loaded checkpoint from {path} (epoch {ckpt["epoch"]})')
    return ckpt['epoch']


# ══════════════════════════════════════════════════════════════════
# 6. TRAINING EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def run_training_experiment(
    num_epochs:   int   = 30,
    batch_size:   int   = 128,
    d_model:      int   = 256,
    N:            int   = 3,
    num_heads:    int   = 8,
    d_ff:         int   = 512,
    dropout:      float = 0.1,
    warmup_steps: int   = 4000,
    smoothing:    float = 0.1,
    checkpoint_dir: str = '.',
    use_wandb:    bool  = False,
):
    import os
    # from dataset import get_dataloaders, Multi30kDataset, PAD_IDX

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')

    if use_wandb:
        import wandb
        wandb.init(project='da6401-a3', config=dict(
            d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
            dropout=dropout, warmup_steps=warmup_steps, smoothing=smoothing,
            batch_size=batch_size, num_epochs=num_epochs))

    # ── Data ──
    train_dl, val_dl, test_dl = get_dataloaders(batch_size)
    src_vocab = Multi30kDataset.src_vocab
    tgt_vocab = Multi30kDataset.tgt_vocab

    # ── Model ──
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx=PAD_IDX, smoothing=smoothing)

    best_val_loss = float('inf')
    best_ckpt     = os.path.join(checkpoint_dir, 'best_checkpoint.pt')

    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch(train_dl, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_dl,   model, loss_fn, None, None,
                               epoch_num=epoch, is_train=False, device=device)

        print(f'Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={optimizer.param_groups[0]["lr"]:.2e}')

        if use_wandb:
            import wandb
            wandb.log({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                       'lr': optimizer.param_groups[0]['lr']})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_ckpt)

        # Also save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}.pt'))

    # ── Final eval ──
    print('\nLoading best checkpoint for final BLEU evaluation...')
    load_checkpoint(best_ckpt, model)
    bleu = evaluate_bleu(model, test_dl, tgt_vocab, device=device)
    print(f'\n Test BLEU: {bleu:.2f}')

    if use_wandb:
        import wandb
        wandb.log({'test_bleu': bleu})
        wandb.finish()

    return model, bleu


# print('train.py ✅')