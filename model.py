# ════════════════════════════════════════════════════════════════════
# model.py  — Full Transformer + infer()
# ════════════════════════════════════════════════════════════════════
import math, copy, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) · V
    Q, K: (..., seq, d_k)   V: (..., seq, d_v)
    mask (bool): True  → mask out (set to -inf)
    Returns: (output, attn_weights)
    """
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (..., seq_q, seq_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)                               # (..., seq_q, seq_k)
    # Replace NaN (all-masked rows) with 0 so gradients are clean
    attn_w = torch.nan_to_num(attn_w, nan=0.0)
    output = torch.matmul(attn_w, V)                                 # (..., seq_q, d_v)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════
# 2. MASK HELPERS
# ══════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[batch, 1, 1, src_len]  True where PAD"""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = 1) -> torch.Tensor:
    """[batch, 1, tgt_len, tgt_len]  True where PAD or future"""
    batch, tgt_len = tgt.shape
    # Causal mask: upper triangle excluding diagonal
    causal = torch.triu(torch.ones(tgt_len, tgt_len, device=tgt.device), diagonal=1).bool()
    # Padding mask
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)            # [B, 1, 1, T]
    # Combine: mask position if PAD *or* future
    tgt_mask = causal.unsqueeze(0).unsqueeze(0) | pad_mask           # [B, 1, T, T]
    return tgt_mask


# ══════════════════════════════════════════════════════════════════
# 3. MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, D] → [B, H, S, d_k]"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = query.size(0)

        Q = self._split_heads(self.W_q(query))   # [B, H, sq, d_k]
        K = self._split_heads(self.W_k(key))
        V = self._split_heads(self.W_v(value))

        out, _ = scaled_dot_product_attention(Q, K, V, mask)        # [B, H, sq, d_k]
        out    = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)  # [B, sq, D]
        return self.W_o(out)


# ══════════════════════════════════════════════════════════════════
# 4. POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Build [max_len, d_model] sinusoidal table
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)         # [max_len, 1]
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                        -(math.log(10000.0) / d_model))                         # [d_model/2]
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════
# 5. FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════
# 6. ENCODER LAYER
# ══════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """Pre-LayerNorm variant (more stable training)."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # Pre-LN: norm → sublayer → residual
        _x = self.norm1(x)
        x  = x + self.dropout(self.self_attn(_x, _x, _x, src_mask))
        _x = self.norm2(x)
        x  = x + self.dropout(self.ffn(_x))
        return x


# ══════════════════════════════════════════════════════════════════
# 7. DECODER LAYER
# ══════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        _x = self.norm1(x)
        x  = x + self.dropout(self.self_attn(_x, _x, _x, tgt_mask))
        _x = self.norm2(x)
        x  = x + self.dropout(self.cross_attn(_x, memory, memory, src_mask))
        _x = self.norm3(x)
        x  = x + self.dropout(self.ffn(_x))
        return x


# ══════════════════════════════════════════════════════════════════
# 8. ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor,
        src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════
# 9. FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Encoder-Decoder Transformer for De→En translation.

    When no arguments are given, the model is self-contained:
    it loads its own tokenisers and weights inside __init__.
    """

    def __init__(
        self,
        src_vocab_size: int  = None,
        tgt_vocab_size: int  = None,
        d_model:   int   = 256,
        N:         int   = 3,
        num_heads: int   = 8,
        d_ff:      int   = 512,
        dropout:   float = 0.1,
        pad_idx:   int   = 1,
        checkpoint_path: str = None,
        gdrive_id:       str = "1IFtLvl28elhoR-p5zUEPHSF6p_RoETdH",
    ):
        super().__init__()

        # ── If called by autograder without vocab sizes, load everything ──
        self._autograder_mode = (src_vocab_size is None)
        if self._autograder_mode:
            self._load_self_contained(d_model, N, num_heads, d_ff, dropout, pad_idx,
                                      checkpoint_path, gdrive_id)
            return

        self._build(src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, dropout, pad_idx)

        if checkpoint_path is not None:
            if gdrive_id and not os.path.exists(checkpoint_path):
                import gdown
                gdown.download(id=gdrive_id, output=checkpoint_path, quiet=False)
            ckpt = torch.load(checkpoint_path, map_location='cpu')
            self.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print('Checkpoint loaded successfully.')

    def _build(self, src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, dropout, pad_idx):
        self.d_model    = d_model
        self.pad_idx    = pad_idx

        self.src_embed  = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed  = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe     = PositionalEncoding(d_model, dropout)
        self.tgt_pe     = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder    = Encoder(enc_layer, N)
        self.decoder    = Decoder(dec_layer, N)
        self.fc_out     = nn.Linear(d_model, tgt_vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Autograder self-contained mode ──────────────────────────────
    def _load_self_contained(
        self, d_model, N, num_heads, d_ff, dropout, pad_idx,
        checkpoint_path, gdrive_id
    ):
        """
        Called by autograder: Transformer().to(device) with no args.
        Loads vocab, tokenisers, model weights all here.
        """
        import spacy
        from datasets import load_dataset
        from collections import Counter

        UNK, PAD, SOS, EOS = 0, 1, 2, 3
        SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']

        # ── Tokenisers ──
        try:
            self._de_nlp = spacy.load('de_core_news_sm')
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'de_core_news_sm'])
            self._de_nlp = spacy.load('de_core_news_sm')
        try:
            self._en_nlp = spacy.load('en_core_web_sm')
        except Exception:
            import subprocess, sys
            subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            self._en_nlp = spacy.load('en_core_web_sm')

        def tok_de(t): return [w.text.lower() for w in self._de_nlp.tokenizer(t)]
        def tok_en(t): return [w.text.lower() for w in self._en_nlp.tokenizer(t)]
        self._tok_de = tok_de
        self._tok_en = tok_en

        # ── Build vocab from training data ──
        raw   = load_dataset('bentrevett/multi30k', trust_remote_code=True)
        train = raw['train']

        def build_vocab(token_lists, freq=2):
            stoi = {t: i for i, t in enumerate(SPECIALS)}
            itos = {i: t for t, i in stoi.items()}
            cnt  = Counter(w for toks in token_lists for w in toks)
            for w, c in cnt.items():
                if c >= freq and w not in stoi:
                    idx = len(stoi); stoi[w] = idx; itos[idx] = w
            return stoi, itos

        de_toks = [tok_de(ex['de']) for ex in train]
        en_toks = [tok_en(ex['en']) for ex in train]
        self._src_stoi, self._src_itos = build_vocab(de_toks)
        self._tgt_stoi, self._tgt_itos = build_vocab(en_toks)

        src_vocab_size = len(self._src_stoi)
        tgt_vocab_size = len(self._tgt_stoi)
        print(f'Built vocabularies:\n  Source: {src_vocab_size}  Target: {tgt_vocab_size}')

        self._sos = SOS; self._eos = EOS; self._pad = PAD; self._unk = UNK
        self._pad_idx = PAD

        # ── Build model ──
        self._build(src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, dropout, PAD)

        # ── Download & load weights ──
        _cp = checkpoint_path or 'best_checkpoint.pt'
        if gdrive_id and not os.path.exists(_cp):
            import gdown
            gdown.download(id=gdrive_id, output=_cp, quiet=False)
        if os.path.exists(_cp):
            ckpt = torch.load(_cp, map_location='cpu')
            self.load_state_dict(ckpt.get('model_state_dict', ckpt))
            print('Checkpoint loaded successfully.')

    # ── AUTOGRADER HOOKS ────────────────────────────────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── INFER (autograder contract) ──────────────────────────────────

    def infer(self, src_sentence: str, max_len: int = 100) -> str:
        """
        End-to-end De→En translation: raw string in, raw string out.
        Works in both training mode and autograder self-contained mode.
        """
        self.eval()
        device = next(self.parameters()).device

        # ── Choose vocab depending on mode ──
        if self._autograder_mode:
            src_stoi = self._src_stoi
            tgt_itos = self._tgt_itos
            tok_de   = self._tok_de
            sos, eos, pad, unk = self._sos, self._eos, self._pad, self._unk
        else:
            # During training: use module-level vocabs from dataset.py
            # from dataset import Multi30kDataset, SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX
            import spacy
            de_nlp   = spacy.load('de_core_news_sm')
            tok_de   = lambda t: [w.text.lower() for w in de_nlp.tokenizer(t)]
            src_stoi = Multi30kDataset.src_vocab.stoi
            tgt_itos = Multi30kDataset.tgt_vocab.itos
            sos, eos, pad, unk = SOS_IDX, EOS_IDX, PAD_IDX, UNK_IDX

        # ── Tokenise & numericalize source ──
        tokens  = tok_de(src_sentence)
        ids     = [sos] + [src_stoi.get(t, unk) for t in tokens] + [eos]
        src     = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # [1, S]
        src_mask = make_src_mask(src, pad_idx=pad)

        # ── Encode once ──
        with torch.no_grad():
            memory = self.encode(src, src_mask)

            # ── Greedy autoregressive decode ──
            ys = torch.tensor([[sos]], dtype=torch.long, device=device)
            for _ in range(max_len):
                tgt_mask = make_tgt_mask(ys, pad_idx=pad)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)  # [1, t, V]
                next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
                ys       = torch.cat([ys, next_tok], dim=1)
                if next_tok.item() == eos:
                    break

        # ── Detokenize ──
        token_ids = ys[0].tolist()
        words     = []
        for i in token_ids:
            if i in (sos, eos, pad):
                continue
            words.append(tgt_itos.get(i, '<unk>'))
        return ' '.join(words)


# print('model.py ✅')