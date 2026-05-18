"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # Get the scaling factor
    d_k = K.shape[-1]
    
    # Compute attention scores: Q · K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # mask is True where we want to mask out, so set those to -inf
        scores = scores.masked_fill(mask, float('-inf'))
    
    # Apply softmax to get attention weights
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Handle NaN values from softmax over all -inf (shouldn't happen with proper masking)
    attn_weights = torch.nan_to_num(attn_weights, 0.0)
    
    # Apply attention weights to values
    output = torch.matmul(attn_weights, V)
    
    return output, attn_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # Create boolean mask where True means padding
    # src == pad_idx gives [batch, src_len]
    # Reshape to [batch, 1, 1, src_len] for broadcasting
    mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    
    # Padding mask: True where tgt == pad_idx
    # Shape: [batch, tgt_len]
    pad_mask = (tgt == pad_idx)
    
    # Causal mask: lower triangular matrix to prevent attending to future positions
    # Create a mask where future positions are True
    # torch.triu creates upper triangular, so we use it with k=1 to get future positions
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    )
    
    # Combine padding and causal mask
    # Expand padding mask to [batch, tgt_len, tgt_len] by broadcasting
    pad_mask_expanded = pad_mask.unsqueeze(2)  # [batch, tgt_len, 1]
    
    # Combine: True if it's padding OR it's a future position
    combined_mask = pad_mask_expanded | causal_mask.unsqueeze(0)  # [batch, tgt_len, tgt_len]
    
    # Add head dimension: [batch, 1, tgt_len, tgt_len]
    combined_mask = combined_mask.unsqueeze(1)
    
    return combined_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        # Linear projections for Q, K, V and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """
        batch_size = query.shape[0]
        
        # Project Q, K, V
        Q = self.W_q(query)  # [batch, seq_q, d_model]
        K = self.W_k(key)    # [batch, seq_k, d_model]
        V = self.W_v(value)  # [batch, seq_k, d_model]
        
        # Reshape for multi-head attention
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k] -> [batch, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # Apply dropout to attention weights
        attn_output = self.dropout(attn_output)
        
        # Concatenate heads: [batch, num_heads, seq_q, d_k] -> [batch, seq_q, num_heads, d_k] -> [batch, seq_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        return output


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        # Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Dimension indices: create the denominator for the sine/cosine formula
        # div_term = 1 / (10000^(2i/d_model))
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even dimensions
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd dimensions
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as a buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        seq_len = x.shape[1]
        # Add positional encoding and apply dropout
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Post-LayerNorm: normalize after Add
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        # Self-attention with post-norm (Add & Norm)
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with post-norm (Add & Norm)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # Post-LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention to encoder output (memory)
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(self,
                src_vocab_size=None,
                tgt_vocab_size=None,
                d_model=512,
                N=6,
                num_heads=8,
                d_ff=2048,
                dropout=0.1,
            ):

        super().__init__()

        import spacy
        import gdown
        from dataset import Multi30kDataset

        # =========================================================
        # TOKENIZERS
        # =========================================================

        self.src_tokenizer = spacy.blank("de")
        self.tgt_tokenizer = spacy.blank("en")

        # =========================================================
        # VOCAB
        # =========================================================

        train_dataset = Multi30kDataset(split="train")

        self.src_vocab = train_dataset.src_vocab
        self.tgt_vocab = train_dataset.tgt_vocab

        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)

        # ========================================================= 
        # https://drive.google.com/file/d/1lHSAlDF55R7j9mfLh2wIaCzC7kLOBdS2/view?usp=sharing
        # https://drive.google.com/file/d/1lHSAlDF55R7j9mfLh2wIaCzC7kLOBdS2/view?usp=sharing
        # =========================================================

        checkpoint_path = "best_checkpoint.pt"

        if not os.path.exists(checkpoint_path):

            file_id = "1lHSAlDF55R7j9mfLh2wIaCzC7kLOBdS2"

            url = f"https://drive.google.com/uc?id={file_id}"

            gdown.download(
                url,
                checkpoint_path,
                quiet=False
            )

        # =========================================================
        # LOAD CHECKPOINT FIRST
        # =========================================================

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        # =========================================================
        # READ CONFIG FROM CHECKPOINT
        # =========================================================

        if "model_config" in checkpoint:

            config = checkpoint["model_config"]

            d_model = config["d_model"]
            N = config["N"]
            num_heads = config["num_heads"]
            d_ff = config["d_ff"]
            dropout = config["dropout"]

        # =========================================================
        # SAVE CONFIG
        # =========================================================

        self.d_model = d_model
        self.N = N
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout

        # =========================================================
        # BUILD MODEL
        # =========================================================

        self.src_embed = nn.Embedding(
            self.src_vocab_size,
            d_model
        )

        self.tgt_embed = nn.Embedding(
            self.tgt_vocab_size,
            d_model
        )

        self.pos_encoding = PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(
            encoder_layer,
            N
        )

        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.decoder = Decoder(
            decoder_layer,
            N
        )

        self.fc_out = nn.Linear(
            d_model,
            self.tgt_vocab_size
        )

        # =========================================================
        # LOAD WEIGHTS
        # =========================================================

        self.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print("Checkpoint loaded successfully.")
    
    def _load_vocab_and_tokenizers(self):
        """Load vocabulary and spacy tokenizers."""
        try:
            from dataset import Multi30kDataset
            dataset = Multi30kDataset()
            self.src_vocab = dataset.src_vocab
            self.tgt_vocab = dataset.tgt_vocab
            self.src_tokenizer = dataset.src_tokenizer
            self.tgt_tokenizer = dataset.tgt_tokenizer
            
            if self.src_vocab_size is None:
                self.src_vocab_size = len(self.src_vocab)
            if self.tgt_vocab_size is None:
                self.tgt_vocab_size = len(self.tgt_vocab)
        except:
            # If dataset loading fails, use defaults
            if self.src_vocab_size is None:
                self.src_vocab_size = 10000
            if self.tgt_vocab_size is None:
                self.tgt_vocab_size = 10000

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # Embed and add positional encoding
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through encoder
        memory = self.encoder(x, src_mask)
        return memory

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Embed and add positional encoding
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Pass through decoder
        x = self.decoder(x, memory, src_mask, tgt_mask)
        
        # Project to vocabulary
        logits = self.fc_out(x)
        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        logits = self.decode(memory, src_mask, tgt, tgt_mask)
        return logits

    def infer(self, german_sentence: str, max_len: int = 100) -> str:

        self.eval()

        device = next(self.parameters()).device

        with torch.no_grad():

            # =====================================================
            # TOKENIZE
            # =====================================================

            tokens = [
                token.text.lower()
                for token in self.src_tokenizer.tokenizer(german_sentence)
            ]

            # =====================================================
            # CONVERT TO IDS
            # =====================================================

            src_indices = [self.src_vocab.stoi["<sos>"]]

            for token in tokens:

                src_indices.append(
                    self.src_vocab.stoi.get(
                        token,
                        self.src_vocab.stoi["<unk>"]
                    )
                )

            src_indices.append(
                self.src_vocab.stoi["<eos>"]
            )

            # =====================================================
            # TENSOR
            # =====================================================

            src_tensor = torch.LongTensor(
                src_indices
            ).unsqueeze(0).to(device)

            # =====================================================
            # MASK
            # =====================================================

            src_mask = make_src_mask(
                src_tensor,
                pad_idx=self.src_vocab.stoi["<pad>"]
            ).to(device)

            # =====================================================
            # ENCODE
            # =====================================================

            memory = self.encode(
                src_tensor,
                src_mask
            )

            # =====================================================
            # GREEDY DECODING
            # =====================================================

            tgt_indices = [
                self.tgt_vocab.stoi["<sos>"]
            ]

            for _ in range(max_len):

                tgt_tensor = torch.LongTensor(
                    tgt_indices
                ).unsqueeze(0).to(device)

                tgt_mask = make_tgt_mask(
                    tgt_tensor,
                    pad_idx=self.tgt_vocab.stoi["<pad>"]
                ).to(device)

                output = self.decode(
                    memory,
                    src_mask,
                    tgt_tensor,
                    tgt_mask
                )

                next_token = output[:, -1, :].argmax(-1).item()

                tgt_indices.append(next_token)

                if next_token == self.tgt_vocab.stoi["<eos>"]:
                    break

            # =====================================================
            # DETOKENIZE
            # =====================================================

            output_tokens = []

            for idx in tgt_indices[1:]:

                token = self.tgt_vocab.itos[idx]

                if token == "<eos>":
                    break

                if token not in ["<pad>", "<sos>", "<unk>"]:

                    output_tokens.append(token)

            return " ".join(output_tokens)