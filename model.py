import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def rope(x, pos):
    # x :: (batch, n_heads, seq, d_head)
    # pos :: (seq,) absolute position indices
    d_head = x.shape[-1]
    freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=x.device, dtype=x.dtype) / d_head))
    angles = pos[:, None] * freqs[None, :]  # (seq, d_head/2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.empty_like(x)
    out[..., 0::2] = out_even
    out[..., 1::2] = out_odd
    return out


class FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, n_heads * self.d_head)
        self.Wk = nn.Linear(d_model, n_heads * self.d_head)
        self.Wv = nn.Linear(d_model, n_heads * self.d_head)
        self.Wo = nn.Linear(n_heads * self.d_head, d_model)

        # Precompute RoPE cos/sin tables for positions [0, max_seq_len).
        d_head = self.d_head
        freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2) / d_head))
        pos = torch.arange(max_seq_len)
        angles = pos[:, None] * freqs[None, :]
        self.register_buffer('rope_cos', torch.cos(angles), persistent=False)
        self.register_buffer('rope_sin', torch.sin(angles), persistent=False)

    def apply_rope(self, x, pos_offset, n_token):
        # x :: (batch, n_heads, seq, d_head)
        if pos_offset + n_token <= self.rope_cos.shape[0]:
            cos = self.rope_cos[pos_offset:pos_offset + n_token]
            sin = self.rope_sin[pos_offset:pos_offset + n_token]
        else:
            # Fall back to dynamic computation if positions exceed precomputed range.
            d_head = x.shape[-1]
            pos = torch.arange(n_token, device=x.device) + pos_offset
            freqs = 1.0 / (10000.0 ** (torch.arange(0, d_head, 2, device=x.device, dtype=x.dtype) / d_head))
            angles = pos[:, None] * freqs[None, :]
            cos = torch.cos(angles)
            sin = torch.sin(angles)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.empty_like(x)
        out[..., 0::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None):
        batch, n_token, d_model = x.shape
        n_heads, d_head = self.n_heads, self.d_head

        Q = self.Wq(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        K = self.Wk(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)
        V = self.Wv(x).reshape(batch, n_token, n_heads, d_head).transpose(1, 2)

        Q = self.apply_rope(Q, pos_offset, n_token)
        K = self.apply_rope(K, pos_offset, n_token)

        if k_cache is not None:
            K = torch.cat([k_cache, K], dim=2)
            V = torch.cat([v_cache, V], dim=2)

        # attn_mask: (batch, seq) bool, True = real token, False = padding.
        if infer:
            # KV cache handles autoregression; only apply padding mask if given.
            if attn_mask is not None:
                pad = attn_mask[:, None, None, :].to(Q.dtype)
                pad = (1.0 - pad) * torch.finfo(Q.dtype).min
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=pad)
            else:
                out = F.scaled_dot_product_attention(Q, K, V)
        else:
            # Training: need causal. If padding mask given, combine with causal.
            if attn_mask is None:
                out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                T = Q.shape[-2]
                Tk = K.shape[-2]
                causal = torch.ones(T, Tk, dtype=torch.bool, device=Q.device).tril()
                causal_add = torch.where(causal, 0.0, torch.finfo(Q.dtype).min)
                pad = attn_mask[:, None, None, :].to(Q.dtype)
                pad = (1.0 - pad) * torch.finfo(Q.dtype).min
                combined = causal_add[None, None, :, :] + pad
                out = F.scaled_dot_product_attention(Q, K, V, attn_mask=combined)
        out = out.transpose(1, 2).reshape(batch, n_token, n_heads * d_head)
        return self.Wo(out), K, V


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FFN(d_model)

    def forward(self, x, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None):
        attn, k_new, v_new = self.attn(self.ln1(x), k_cache=k_cache, v_cache=v_cache,
                                       infer=infer, pos_offset=pos_offset, attn_mask=attn_mask)
        x = x + attn
        x = x + self.ffn(self.ln2(x))
        return x, k_new, v_new


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_len):
        super().__init__()
        self.W_embed = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, max_seq_len) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, k_cache=None, v_cache=None, infer=False, pos_offset=0, attn_mask=None):
        x = self.W_embed[token_ids]
        if not infer:
            # Training path: gradient checkpointing + skip building KV caches
            # to save activation memory.
            for block in self.blocks:
                x, _, _ = checkpoint(block, x, attn_mask=attn_mask, use_reentrant=False)
            x = self.ln_f(x)
            logits = self.head(x)
            return logits, None, None

        k_caches = []
        v_caches = []
        for i, block in enumerate(self.blocks):
            x, k, v = block(
                x,
                k_cache=k_cache[i] if k_cache is not None else None,
                v_cache=v_cache[i] if v_cache is not None else None,
                infer=infer,
                pos_offset=pos_offset
            )
            k_caches.append(k)
            v_caches.append(v)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, k_caches, v_caches