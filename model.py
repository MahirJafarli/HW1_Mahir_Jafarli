from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Key, Query, Value projections in a single linear layer
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask to ensure attention is only applied to past positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
        )

    def forward(self, x, kv_cache=None):
        B, T, C = x.size() # Batch size, sequence length, embedding dimension

        # Calculate Query, Key, Value for all heads in batch
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, H, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Handle KV cache for inference if provided (Part D compatibility)
        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)
            new_kv_cache = (k, v)
        else:
            new_kv_cache = None

        K_len = k.size(2)

        # Compute causal attention scores: (B, H, T, K_len)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # Apply causal mask
        if kv_cache is None:
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        else:
            # During cached inference, new token attends to all previous tokens
            pass

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, H, T, head_dim)

        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        
        if kv_cache is not None:
            return y, new_kv_cache
        return y

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x, approximate='tanh')
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None):
        if kv_cache is not None:
            attn_out, new_kv_cache = self.attn(self.ln_1(x), kv_cache=kv_cache)
            x = x + attn_out
            x = x + self.mlp(self.ln_2(x))
            return x, new_kv_cache
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying
        self.lm_head.weight = self.transformer.wte.weight

        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        """Calculates unique trainable parameters (tied weights counted once)."""
        V, T, d, L = self.config.vocab_size, self.config.block_size, self.config.n_embd, self.config.n_layer
        return V * d + T * d + L * (12 * (d ** 2) + 4 * d) + 2 * d

    def forward(self, idx, targets=None, kv_caches=None):
        device = idx.device
        B, T = idx.size()
        
        if kv_caches is None:
            pos = torch.arange(0, T, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            for block in self.transformer.h:
                x = block(x)
        else:
            # Inference with KV cache
            past_len = kv_caches[0][0].size(2) if kv_caches[0] is not None else 0
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=device)
            tok_emb = self.transformer.wte(idx)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)

            new_kv_caches = []
            for i, block in enumerate(self.transformer.h):
                x, new_kv = block(x, kv_cache=kv_caches[i])
                new_kv_caches.append(new_kv)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            # Inference optimization: compute logits for the last position only
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        if kv_caches is not None:
            return logits, loss, new_kv_caches
        return logits, loss