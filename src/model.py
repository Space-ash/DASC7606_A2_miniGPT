"""Small GPT, adapted from nanoGPT's MIT-licensed architecture (see LICENSE).

Student starter: complete the four TODO methods. See README.md.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    attention_backend: str = "manual"

    def __post_init__(self):
        for name in ("vocab_size", "block_size", "n_layer", "n_head", "n_embd"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.attention_backend != "manual":
            raise ValueError("unknown attention backend")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.ones(config.block_size, config.block_size,
                                               dtype=torch.bool).tril(), persistent=False)

    def forward(self, x):
        """Apply explicit multi-head causal self-attention to [B, T, C]."""
        B, T, C = x.shape
        n_head = self.config.n_head
        head_size = C // n_head

        # Project each token to queries, keys and values, each of shape [B, T, C].
        q, k, v = self.c_attn(x).split(C, dim=-1)
        # Split features into heads: [B, T, C] -> [B, n_head, T, head_size].
        q = q.view(B, T, n_head, head_size).transpose(1, 2)
        k = k.view(B, T, n_head, head_size).transpose(1, 2)
        v = v.view(B, T, n_head, head_size).transpose(1, 2)

        # Compute scaled query-key scores: [B, n_head, T, T].
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        # Block future keys; the [T, T] mask broadcasts over batches and heads.
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        # Normalize over key positions, then apply attention dropout.
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        y = weights @ v

        # Merge heads; contiguous() makes the transposed tensor safe to view.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # Mix information from all heads and regularize the output.
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Transform each token's features independently, preserving [B, T, C]."""
        # Expand the feature dimension: [B, T, C] -> [B, T, 4 * C].
        x = self.c_fc(x)
        # Apply a nonlinear activation without changing the shape.
        x = F.gelu(x)
        # Project back to the model width: [B, T, 4 * C] -> [B, T, C].
        x = self.c_proj(x)
        # Randomly zero output elements during training for regularization.
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """Apply pre-normalized attention and MLP with residual connections."""
        # Normalize the input, then add the attention output to the residual stream.
        x = x + self.attn(self.ln_1(x))
        # Normalize the updated stream before the MLP and its residual addition.
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None):
        if idx.ndim != 2 or not 0 < idx.shape[1] <= self.config.block_size:
            raise ValueError("input must be [batch, sequence] within block_size")
        pos = torch.arange(idx.shape[1], device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        logits = self.lm_head(self.transformer.ln_f(x))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None,
                 eos_id=None, generator=None):
        """Append sampled tokens to nonempty prompts, stopping on EOS or length."""
        # Validate generation settings before changing the model's mode.
        if idx.ndim != 2 or idx.shape[0] == 0 or idx.shape[1] == 0:
            raise ValueError("prompts must be nonempty [batch, sequence] tensors")
        if idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("prompts must contain integer token IDs")
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("temperature must be finite and positive")
        if top_k is not None and (type(top_k) is not int or top_k <= 0):
            raise ValueError("top_k must be a positive integer or None")
        if eos_id is not None and (type(eos_id) is not int
                                   or not 0 <= eos_id < self.config.vocab_size):
            raise ValueError("eos_id must be a valid vocabulary ID or None")

        was_training = self.training
        self.eval()
        try:
            # Track newly generated EOS independently for each row in the batch.
            finished = torch.zeros(idx.shape[0], dtype=torch.bool, device=idx.device)
            for _ in range(max_new_tokens):
                # Crop only the model's input; keep the full prompt in the result.
                context = idx[:, -self.config.block_size:]
                logits, _ = self(context)
                # The last position predicts the next token for each sequence.
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    # Keep at most k candidates; top-k=1 uses greedy tie-breaking.
                    if top_k == 1:
                        indices = logits.argmax(dim=-1, keepdim=True)
                        values = logits.gather(-1, indices)
                    else:
                        values, indices = torch.topk(
                            logits, min(top_k, logits.shape[-1]), dim=-1)
                    logits = torch.full_like(logits, float("-inf")).scatter(
                        -1, indices, values)

                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1, generator=generator)
                if eos_id is not None:
                    # Finished rows append EOS while unfinished rows keep sampling.
                    next_token = next_token.masked_fill(finished[:, None], eos_id)
                    finished = finished | (next_token[:, 0] == eos_id)
                idx = torch.cat((idx, next_token), dim=1)
                if eos_id is not None and finished.all():
                    break
            return idx
        finally:
            # Restore the previous mode even if forward or sampling raises an error.
            self.train(was_training)
