"""Focused attention checks that do not depend on the remaining GPT TODOs."""
import math

import pytest
import torch

from src.model import CausalSelfAttention, GPTConfig


@pytest.mark.parametrize("batch,length,width,heads", [
    (1, 1, 8, 1),
    (2, 5, 16, 2),
    (2, 16, 24, 3),
])
def test_attention_shapes_and_gradients(batch, length, width, heads):
    torch.manual_seed(7)
    cfg = GPTConfig(block_size=16, n_embd=width, n_head=heads, dropout=0.1)
    attention = CausalSelfAttention(cfg).train()
    x = torch.randn(batch, length, width, requires_grad=True)
    y = attention(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in attention.parameters())


def test_attention_matches_prefix_only_reference():
    """Compute each head/query using only its allowed prefix, without a mask."""
    torch.manual_seed(11)
    cfg = GPTConfig(block_size=8, n_embd=12, n_head=3, dropout=0.0)
    attention = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 5, cfg.n_embd)
    with torch.no_grad():
        q, k, v = attention.c_attn(x).chunk(3, dim=-1)
        head_size = cfg.n_embd // cfg.n_head
        heads = []
        for head in range(cfg.n_head):
            features = slice(head * head_size, (head + 1) * head_size)
            positions = []
            for pos in range(x.shape[1]):
                query = q[:, pos:pos + 1, features]
                keys = k[:, :pos + 1, features]
                values = v[:, :pos + 1, features]
                scores = (query @ keys.transpose(-2, -1)) / math.sqrt(head_size)
                positions.append(scores.softmax(dim=-1) @ values)
            heads.append(torch.cat(positions, dim=1))
        expected = attention.c_proj(torch.cat(heads, dim=-1))
        torch.testing.assert_close(attention(x), expected)


def test_attention_blocks_future_values_and_gradients():
    torch.manual_seed(13)
    attention = CausalSelfAttention(GPTConfig(
        block_size=8, n_embd=16, n_head=2, dropout=0.0)).eval()
    x = torch.randn(2, 8, 16, requires_grad=True)
    y = attention(x)
    altered = x.detach().clone()
    altered[:, 4:] += torch.randn_like(altered[:, 4:]) * 10
    with torch.no_grad():
        torch.testing.assert_close(y[:, :4], attention(altered)[:, :4])
    y[:, :4].square().sum().backward()
    assert torch.count_nonzero(x.grad[:, 4:]).item() == 0
