"""Generation edge cases beyond the starter's public examples."""
import pytest
import torch

from src.model import GPT, GPTConfig


@pytest.fixture
def model():
    torch.manual_seed(17)
    return GPT(GPTConfig(vocab_size=8, block_size=4, n_layer=1,
                         n_head=2, n_embd=8, dropout=0.1))


@pytest.mark.parametrize("shape", [(0, 2), (2, 0), (2,)])
def test_empty_or_malformed_prompts_rejected(model, shape):
    with pytest.raises(ValueError, match="nonempty"):
        model.generate(torch.empty(shape, dtype=torch.long), 1)


@pytest.mark.parametrize("kwargs", [
    {"max_new_tokens": -1}, {"max_new_tokens": 1.5},
    {"temperature": float("inf")}, {"temperature": -1},
    {"top_k": -1}, {"top_k": 1.5},
    {"eos_id": -1}, {"eos_id": 8},
])
def test_invalid_generation_settings(model, kwargs):
    settings = {"max_new_tokens": 1, **kwargs}
    with pytest.raises(ValueError):
        model.generate(torch.tensor([[1]]), **settings)
    assert model.training


@pytest.mark.parametrize("training", [True, False])
def test_zero_length_preserves_prompt_and_mode(model, training):
    model.train(training)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    result = model.generate(prompt, 0)
    assert torch.equal(result, prompt)
    assert model.training == training


def test_batch_eos_waits_for_all_rows(model, monkeypatch):
    calls = []

    def scripted_forward(idx, targets=None):
        assert not model.training
        assert not torch.is_grad_enabled()
        calls.append(idx.clone())
        logits = torch.full((2, idx.shape[1], 8), -100.0)
        # Row 0 ends first; its next prediction must be replaced with EOS.
        choices = [5, 3] if len(calls) == 1 else [4, 5]
        for row, token in enumerate(choices):
            logits[row, -1, token] = 100.0
        return logits, None

    monkeypatch.setattr(model, "forward", scripted_forward)
    prompt = torch.tensor([[1], [2]])
    result = model.generate(prompt, 10, top_k=1, eos_id=5)
    assert torch.equal(result, torch.tensor([[1, 5, 5], [2, 3, 5]]))
    assert torch.equal(prompt, torch.tensor([[1], [2]]))
    assert len(calls) == 2
    assert model.training


@pytest.mark.parametrize("training", [True, False])
def test_mode_restored_after_forward_error(model, monkeypatch, training):
    model.train(training)

    def failing_forward(*args, **kwargs):
        assert not model.training
        raise RuntimeError("simulated forward failure")

    monkeypatch.setattr(model, "forward", failing_forward)
    with pytest.raises(RuntimeError, match="simulated"):
        model.generate(torch.tensor([[1]]), 1)
    assert model.training == training


def test_top1_breaks_ties_like_argmax(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    result = model.generate(torch.tensor([[1]]), 4, top_k=1)
    assert torch.equal(result, torch.tensor([[1, 0, 0, 0, 0]]))


def test_large_top_k_matches_unfiltered_sampling(model):
    prompt = torch.tensor([[1, 2]])
    unfiltered = model.generate(
        prompt, 6, generator=torch.Generator().manual_seed(23))
    large_k = model.generate(
        prompt, 6, top_k=100, generator=torch.Generator().manual_seed(23))
    assert torch.equal(unfiltered, large_k)
