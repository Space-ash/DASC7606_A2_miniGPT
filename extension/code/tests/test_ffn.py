"""Budget, numeric equivalence, portability and checkpoint-format checks."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from minigpt_ext.model import GPT, GPTConfig, MLP
from minigpt_ext.utils import ARCHITECTURE, load_model, load_config, save_checkpoint


@pytest.mark.parametrize("kind,count", [("gelu", 13891584), ("silu", 13891584),
                                       ("swiglu", 13894656)])
def test_formal_parameter_budget(kind, count):
    cfg = GPTConfig(ffn_type=kind)
    model = GPT(cfg)
    assert model.parameter_count() == cfg.expected_parameters() == count
    assert model.lm_head.weight is model.transformer.wte.weight
    assert cfg.ffn_hidden_size == (1024 if kind == "swiglu" else 1536)


def test_formal_matrix_has_only_declared_changes():
    extension = Path(__file__).resolve().parents[2]
    configs = list((extension / "configs").glob("*_seed*.yaml"))
    assert len(configs) == 9
    base = load_config(extension / "configs/gelu_seed1337.yaml")
    for path in configs:
        cfg = load_config(path)
        kind, seed = path.stem.split("_seed")
        assert cfg["model"]["ffn_type"] == kind
        assert cfg["model"]["ffn_hidden_size"] == (1024 if kind == "swiglu" else 1536)
        assert cfg["training"]["seed"] == int(seed)
        assert cfg["training"]["max_steps"] * cfg["training"]["batch_size"] * cfg["training"]["gradient_accumulation"] * cfg["model"]["block_size"] == 10240000
        model = dict(cfg["model"])
        model.update(ffn_type="gelu", ffn_hidden_size=1536)
        training = dict(cfg["training"], seed=1337)
        assert model == base["model"] and training == base["training"]


def test_gelu_same_initialization_logits_and_gradients_as_core():
    core = pytest.importorskip("src.model", reason="core repo available for equivalence audit only")
    values = dict(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    torch.manual_seed(19)
    original = core.GPT(core.GPTConfig(**values))
    torch.manual_seed(19)
    extended = GPT(GPTConfig(**values))
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(tensor, extended.state_dict()[name], rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4]])
    a, loss_a = original(ids, ids)
    b, loss_b = extended(ids, ids)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    loss_a.backward()
    loss_b.backward()
    for (_, p), (_, q) in zip(original.named_parameters(), extended.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)


def test_gate_is_independent_and_receives_gradients():
    cfg = GPTConfig(vocab_size=32, n_layer=1, n_embd=12, n_head=3,
                    ffn_type="swiglu", dropout=0.0)
    layer = MLP(cfg)
    assert layer.c_fc.weight is not layer.c_gate.weight
    x = torch.randn(2, 5, 12, requires_grad=True)
    layer(x).square().mean().backward()
    for parameter in (layer.c_fc.weight, layer.c_gate.weight, layer.c_proj.weight):
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


@pytest.mark.parametrize("kind", ["gelu", "silu", "swiglu"])
def test_checkpoint_metadata_required_and_count_verified(kind, tmp_path):
    cfg = GPTConfig(vocab_size=32, n_layer=1, n_embd=12, n_head=3, ffn_type=kind)
    model = GPT(cfg)
    checkpoint = dict(format_version=1, architecture_family=ARCHITECTURE,
                      model_config=asdict(cfg), model=model.state_dict(),
                      parameter_count=model.parameter_count())
    path = tmp_path / "model.pt"
    save_checkpoint(path, checkpoint)
    restored, _ = load_model(path)
    assert restored.config == cfg
    checkpoint["parameter_count"] += 1
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="parameter metadata"):
        load_model(path)
    checkpoint.pop("architecture_family")
    save_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match="architecture metadata"):
        load_model(path)


@pytest.mark.parametrize("kwargs", [{"ffn_type": "unknown"}, {"ffn_hidden_size": 0},
                                    {"ffn_hidden_size": True}, {"ffn_hidden_size": 1.5}])
def test_invalid_ffn_config(kwargs):
    with pytest.raises(ValueError):
        GPTConfig(**kwargs)


@pytest.mark.parametrize("entry", ["train.py", "evaluate.py", "sample.py"])
def test_entry_points_from_unrelated_directory(entry, tmp_path):
    path = Path(__file__).resolve().parents[1] / entry
    result = subprocess.run([sys.executable, "-I", str(path), "--help"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="local CPU environment; CUDA smoke required on server")
@pytest.mark.parametrize("kind", ["gelu", "silu", "swiglu"])
def test_cuda_bf16(kind):
    if not torch.cuda.is_bf16_supported():
        pytest.skip("allocated GPU lacks BF16 support")
    model = GPT(GPTConfig(vocab_size=32, block_size=16, n_layer=1,
                          n_head=3, n_embd=12, ffn_type=kind)).cuda()
    ids = torch.randint(32, (2, 8), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = model(ids, ids)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
