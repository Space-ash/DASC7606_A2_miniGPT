"""Exercise all three FFNs without starting a formal experiment."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from minigpt_ext.utils import load_model, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-root", required=True, type=Path)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    configs = code.parent / "configs" / "smoke"
    if args.out_root.exists() and any(args.out_root.iterdir()):
        raise FileExistsError("choose a fresh smoke output directory")
    args.out_root.mkdir(parents=True, exist_ok=True)

    def run(entry, *arguments):
        command = [sys.executable, "-s", "-u", str(code / entry), *map(str, arguments)]
        print("RUN:", " ".join(command), flush=True)
        subprocess.run(command, check=True)

    def train(kind, out, *extra):
        run("train.py", "--config", configs / f"{kind}.yaml", "--out_dir", out,
            "--data-dir", args.data_dir, "--device", args.device, *extra)

    summary = {"purpose": "smoke_only_not_formal_evidence", "device": args.device,
               "runs": [], "resume_check": None}
    for kind in ("gelu", "silu", "swiglu"):
        out = args.out_root / kind
        train(kind, out)
        run("plot_run.py", out)
        run("sample.py", "--checkpoint", out / "best.pt", "--tokenizer",
            args.data_dir / "tokenizer.json", "--out", out / "samples.json",
            "--device", args.device, "--max-new-tokens", 16)
        run("evaluate.py", "--checkpoint", out / "best.pt", "--tokens",
            args.data_dir / "val.bin", "--tokenizer", args.data_dir / "tokenizer.json",
            "--out", out / "evaluation.json", "--device", args.device, "--max-tokens", 1024)
        read = lambda name: json.loads((out / name).read_text())
        metrics, evaluation, samples = (read(name) for name in
                                        ("metrics.json", "evaluation.json", "samples.json"))
        assert metrics["status"] == "complete" and metrics["step"] == 20
        assert metrics["train_tokens"] == 10240
        assert evaluation["evaluated_tokens"] == 1024 and math.isfinite(evaluation["cross_entropy"])
        assert len(samples["samples"]) == 3
        assert sha256(out / "best.pt") == evaluation["checkpoint_sha256"] == samples["checkpoint_sha256"]
        restored, checkpoint = load_model(out / "best.pt")
        assert restored.config.ffn_type == kind and checkpoint["parameter_count"] == restored.parameter_count()
        summary["runs"].append({"ffn_type": kind, "output_directory": out.as_posix(),
            "status": "pass", "step": metrics["step"], "train_tokens": metrics["train_tokens"],
            "parameter_count": metrics["parameter_count"],
            "effective_precision": metrics["effective_precision"],
            "evaluated_tokens": evaluation["evaluated_tokens"],
            "cross_entropy": evaluation["cross_entropy"],
            "source_sha256": read("environment.json")["source_sha256"]})

    # An interrupted run must reproduce the uninterrupted run's parameters and RNG.
    resumed = args.out_root / "swiglu_resume"
    train("swiglu", resumed, "--stop-after-steps", 10)
    train("swiglu", resumed, "--resume")
    full = torch.load(args.out_root / "swiglu" / "last.pt", map_location="cpu", weights_only=True)
    restored = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    assert restored["step"] == full["step"] == 20
    for name, tensor in full["model"].items():
        torch.testing.assert_close(tensor, restored["model"][name], rtol=0, atol=0)
    assert torch.equal(full["torch_rng"], restored["torch_rng"])
    assert torch.equal(full["batch_rng"], restored["batch_rng"])
    summary["resume_check"] = {"status": "pass", "interrupted_step": 10,
                               "final_step": 20, "model_and_rng_bitwise_equal": True}
    summary["total_smoke_training_tokens"] = 4 * 10240
    write_json(args.out_root / "summary.json", summary)
    print("ALL FFN SMOKE PATHS AND EXACT RESUME: PASS", flush=True)


if __name__ == "__main__":
    main()
