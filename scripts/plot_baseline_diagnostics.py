"""Create baseline report figures and a result summary from real run evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--report-dir", type=Path, default=Path("report"))
    args = parser.parse_args()
    read = lambda name: json.loads((args.run / name).read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.run / "eval.jsonl").read_text().splitlines()]
    metrics, evaluation, environment = (read(name) for name in
        ("metrics.json", "evaluation.json", "environment.json"))
    samples = read("samples.json")
    summary = {
        "run": args.run.as_posix(), "metrics": metrics,
        "full_fp32_evaluation": evaluation, "environment": environment,
        "sampling": {k: v for k, v in samples.items() if k != "samples"},
        "curve_observations": {
            "evaluation_points": len(rows),
            "initial_sampled_val_loss": rows[0]["val_loss"],
            "final_sampled_val_loss": rows[-1]["val_loss"],
            "strictly_decreasing_at_recorded_points": all(
                b["val_loss"] < a["val_loss"] for a, b in zip(rows, rows[1:])),
            "last_500_step_val_improvement": rows[-3]["val_loss"] - rows[-1]["val_loss"],
        },
        "timing_note": "Trainer runtime excludes initial preparation and postprocessing; "
                       "train_seconds sums update compute timings. Peak memory is in MiB.",
        "protocol_note": "Both plotted curves use sampled windows in eval mode with "
                         "BF16 autocast. Full-split FP32 results are a separate protocol.",
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "baseline_results.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    for ax, selected, title in zip(axes, (rows, [r for r in rows if r["step"] >= 1000]),
                                   ("Entire training run", "Late phase: steps 1,000-2,500")):
        for key, label, color in (("train_loss", "Sampled train split", "#276B96"),
                                  ("val_loss", "Sampled validation split", "#C36B35")):
            ax.plot([r["train_tokens"] / 1e6 for r in selected],
                    [r[key] for r in selected], label=label, color=color,
                    linewidth=1.8, marker="o", markersize=3)
        ax.set(title=title, xlabel="Processed target tokens (millions)",
               ylabel="Cross-entropy (nats/token)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Baseline learning curves | sampled BF16 evaluation; dropout disabled", fontsize=12)
    figures = args.report_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(figures / f"baseline_training_diagnostics.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("Saved baseline_results.json and baseline_training_diagnostics PNG/SVG")


if __name__ == "__main__":
    main()
