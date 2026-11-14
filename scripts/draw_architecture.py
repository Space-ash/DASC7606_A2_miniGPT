"""Render report-ready architecture figures from the actual baseline config.

Run from any directory: python scripts/draw_architecture.py
Outputs are PNG (300 dpi) and editable-text SVG under report/figures/.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.model import GPT, GPTConfig

OUT = ROOT / "report" / "figures"
NAVY, GRAY, LINE = "#152B42", "#526477", "#53677B"
BLUE, TEAL, AMBER = "#EAF2FC", "#E6F5F0", "#FFF2DB"
RED, BORDER = "#A54D3A", "#C9D5DF"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "font.size": 11, "text.color": NAVY})


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, width), ylim=(0, height), aspect="equal")
    ax.axis("off")
    return fig, ax


def label(ax, x, y, text, size=11, weight="normal", color=NAVY, ha="center"):
    return ax.text(x, y, text, fontsize=size, weight=weight, color=color,
                   ha=ha, va="center", linespacing=1.4)


def box(ax, x, y, title, detail="", w=3.35, h=.68, color=BLUE, size=11):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.025,rounding_size=0.08", fc=color, ec=BORDER, lw=1))
    if detail:
        label(ax, x, y+.13, title, size, "bold")
        label(ax, x, y-.15, detail, size-1, color=GRAY)
    else:
        label(ax, x, y, title, size, "bold")


def arrow(ax, start, end, color=LINE, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
        lw=1.25, color=color, linestyle="--" if dashed else "-", shrinkA=1, shrinkB=1))


def path(ax, points, color=RED, dashed=False):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=1.25, linestyle="--" if dashed else "-")


def add(ax, x, y):
    ax.add_patch(Circle((x, y), .17, fc="white", ec=LINE, lw=1.2, zorder=3))
    label(ax, x, y, "+", 15)


def panel(ax, left, bottom, width, height, title):
    ax.add_patch(FancyBboxPatch((left, bottom), width, height,
        boxstyle="round,pad=0.025,rounding_size=0.12", fc="#FAFCFE", ec=BORDER, lw=1))
    label(ax, left+.18, bottom+height-.30, title, 12, "bold", ha="left")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)


def detailed(c, count):
    fig, ax = canvas(16, 12.5)
    C, V, H = c.n_embd, c.vocab_size, c.n_head
    label(ax, .45, 12.06, "MiniGPT | Decoder-only Transformer", 22, "bold", ha="left")
    label(ax, .45, 11.63,
        f"Baseline: {c.n_layer} blocks · {H} heads · width {C} · context ≤ {c.block_size} · "
        f"vocabulary {V:,} · {count:,} parameters", 11, color=GRAY, ha="left")
    panel(ax, .30, 3.15, 4.85, 8.12, "A  Full forward network")
    panel(ax, 5.48, 3.15, 4.78, 8.12, "B  One pre-norm block")
    panel(ax, 10.59, 3.15, 5.10, 8.12, "C  Causal self-attention")

    # A: embedding addition, repeated blocks and the tied output projection.
    a = 2.73
    box(ax, 1.67, 10.30, "Token IDs", "idx: [B,T]", w=1.82, color=TEAL)
    box(ax, 3.79, 10.30, "Positions", "0…T−1: [T]", w=1.82, color=TEAL)
    box(ax, 1.67, 9.30, "Token embedding", "[B,T,C]", w=1.82, size=10.5)
    box(ax, 3.79, 9.30, "Position embedding", "[T,C] → broadcast", w=1.82, size=10.5)
    for x in (1.67, 3.79):
        arrow(ax, (x, 9.94), (x, 9.66))
        path(ax, [(x, 8.94), (x, 8.52), (a, 8.52)], color=LINE)
    add(ax, a, 8.52)
    box(ax, a, 7.83, "Embedding dropout", "[B,T,C]", w=3.1)
    arrow(ax, (a, 8.34), (a, 8.19))
    nodes = [(6.83, f"Transformer block × {c.n_layer}", "[B,T,C] → [B,T,C]", TEAL),
             (5.83, "Final LayerNorm", "[B,T,C]", BLUE),
             (4.83, "LM head: Linear(C,V)", "No bias · tied embedding weights", BLUE),
             (3.83, "Logits", "[B,T,V] · unnormalized scores", AMBER)]
    previous = 7.83
    for y, title, detail, color in nodes:
        box(ax, a, y, title, detail, w=3.7, color=color)
        arrow(ax, (a, previous-.36), (a, y+.36))
        previous = y
    path(ax, [(.74, 9.30), (.52, 9.30), (.52, 4.83), (.84, 4.83)], color=RED, dashed=True)
    arrow(ax, (.60, 4.83), (.84, 4.83), color=RED, dashed=True)
    ax.text(.62, 7.12, "shared W [V,C]", rotation=90, fontsize=9, color=RED,
            ha="left", va="center")
    arrow(ax, (4.62, 6.83), (5.36, 6.83), dashed=True)

    # B: bypasses carry the unnormalized residual stream into each addition.
    b = 7.87
    label(ax, b, 10.39, "Input x₀: [B,T,C]", 11, "bold")
    for y, title, detail, color in [
        (9.48, "LayerNorm₁", "Normalize each token over C features", BLUE),
        (8.38, "Causal self-attention", "[B,T,C] → [B,T,C] · see C", TEAL),
        (6.38, "LayerNorm₂", "Uses the updated residual stream x₁", BLUE),
        (5.28, "Token-wise MLP", f"C → {4*C} → C · GELU · dropout", TEAL),
    ]:
        box(ax, b, y, title, detail, w=3.35, color=color)
    arrow(ax, (b, 10.16), (b, 9.84))
    arrow(ax, (b, 9.12), (b, 8.74))
    add(ax, b, 7.38)
    arrow(ax, (b, 8.02), (b, 7.56))
    arrow(ax, (b, 7.20), (b, 6.74))
    arrow(ax, (b, 6.02), (b, 5.64))
    add(ax, b, 4.28)
    arrow(ax, (b, 4.92), (b, 4.46))
    arrow(ax, (b, 4.10), (b, 3.73))
    label(ax, b, 3.51, "Output x₂: [B,T,C]", 11, "bold")
    path(ax, [(b, 10.04), (5.82, 10.04), (5.82, 7.38), (7.68, 7.38)])
    arrow(ax, (7.40, 7.38), (7.68, 7.38), color=RED)
    path(ax, [(b, 6.97), (9.91, 6.97), (9.91, 4.28), (8.06, 4.28)])
    arrow(ax, (8.35, 4.28), (8.06, 4.28), color=RED)
    label(ax, 6.11, 7.64, "x₀", 10, color=RED)
    label(ax, 9.55, 4.54, "x₁", 10, color=RED)
    label(ax, 8.68, 7.38, "x₁", 10, color=GRAY)
    arrow(ax, (9.58, 8.38), (10.47, 8.38), dashed=True)

    # C: dimensions follow the manual attention implementation in src/model.py.
    d = 13.14
    label(ax, d, 10.39, "Input: [B,T,C]", 11, "bold")
    stages = [
        (9.48, "Q / K / V projection", "[B,T,3C] → three [B,T,C] tensors"),
        (8.48, f"Split {H} heads", f"Q, K, V: [B,H,T,d] · d=C/H={C//H}"),
        (7.48, "Scaled dot-product scores", "QKᵀ / √d: [B,H,T,T]"),
        (6.48, "Causal mask → softmax → dropout", "Future keys masked to −∞ · [B,H,T,T]"),
        (5.48, "Weighted value sum", "Attention weights @ V: [B,H,T,d]"),
        (4.48, "Merge heads → projection → dropout", "[B,T,C] → [B,T,C]"),
    ]
    arrow(ax, (d, 10.16), (d, 9.84))
    for i, (y, title, detail) in enumerate(stages):
        box(ax, d, y, title, detail, w=4.46, size=10.8, color=TEAL)
        if i:
            arrow(ax, (d, stages[i-1][0]-.36), (d, y+.36))
    label(ax, d, 3.59, "Each position attends only to itself and earlier positions.", 9.5, color=GRAY)

    # Both protocols reuse panel A; neither is another trainable decoder.
    label(ax, .45, 2.80, "Two uses of the same forward network", 12, "bold", ha="left")
    label(ax, .52, 2.12, "TRAIN", 10, "bold", color=GRAY, ha="left")
    box(ax, 3.85, 2.12, "All logits + shifted targets", "[B,T,V] + [B,T]", w=3.75, color=AMBER)
    box(ax, 8.43, 2.12, "Cross-entropy (FP32)", "Flatten B×T positions; average token loss", w=4.25, color=AMBER)
    box(ax, 13.49, 2.12, "Scalar loss", "Backward → optimizer updates weights", w=3.75, color=AMBER)
    arrow(ax, (5.76, 2.12), (6.27, 2.12))
    arrow(ax, (10.60, 2.12), (11.56, 2.12))
    label(ax, .52, 1.06, "GENERATE", 10, "bold", color=GRAY, ha="left")
    generation = [(3.35, "Last-position logits", "[B,V]"),
                  (7.12, "Temperature → top-k → softmax", "Probabilities: [B,V]"),
                  (10.75, "Sample one token", "[B,1]"),
                  (14.03, "Append and repeat", f"Crop input to last ≤{c.block_size} tokens")]
    widths = [2.65, 4.10, 2.60, 3.15]
    for i, ((x, title, detail), w) in enumerate(zip(generation, widths)):
        box(ax, x, 1.06, title, detail, w=w, color=TEAL, size=10)
        if i:
            prev_x = generation[i-1][0]
            arrow(ax, (prev_x+widths[i-1]/2+.03, 1.06), (x-w/2-.03, 1.06))
    label(ax, 8, .29, "B: batch size · T: current input length · C: model width · V: vocabulary size · H: heads · d: head width. "
          f"Dropout p={c.dropout} during training; disabled for generation. Stop on all-row EOS or length limit.", 9, color=GRAY)
    save(fig, "minigpt_architecture_detailed")


def compact(c, count):
    fig, ax = canvas(12, 8.3)
    label(ax, .40, 7.91, "MiniGPT: decoder-only Transformer", 19, "bold", ha="left")
    label(ax, .40, 7.51, f"{c.n_layer} blocks · {c.n_head} heads · C={c.n_embd} · V={c.vocab_size:,} · "
          f"context ≤{c.block_size} · {count:,} parameters", 10.5, color=GRAY, ha="left")
    panel(ax, .35, 1.80, 4.40, 5.32, "Forward network")
    panel(ax, 5.15, 1.80, 6.50, 5.32, "One Transformer block (pre-norm)")
    a, b = 2.55, 8.45
    items = [(6.19, "Token + learned position embeddings", "IDs [B,T], positions [T] → [B,T,C]", BLUE),
             (5.23, f"Dropout → Transformer block × {c.n_layer}", "[B,T,C] → [B,T,C]", TEAL),
             (4.27, "Final LayerNorm", "[B,T,C]", BLUE),
             (3.31, "LM head (tied to token embedding)", "Linear(C,V), no bias", BLUE),
             (2.35, "Logits", "[B,T,V]", AMBER)]
    for i, (y, title, detail, color) in enumerate(items):
        box(ax, a, y, title, detail, w=3.92, color=color, size=10)
        if i:
            arrow(ax, (a, items[i-1][0]-.36), (a, y+.36))
    label(ax, b, 6.48, "x₀: [B,T,C]", 11, "bold")
    box(ax, b, 5.78, "LayerNorm₁ → causal multi-head attention", f"{c.n_head} heads, d={c.n_embd//c.n_head}; future positions masked", w=5.22, color=TEAL, size=10.5)
    add(ax, b, 4.96)
    box(ax, b, 4.09, "LayerNorm₂ → token-wise MLP", f"Linear({c.n_embd},{4*c.n_embd}) → GELU → Linear({4*c.n_embd},{c.n_embd}) → dropout", w=5.22, color=TEAL, size=9.8)
    add(ax, b, 3.23)
    label(ax, b, 2.45, "x₂: [B,T,C]", 11, "bold")
    arrow(ax, (b, 6.26), (b, 6.14))
    arrow(ax, (b, 5.42), (b, 5.14))
    arrow(ax, (b, 4.78), (b, 4.45))
    arrow(ax, (b, 3.73), (b, 3.41))
    arrow(ax, (b, 3.05), (b, 2.67))
    path(ax, [(b, 6.23), (5.53, 6.23), (5.53, 4.96), (8.26, 4.96)])
    arrow(ax, (7.98, 4.96), (8.26, 4.96), color=RED)
    path(ax, [(b, 4.64), (11.35, 4.64), (11.35, 3.23), (8.64, 3.23)])
    arrow(ax, (8.92, 3.23), (8.64, 3.23), color=RED)
    label(ax, 6.03, 5.14, "x₀", 10, color=RED)
    label(ax, 10.87, 3.43, "x₁", 10, color=RED)
    arrow(ax, (4.57, 5.23), (5.03, 5.23), dashed=True)
    label(ax, .45, 1.34, "TRAIN", 10, "bold", color=GRAY, ha="left")
    label(ax, 2.02, 1.34, "All logits [B,T,V] + shifted targets [B,T] → cross-entropy → scalar loss", 11, ha="left")
    label(ax, .45, .85, "GENERATE", 10, "bold", color=GRAY, ha="left")
    label(ax, 2.02, .85, "Last logits [B,V] → temperature / top-k → softmax → sample [B,1] → append; repeat", 10.5, ha="left")
    label(ax, .45, .33, f"B: batch size · T≤{c.block_size}: input length · C={c.n_embd}: feature width · V={c.vocab_size}: vocabulary. "
          f"Dropout p={c.dropout}; disabled at inference.", 9.5, color=GRAY, ha="left")
    save(fig, "minigpt_architecture_report")


def main():
    config = yaml.safe_load((ROOT / "configs/baseline.yaml").read_text(encoding="utf-8"))
    c = GPTConfig(**config["model"])
    torch.set_num_threads(2)
    model = GPT(c).eval()
    # Verify the displayed endpoint shapes on a real, small forward pass.
    with torch.no_grad():
        tokens = torch.zeros((2, 8), dtype=torch.long)
        logits, loss = model(tokens, tokens)
        assert logits.shape == (2, 8, c.vocab_size) and loss.ndim == 0
    count = model.parameter_count()
    detailed(c, count)
    compact(c, count)
    print(f"Verified logits {tuple(logits.shape)}, scalar loss, parameters {count:,}")
    print(f"Figures saved to {OUT}")


if __name__ == "__main__":
    main()
