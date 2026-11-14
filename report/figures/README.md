# MiniGPT architecture figures

## Baseline learning curves

`baseline_training_diagnostics.png` / `.svg` show real baseline train/validation
split losses over the entire run and a late-phase zoom. Both curves come from
`eval.jsonl`: fixed sampled windows, BF16 autocast, model in eval mode (dropout
disabled). These are separate from full-split FP32 CE/PPL. No smoothing or
synthetic results are used. Regenerate with `python scripts/plot_baseline_diagnostics.py`.
The associated machine-readable result record is `report/baseline_results.json`.

Suggested caption: **Baseline learning curves.** Cross-entropy on sampled
training and validation windows, evaluated in BF16 with dropout disabled, over
10.24 million processed target tokens. The right panel magnifies the late
training phase; the vertical scales differ. Validation loss decreases at all
recorded checkpoints. Full-split FP32 validation yields CE 2.2994 and PPL 9.9679.

## Architecture diagrams

Figures reflect `src/model.py` and the unchanged `configs/baseline.yaml`.
They describe the architecture, not training results. Labels are in English for report use.

| Figure | Purpose | PNG dimensions | Other format |
|---|---|---|---|
| `minigpt_architecture_report.png` | Compact report figure: full network, residual block, training/generation | 3600 × 2490, 300 dpi | `minigpt_architecture_report.svg` |
| `minigpt_architecture_detailed.png` | Detailed study figure: embeddings, full network, block, attention, tensor shapes | 4800 × 3750, 300 dpi | `minigpt_architecture_detailed.svg` |

Use SVG for scaling without raster blur where supported; use PNG otherwise.
SVG labels remain text for editing. Fonts may depend on the viewing application.
Solid arrows show computation, red solid bypasses are residual connections,
red dashed lines indicate shared weights, and gray dashed arrows expand a module.

## Notation and baseline

- B: batch size (16 in baseline training).
- T: current input length, at most 256 per model call.
- C: hidden width, 384.
- V: vocabulary size, 8192.
- H: attention heads, 6; d = C/H = 64.
- Six blocks have separate parameters. Token embeddings and the LM head share one weight matrix of shape [V,C].
- Total unique trainable parameters: 13,891,584.
- Dropout probability: 0.1 in training; disabled in evaluation/generation.
- Token and position IDs are looked up separately, then the embedding vectors are added; IDs themselves are not added.
- Attention and MLP preserve [B,T,C]; the MLP temporarily expands to [B,T,1536].
- The forward network returns raw logits [B,T,V]. Softmax for sampling is applied outside that forward path.
- Training targets are shifted by one token. Cross-entropy receives all positions and returns a scalar mean loss.
- Generation crops only the next model input; it preserves the original prompt in the returned sequence.

## Suggested report caption

> **MiniGPT architecture.** Token embeddings and learned positional embeddings are added and passed through six pre-normalized Transformer blocks. Each block contains six-head causal self-attention and a position-wise MLP (384–1536–384), each wrapped in a residual connection. A final LayerNorm and an output projection tied to the token embedding matrix produce logits over 8,192 tokens. Training uses next-token cross-entropy; autoregressive generation samples from the last-position logits using temperature and optional top-k filtering. B denotes batch size and T denotes input length (at most 256).

## Regenerate

From the project root in the local debugging environment:

```powershell
conda activate dasc_2a
python scripts/draw_architecture.py
```

The script checks baseline output shapes using a small real forward pass before rendering.
It requires no dataset or trained checkpoint and does not modify model/configuration files.
