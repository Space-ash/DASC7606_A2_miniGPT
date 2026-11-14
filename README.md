# MiniGPT: causal language modelling and FFN experiments

A student implementation for HKU DASC7606 Assignment 2. The project implements
explicit causal multi-head attention, a GELU feed-forward network, pre-norm
residual blocks and autoregressive sampling. The optional extension compares
GELU, SiLU and a near-parameter-matched SwiGLU feed-forward network.

## Project structure

- `src/`: core model, training, sampling and full-validation evaluation.
- `configs/`: baseline, four-layer and lower-learning-rate experiments, plus smoke settings.
- `tests/`: public implementation and packaging checks.
- `scripts/`: data download, setup checks, plotting and submission utilities.
- `extension/code/`: independent FFN extension and its tests.
- `extension/configs/`: three FFN variants, three training seeds and smoke settings.
- `data/`: data-license notices only; download the fixed data separately.

Reports, experimental evidence, model weights, submission packages, notebooks,
editor settings and personal working notes are excluded from the current Git
tree. Submission materials are prepared locally.

## Setup and core workflow

Use Python 3.12 or 3.13 and install a PyTorch build compatible with the target
CPU/CUDA environment. The requirements pin the tested package versions.

```sh
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/check_setup.py
python -m pytest -q tests
```

The downloader verifies the unchanged official v1.0.0 data archive and file
checksums. Keep the supplied tokenizer and data splits unchanged.
Formal training uses CUDA with BF16 autocast; CPU supports local checks.

```sh
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda
python -m src.train --config configs/model_depth4.yaml --out_dir outputs/model_depth4 --device cuda
python -m src.train --config configs/optimizer_lr3e4.yaml --out_dir outputs/optimizer_lr3e4 --device cuda
python scripts/plot_run.py outputs/baseline
python -m src.evaluate --checkpoint outputs/baseline/best.pt --tokens data/val.bin --max-tokens 0 --out outputs/baseline/evaluation.json --device cuda
python -m src.sample --checkpoint outputs/baseline/best.pt --tokenizer data/tokenizer.json --out outputs/baseline/samples.json --device cuda --seed 2026 --temperature 0.8 --top-k 40 --max-new-tokens 160
```

Choose a new output directory for each fresh run. Each formal configuration uses
2,500 updates and 10,240,000 processed target tokens. Compare interventions
against the unchanged baseline with matching budgets and evaluation protocols.

## Independent FFN extension

GELU and SiLU use hidden width 1536; SwiGLU uses width 1024 to match the principal
projection parameter budget. Configurations cover seeds 1337, 2026 and 3407.

```sh
python -m pytest -q tests extension/code/tests
python extension/code/train.py --config extension/configs/swiglu_seed1337.yaml --data-dir data --out_dir outputs/extension/swiglu_seed1337 --device cuda
python extension/code/plot_run.py outputs/extension/swiglu_seed1337
python extension/code/evaluate.py --checkpoint outputs/extension/swiglu_seed1337/best.pt --tokens data/val.bin --tokenizer data/tokenizer.json --max-tokens 0 --out outputs/extension/swiglu_seed1337/evaluation.json --device cuda
python extension/code/sample.py --checkpoint outputs/extension/swiglu_seed1337/best.pt --tokenizer data/tokenizer.json --out outputs/extension/swiglu_seed1337/samples.json --device cuda --seed 2026 --temperature 0.8 --top-k 40 --max-new-tokens 160
```

Use the extension loader for SwiGLU weights. The extension does not replace
either required core experiment or the core-model checkpoint.

## Attribution

Adapted from Andrej Karpathy's MIT-licensed
[nanoGPT](https://github.com/karpathy/nanoGPT) and the course starter code.
The extension implements SwiGLU following
[Shazeer (2020)](https://arxiv.org/abs/2002.05202), retaining course-model biases
and output dropout. See `LICENSE`, `extension/code/NOTICE.md` and the data-license
notices for attribution and usage terms.
