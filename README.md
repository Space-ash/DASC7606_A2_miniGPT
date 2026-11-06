# DASC7606_A2_miniGPT

DASC7606 Assignment 2 的个人工作区，基于官方 v1.0.0 student starter。

**报告页数采用项目所有者提供的更新：4–5 页。** 官方原始 PDF 和 README 的两页规定是旧要求。

当前状态：保留 `src/model.py` 的四个 TODO，尚未实现或训练模型。先完成 TODO，再运行测试与 smoke run。

初始化检查（2026-11-06，Mac 本地 CPU）：已创建 `.venv`，Python 3.12.14 / PyTorch 2.10.0；`pip check` 与官方 `scripts/check_setup.py` 均通过，baseline 构造参数量为 13,891,584。此检查验证环境与固定数据，不代表 TODO 正确性测试通过；尚未运行公开模型测试、smoke training 或正式训练。CUDA 训练环境需在对应 GPU 机器另行配置。

## 工作区使用

- `src/`：模型、训练、生成、评估；从 `src/model.py` 开始。
- `configs/`：官方 baseline 与 smoke 配置；baseline 保持不变。
- `scripts/`、`tests/`：官方脚本与公开测试；新增 `scripts/download_data.py` 用于固定数据下载与校验。
- `data/`：本地固定数据；二进制文件、分词器、manifest 不上传 Git。
- `report/`：报告源文件；最终导出到根目录 `report.pdf`。
- `outputs/`：训练结果与 checkpoint；`work/`：临时下载与日志；均不上传 Git。
- `.venv/`：本地 Python 环境，不上传 Git。VS Code 默认使用该环境。

本地 Python 3.12 环境激活后执行：

```sh
source .venv/bin/activate
python scripts/check_setup.py
```

重新克隆仓库后，用 Python 3.12/3.13 建立环境、安装下方依赖，再运行：

```sh
python scripts/download_data.py
python scripts/check_setup.py
```

下载脚本固定使用官方 v1.0.0 数据，验证归档和数据文件 SHA-256；遇到已有文件被修改时会停止。
GPU 机器应单独安装匹配硬件的 PyTorch CUDA 构建；本地 Mac 使用 CPU，官方脚本不支持 MPS。

先在 [实验记录表](docs/EXPERIMENTS.md) 写下两个实验的假设，再复制 baseline 配置分别创建实验 YAML。示例：

```sh
# 创建相应配置之后运行；两个实验均与原始 baseline 比较。
python -m src.train --config configs/model_experiment.yaml --out_dir outputs/model_experiment --device cuda
python -m src.train --config configs/optimizer_experiment.yaml --out_dir outputs/optimizer_experiment --device cuda
```

- [作业要求与进度](docs/ASSIGNMENT_NOTES.md)
- [4–5 页报告提纲](docs/REPORT_OUTLINE.md)
- [官方 PDF](docs/DASC7606_Assignment_2.pdf)
- [官方 README 原件](docs/UPSTREAM_README.md)（含旧页数规定）

提交前在本 README 中补充实际环境、三个 run 的精确命令与最终 checkpoint 选择理由。

---

以下为基于官方 README 的运行说明，报告长度已按上述更新调整。

**Assignment released 29 October 2026. Due Sunday, 22 November 2026,
23:59 Hong Kong time (UTC+8).**

Submit one checked ZIP (maximum 100 MB) through your course's Moodle assignment.
There is one final submission attempt: saving a draft is not submitting it.
Check all files before clicking **Submit assignment**.

The marking split follows Assignment 1: **code and model performance 80 marks;
report and reproducibility 20 marks**. Code/model assessment covers correctness,
the baseline and controlled runs, reproducible training evidence, and performance
on unseen text. The report must explain methods, comparisons, curves, generations
and limitations in 4–5 pages (updated requirement provided by the project owner).

Late penalty: up to 1 day 10%; up to 2 days 20%; up to 7 days 50%; more than
7 days 100%. Late submissions close on 29 November 2026 at 23:59 HKT.
Consult Moodle and the released assignment slides for the authoritative handout.

This repository contains only student starter code, public tests and utilities.
The fixed data bundle is available from the versioned release below. No teacher solutions,
reference weights, hidden evaluation data or private grading materials are included.

## Downloads (v1.0.0)

- [Starter code ZIP](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/assignment2-starter-v1.0.0.zip)
- [Fixed data ZIP](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/assignment2-data-v1.0.0.zip)
- [SHA-256 checksums](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/SHA256SUMS.txt)
- [Assignment slides](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/DASC7606_Assignment_2.pdf)
- [Release notes](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/tag/v1.0.0)

Use both ZIP files from this version. The GitHub-generated source archives do not
include the data bundle. If using Git instead, check out tag `v1.0.0` and download
the data ZIP separately.

## What you will do

Complete four methods in `src/model.py`, train a small decoder-only Transformer
from random initialization, generate short-story continuations, and analyze
controlled experiments. No pretrained weights are provided or required.

1. `CausalSelfAttention.forward`: explicit masked multi-head attention.
2. `MLP.forward`: expansion, GELU, projection and dropout.
3. `Block.forward`: pre-normalization residual connections.
4. `GPT.generate`: context cropping, temperature/top-k sampling and EOS handling.

The provided embedding layers, initialization, optimizer, training loop, checkpoint
format and data pipeline are complete. Do not replace attention with a built-in
complete attention module or use a complete pretrained model. Preserve public
class/method signatures and checkpoint key names. The current evaluator supports
this GPT architecture family with configurable depth, width and head count, not
arbitrary architectures; discuss changes outside that family with the TA first.

## 1. Environment

Use Python 3.12 or 3.13 in a dedicated environment. The teacher GPU reference was
tested with Python 3.13, PyTorch 2.10.0+cu130 and one A10G. This is not a claim of
compatibility with any university cluster image. CPU is supported for correctness
tests and the smoke run; the scripts currently support `cpu` and `cuda`, not MPS.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
# Install PyTorch 2.10.0 for your hardware using the official selector:
# https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

Environment installation may require Internet access. The training workflow does
not. The launch scripts do not install packages or allocate a server for you.
CUDA baseline uses BF16; GPUs without BF16 need a separately checked configuration.

## 2. Obtain and verify the fixed data

Download the two ZIP files and `SHA256SUMS.txt` linked above into one directory.
Verify both downloads before extracting them:

```sh
# Linux:
sha256sum -c SHA256SUMS.txt
# macOS alternative:
shasum -a 256 -c SHA256SUMS.txt

unzip assignment2-starter-v1.0.0.zip
cd assignment2-starter-v1.0.0
unzip ../assignment2-data-v1.0.0.zip
```

If using a clone instead, extract the data ZIP at the root of that repository.
It contains one top-level `data/` directory:

```text
data/
  train.bin
  val.bin
  tokenizer.json
  manifest.json
  LICENSES.md
  CDLA-Sharing-1.0.html
```

Keep the tokenizer and data unchanged. They are a fixed TinyStories subset, with
a tokenizer fitted only on training stories. Do not regenerate or download other
training text. Never train on the validation set. No hidden test data are supplied.

```sh
python scripts/check_setup.py
```

This verifies dependencies, vocabulary, data hashes and model construction without
calling unfinished TODO methods. It does not prove your implementation is correct.

## 3. Implement, test and smoke-test

```sh
python -m pytest -q
bash scripts/run_smoke.sh cpu
# On a supported GPU, after completing the TODOs:
bash scripts/run_smoke.sh cuda
```

The fresh starter intentionally raises `NotImplementedError` and model tests fail
until the TODOs are completed. Public tests are examples, not exhaustive grading
tests. Smoke training runs only 20 updates and is not a baseline submission.
Every run refuses to overwrite a nonempty output directory. Use a new output path
for a new run, or explicitly resume an interrupted run with the same config.

## 4. Train the baseline

```sh
bash scripts/train_baseline.sh cuda
# Equivalent:
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda
# Resume only an interrupted run:
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda --resume
```

The baseline has 6 layers, 6 heads, width 384, context 256 and vocabulary 8192.
It trains for 2,500 optimizer updates, with 16 sequences per batch and no gradient
accumulation: 10,240,000 processed target tokens. Learning rate, seed and all other
baseline settings are specified in `configs/baseline.yaml`; keep that file unchanged.

`best.pt` is selected by public validation loss. `last.pt` contains optimizer/RNG
state for resuming and is not a submission artifact. Logs, metrics and the exact
configuration are written into each run directory. Timing is informational, not
a grading criterion. Fixed seeds do not guarantee identical results across GPUs.

```sh
python scripts/plot_run.py outputs/baseline
python -m src.sample --checkpoint outputs/baseline/best.pt --out outputs/baseline/samples.json
python -m src.evaluate --checkpoint outputs/baseline/best.pt --tokens data/val.bin \
  --max-tokens 0 --out outputs/baseline/evaluation.json
```

For CPU use, append `--device cpu` to sample/evaluate. The trainer reports sampled
mixed-precision validation loss; standalone evaluation above uses float32 over the
complete public split. Label these different protocols correctly in your report.

## 5. Controlled experiments

Complete a baseline, one model-side experiment and one
optimization-side experiment, each with the same processed-token budget. Copy the
baseline YAML and change one factor at a time. Keep vocabulary and context fixed;
stay within the 20M-parameter loader limit and the supported architecture family.
Record a hypothesis before running, compare validation results and fixed-prompt
continuations, and discuss failures. Extra experiments are optional.

Train each config into its own directory, then run plotting, sampling and public
evaluation as above. Select your final checkpoint using public validation only.
Do not treat grammatical sentences as proof of narrative consistency.

## 6. Prepare a submission

Create `report.pdf` (4–5 pages; updated requirement): implementation/method, controlled experiment
table and analysis, curves, representative continuations, limitations and
reproducibility. Include all code needed to reproduce the selected model.

```sh
python scripts/package_submission.py --student-id YOUR_STUDENT_ID \
  --baseline outputs/baseline --model-experiment outputs/your_model_run \
  --optimizer-experiment outputs/your_optimizer_run --select baseline \
  --report report.pdf --out submission.zip
bash scripts/check_submission.sh submission.zip
```

`--select` is `baseline`, `model_experiment` or `optimizer_experiment`. Replace
`submission.zip` with your student-ID filename when submitting. The tool includes
code under `src/` and `scripts/`, configs, tests, required run evidence, report and
one `best.pt`. Put any helper modules under `src/`; do not depend on external files.

```text
student-ID.zip
  submission.json
  report.pdf
  best.pt
  src/ ... scripts/ ... configs/ ... tests/ ...
  requirements.txt, LICENSE, README.md
  runs/baseline/ ...
  runs/model_experiment/ ...
  runs/optimizer_experiment/ ...
```

Do not submit data, virtual environments, caches, `last.pt` or other checkpoints.
The checker has a 200 MiB uncompressed package limit and 150 MiB checkpoint limit;
the separate Moodle limit is **100 MB for the final compressed ZIP**. Check its
file size before uploading. Keep only the selected checkpoint; reduce oversized
report images rather than removing required run evidence.
The format checker does not award a grade, verify your report's quality, prove
training provenance or replace the hidden correctness tests. The TA evaluator
must use trusted code and an isolated execution environment.

## Sources and license

Architecture adapted from [nanoGPT](https://github.com/karpathy/nanoGPT), commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, under the included MIT license.
Data: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), snapshot
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; retain the data package's source and
CDLA-Sharing-1.0 notices. Code and data have different licenses.
