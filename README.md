# DASC7606_A2_miniGPT

DASC7606 Assignment 2 的个人工作区，已同步官方 v1.1.1（2026-11-13）说明与扩展工具；核心训练器内部版本仍为 v1.0.0。

**本项目报告仍保留 4–5 页，按所有者于 2026-11-13 再次确认的要求。** 最新公开 v1.1.1 仍规定主报告最多 2 页，两者存在冲突；这里不把 4–5 页描述为教师公开规定，最终请按 Moodle/教师确认的要求提交。官方原文保留在 `docs/UPSTREAM_README.md`、`RUBRIC.md` 和更新后的 PDF。

当前状态（2026-11-14）：`src/model.py` 的四个 TODO 已完成，44 项测试、CPU/CUDA smoke run 均通过。Baseline、4 层模型实验及较低学习率实验已在 HKU GPU Farm 完成，各训练 2,500 步 / 10,240,000 target tokens，并生成曲线、固定提示词续写和完整验证集 FP32 评估。本地备份已核对配置、源码、数据和 checkpoint 哈希。报告与最终提交包尚未完成。

训练环境：NVIDIA GeForce RTX 4080 SUPER；Python 3.12.16 / PyTorch 2.10.0+cu130 / CUDA runtime 13.0；BF16 训练。本地 Windows 调试使用 Conda `dasc_2a`、Python 3.12.16 和 PyTorch 2.10.0+cpu。模型权重和完整 run 目录仅保留在本地/服务器，不进入 Git。

| Run | 唯一主要改动 | 参数量 | Full FP32 CE | Full FP32 PPL |
|---|---|---:|---:|---:|
| baseline | 原始配置 | 13,891,584 | 2.299374 | 9.967941 |
| model_depth4 | n_layer: 6 → 4 | 10,342,656 | 2.344976 | 10.433019 |
| optimizer_lr3e4 | learning_rate: 0.0006 → 0.0003 | 13,891,584 | 2.442079 | 11.496918 |

三组均在完整的 389,048 个公开验证目标 token 上评估。Baseline 当前具有最低的完整验证集 CE；单 seed 结果不支持统计显著性结论。实验假设、运行时间、显存、协议区别及命令见 [实验记录](docs/EXPERIMENTS.md)。网络结构图和 baseline 曲线的 PNG/SVG 见 [报告图片说明](report/figures/README.md)。

## 工作区使用

### Notebook 交互入口

可直接打开 [MiniGPT 实验工作台](notebooks/Assignment2_Workbench.ipynb)，按中文说明联动现有脚本，后续在 GPU 上运行后展示学习曲线、学习率、单步吞吐、三组完整 FP32 评估和固定提示词续写。
Notebook 目前未执行，所有安装、下载、测试、训练和打包开关默认关闭；上述实际训练通过 CLI 完成，模型的四个 TODO 已实现。使用方法见 [Notebook 说明](notebooks/README.md)。

```sh
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
# 在 VS Code 中选择该 Python 环境作为 Notebook 内核。
```

Notebook 是辅助工具；正式提交仍按官方脚本打包代码、单一 checkpoint、三组证据和 4–5 页 PDF。

- `src/`：模型、训练、生成、评估；从 `src/model.py` 开始。
- `configs/`：原始 baseline、smoke 及两组单因素实验配置；baseline 保持不变。
- `scripts/`、`tests/`：官方脚本与公开测试；新增 `scripts/download_data.py` 用于固定数据下载与校验。
- `data/`：本地固定数据；二进制文件、分词器、manifest 不上传 Git。
- `report/`：报告源文件；最终导出到根目录 `report.pdf`。
- `outputs/`：训练结果与 checkpoint；`work/`：临时下载与日志；均不上传 Git。
- 本地 Python 环境不上传 Git。当前 Windows VS Code 配置使用 Conda `dasc_2a`；其他机器按实际环境选择解释器。

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
GPU 机器应单独安装与驱动兼容的 PyTorch CUDA 构建；CPU 可用于本地测试，官方脚本不支持 MPS。

以下为三组实际训练的复现命令（在已配置的 CUDA 环境执行；每个新 run 需要空的输出目录）。两组实验分别相对原始 baseline 只修改一个字段：

```sh
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda
python -m src.train --config configs/model_depth4.yaml --out_dir outputs/model_depth4 --device cuda
python -m src.train --config configs/optimizer_lr3e4.yaml --out_dir outputs/optimizer_lr3e4 --device cuda
```

- [2026-11-13 更新对照](docs/UPSTREAM_UPDATE_2026-11-13.md)
- [最新评分细则](RUBRIC.md)
- [扩展任务与独立提交](EXTENSION.md)
- [作业要求与进度](docs/ASSIGNMENT_NOTES.md)
- [4–5 页报告提纲](docs/REPORT_OUTLINE.md)
- [官方 PDF](docs/DASC7606_Assignment_2.pdf)
- [官方 README 原件](docs/UPSTREAM_README.md)（v1.1.1 原文，含公开的两页规定）

提交前在本 README 中补充实际环境、三个 run 的精确命令与最终 checkpoint 选择理由。


## 2026-11-13 同步要点

- 核心四部分（实现、训练/推理、受控实验、报告）各 20 分，加扩展探索 20 分，共 100 分；不做扩展最高为 80/100。
- 扩展按动机、实现、实验证据、分析各 5 分评分。负结果也可以获得满分；没有固定 loss/CE/PPL 阈值或分数档位。
- 合规的 Colab T4/BF16 结果可接受，不要求原生 BF16 硬件支持，不需换 GPU 重跑。保持 baseline 不变；若官方训练器不能执行 BF16，应先联系 TA。
- 截止前允许修订/重新提交；已锁定的尝试联系 TA 重开。截止仍为 2026-11-22 23:59 HKT。
- 核心代码、数据、tokenizer、baseline、三次 run 和核心 ZIP 格式不变。旧版合规结果继续有效，不因本次更新重训。
- 扩展另交 `STUDENT_ID_extension.zip`，含独立 `extension_report.pdf`、README、code、configs、results；不能替代核心 ZIP，每个压缩包各不超过 100 MB。

```sh
python scripts/package_extension.py --directory extension --student-id YOUR_STUDENT_ID --out YOUR_STUDENT_ID_extension.zip
python scripts/package_extension.py --check YOUR_STUDENT_ID_extension.zip
```

Notebook 已加入扩展说明与默认关闭的打包入口，现有训练开关继续默认关闭。核心检查器和模型实现没有被本次同步替换。

---

以下为最新官方 v1.1.1 运行说明原文（其中主报告两页规定与本项目保留的 4–5 页要求有冲突，见顶部说明）。

**Assignment released 29 October 2026. Due Sunday, 22 November 2026,
23:59 Hong Kong time (UTC+8).**

**Updated 13 November 2026 (v1.1.1).** Four core parts are worth **20 marks each**:
implementation, training/inference, controlled experiments, and report writing.
An additional **20 marks for extension/exploration** are available within Part 3.
The assignment total is **100, not 120**. See [the full rubric and update notice](RUBRIC.md).

**Clarification:** completed Colab T4/BF16 runs that meet the assignment
requirements are accepted without rerunning on another GPU. There are **no fixed
loss, cross-entropy or perplexity thresholds for grades**. Assessment focuses on
implementation correctness and the quality of the experiments and analysis.

Submit your checked core ZIP (`STUDENT_ID.zip`) and, if attempting the extension,
a separate `STUDENT_ID_extension.zip`. Each file must be at most 100 MB.
The core report remains at most two pages; put extension analysis in a separate
`extension_report.pdf`. Already completed compliant core runs remain valid: this
update does not require retraining or repackaging a valid core submission.

Revision/resubmission before the deadline is permitted, including for students
who have already submitted. If Moodle has locked your submitted attempt, contact
the TA to reopen it. A saved draft is not a final submission: verify **Submitted
for grading** after your final upload. Keep your core ZIP when adding the extension.

Late penalty: up to 1 day 10%; up to 2 days 20%; up to 7 days 50%; more than
7 days 100%. Late submissions close on 29 November 2026 at 23:59 HKT.
Consult Moodle and the released assignment slides for the authoritative handout.

This repository contains only student starter code, public tests and utilities.
The fixed data bundle is available from the versioned release below. No teacher solutions,
reference weights, hidden evaluation data or private grading materials are included.

## Downloads (v1.1.1; fixed data unchanged)

- [Starter code ZIP](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.1.1/assignment2-starter-v1.1.1.zip)
- [Fixed data ZIP](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.0.0/assignment2-data-v1.0.0.zip)
- [SHA-256 checksums](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.1.1/SHA256SUMS.txt)
- [Updated assignment slides](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/download/v1.1.1/DASC7606_Assignment_2_v1.1.1.pdf)
- [Release notes](https://github.com/ludunjie1219/DASC7606-Assignment2/releases/tag/v1.1.1)

Use the v1.1.1 starter with the unchanged v1.0.0 data ZIP. The GitHub-generated
source archives do not include data. If using Git, check out tag `v1.1.1` and
download the data ZIP separately. Do not overwrite your completed `src/model.py`
with the starter: existing v1.0.0 training code and core ZIPs remain accepted.

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
arbitrary architectures. Novel architectures belong in the separate extension
package, not the core checkpoint. See [EXTENSION.md](EXTENSION.md).

## 1. Environment

Use Python 3.12 or 3.13 in a dedicated environment. The teacher GPU reference was
tested with Python 3.13, PyTorch 2.10.0+cu130 and one A10G. This is not a claim of
compatibility with any university cluster image. CPU is supported for correctness
tests and the smoke run; the scripts currently support `cpu` and `cuda`, not MPS.

```sh
python3.13 -m venv .venv
source .venv/bin/activate
# Install PyTorch 2.10.0 for your hardware using the official selector:
# https://pytorch.org/get-started/locally/
python -m pip install -r requirements.txt
```

Environment installation may require Internet access. The training workflow does
not. The launch scripts do not install packages or allocate a server for you.
CUDA baseline uses BF16. **Native hardware BF16 support is not a grading
requirement. Colab T4/BF16 runs are accepted** when the released trainer works
with the required baseline configuration, fixed data/tokenizer and token budget,
and the actual environment and run evidence are recorded accurately. If you
already completed such runs, **you do not need to rerun them on another GPU**.
Public tests and format checks remain useful checks, not guarantees of a grade.
This clarification does not waive implementation or experiment requirements or
authorize changing the baseline configuration. If the trainer cannot execute
BF16 in your environment, contact the TA before changing its precision setting.

## 2. Obtain and verify the fixed data

Download the two ZIP files and `SHA256SUMS.txt` linked above into one directory.
Verify both downloads before extracting them:

```sh
# Linux:
sha256sum -c SHA256SUMS.txt
# macOS alternative:
shasum -a 256 -c SHA256SUMS.txt

unzip assignment2-starter-v1.1.1.zip
cd assignment2-starter-v1.1.1
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
Each required experiment changes one factor relative to the original baseline,
not relative to the other experiment. Record a hypothesis, compare validation
results and fixed-prompt continuations, and discuss failures. Additional runs,
combined changes and recent methods may be explored for the extension marks;
there is no maximum number of experiments. Quality matters more than quantity.

Train each config into its own directory, then run plotting, sampling and public
evaluation as above. Select your final checkpoint using public validation only.
Do not treat grammatical sentences as proof of narrative consistency.

## 6. Prepare a submission

Create `report.pdf` (up to two pages): implementation/method, controlled experiment
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
