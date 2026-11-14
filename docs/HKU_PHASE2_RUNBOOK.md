# HKU GPU Farm Phase 2: next-session handoff

Prepared on 2026-11-14 (Hong Kong time), initially before server connection.
Updated after completion: Phase 2 CUDA/BF16 smoke and all three core runs have
passed; complete results are backed up and verified locally. See EXPERIMENTS.md.
The commands below remain a reproduction guide, not instructions to rerun completed work.

## Local readiness

- All four model TODOs are implemented, with explicit manual causal attention.
- Full local test suite: 44 passed. `pip check`: no broken requirements.
- Fixed official data downloaded; archive and per-file SHA-256 verified.
- `python scripts/check_setup.py`: setup_pass; baseline has 13,891,584 parameters.
- Baseline YAML and requirements match the preserved upstream SHA-256 values.
- Shell launchers have LF line endings.
- CPU smoke results: `outputs/local_cpu_smoke_20261114/`.
- Smoke completed 20 updates / 10,240 target tokens, saved and reloaded checkpoints,
  generated three continuations, plotted loss, and evaluated 1,024 validation tokens.
- Smoke uses the smaller smoke model, not the full baseline. CPU execution is FP32
  even though the unchanged smoke config records `precision: bfloat16`.
- This is a workflow check, not a completed assignment run or a full-split evaluation.
- Remote checks subsequently passed on an RTX 4080 SUPER with Python 3.12.16,
  PyTorch 2.10.0+cu130 and CUDA runtime 13.0. CUDA smoke, baseline and both
  controlled experiments completed; their source/config/checkpoint hashes were verified.

## Connection and allocation

Sources (read on 2026-11-14):
- https://ai.hku.hk/index.php/gpu-farm/quickstart
- https://ai.hku.hk/index.php/gpu-farm/advanced-use

Use HKUVPN (also on HKU Wi-Fi), or the CDS wired network. Phase 2 uses
`gpu2gate1.cs.hku.hk` / `gpu2gate2.cs.hku.hk`; account name is the Portal ID.
The Phase 2 password is the separate password set for that account.

In the LOCAL VS Code PowerShell terminal, after replacing the username:

```powershell
ssh YOUR_PORTAL_ID@gpu2gate1.cs.hku.hk
```

On the REMOTE gateway, use tmux before allocating a GPU:

```bash
tmux new -s dasc_2a
gpu-interactive
hostname
nvidia-smi
```

`gpu-interactive` normally allocates 1 GPU, 4 CPU cores and 40 GB system RAM
for up to 6 hours. Verify the actual allocated GPU; do not assume the model
from an example hostname. GPU quota is consumed while allocated, including idle time.
Gateway login alone does not allocate a GPU. Install/run GPU software on an
allocated compute node. Exiting the allocation terminates its processes.

For a second terminal into the SAME allocation, run `hostname -I` on the compute
node, then use the following from a new LOCAL terminal (replace user/IP):

```powershell
ssh -J YOUR_PORTAL_ID@gpu2gate1.cs.hku.hk YOUR_PORTAL_ID@10.X.X.X
```

Keep the original allocation alive; do not request another GPU just for a second
terminal. To detach tmux use Ctrl+B then D; reconnect to the same gateway and run
`tmux attach -t dasc_2a`. tmux belongs on the gateway, not inside the GPU allocation.

## Transfer the local implementation

Local training bundle:
`work/server_upload/dasc7606_2a_server_20261114.zip`.
Its top-level directory is `dasc7606_2a/`. It contains source, tests, configs,
fixed data, supporting documentation and a file-hash manifest.
It excludes Windows environments, `.env`, `.git`, `.vscode`, caches and checkpoints.
This is a server transfer bundle, NOT the final Moodle submission ZIP.

The first transfer used this bundle before committing the implementation to Git.
It is a historical snapshot and does not contain later experiment YAMLs or updated
results documentation. Use the latest Git revision or sync the actual working tree
for future transfers. Determine the destination and check for existing files before extraction.
Do not copy the Windows Conda environment to Linux.

## Remote preflight, before the formal baseline

1. Inspect available Conda/Python, driver, GPU, account quota and destination paths.
2. Use a Linux Python 3.12/3.13 environment. Install CUDA-enabled PyTorch 2.10.0
   matching the ACTUAL driver; select the build after inspecting the server.
   Keep `requirements.txt` and `configs/baseline.yaml` unchanged.
3. Confirm `torch.cuda.is_available()`, actual GPU name, and
   `torch.cuda.is_bf16_supported()`. If BF16 cannot run, investigate/contact TA
   instead of silently changing the baseline's precision.
4. Run `python -m pip check`, `python scripts/check_setup.py`, and `python -m pytest -q`.
5. Run a CUDA smoke test in a fresh directory, then sample and evaluate its checkpoint.
6. Launch the unchanged baseline only after those checks pass.

Baseline command from the REMOTE project root with the correct environment active:

```bash
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda
```

For an interrupted baseline only, resume with the same config/output path:

```bash
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda --resume
```

The formal baseline is 2,500 updates / 10,240,000 target tokens. Start it from
random initialization; do not initialize it from the local smoke checkpoint.
New runs require empty output directories.

For unattended training, the documented `batch` partition supports `sbatch`
jobs (up to 7 days). Prepare the job script after confirming Conda/project paths,
with 1 GPU and 4 CPU cores; do not submit duplicate interactive/batch training.
Batch jobs can be monitored with `squeue -u "$USER"`. Allocation and runtime
settings must be checked on the actual cluster before submission.

## After each formal run

```bash
python scripts/plot_run.py outputs/baseline
python -m src.sample --checkpoint outputs/baseline/best.pt --out outputs/baseline/samples.json --device cuda
python -m src.evaluate --checkpoint outputs/baseline/best.pt --tokens data/val.bin --max-tokens 0 --out outputs/baseline/evaluation.json --device cuda
```

All three core runs and their evidence/checkpoints are backed up locally.
The experiment configs are configs/model_depth4.yaml and configs/optimizer_lr3e4.yaml;
each changes one factor relative to the baseline at the same token budget.
