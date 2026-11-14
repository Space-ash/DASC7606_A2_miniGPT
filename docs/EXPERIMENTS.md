# 实验记录

更新：2026-11-14（香港时间）。三组核心实验均已完成并备份到本地。以下假设在两组实验运行前确定，保留为预先设计记录。三组配置、预算、源码、checkpoint、采样及完整评估一致性检查均通过。

## 运行前设计

| 项目 | Baseline | 模型侧：减少层数 | 优化侧：降低峰值学习率 |
|---|---|---|---|
| 配置 | configs/baseline.yaml | configs/model_depth4.yaml | configs/optimizer_lr3e4.yaml |
| 唯一配置改动 | 无 | model.n_layer: 6 → 4 | training.learning_rate: 0.0006 → 0.0003 |
| 参数量 | 13,891,584 | 10,342,656（构造计算，减少 25.55%） | 13,891,584（构造计算） |
| seed | 1337 | 1337 | 1337 |
| target token budget | 10,240,000 | 10,240,000 | 10,240,000 |
| max_steps / batch / accumulation | 2500 / 16 / 1 | 相同 | 相同 |
| context / vocabulary | 256 / 8192 | 相同 | 相同 |
| min_lr / warmup_steps | 0.00006 / 100 | 相同 | 相同 |
| weight_decay / dropout | 0.1 / 0.1 | 相同 | 相同 |
| precision | BF16 | BF16 | BF16 |
| 初始化 | 随机 | 独立随机初始化 | 独立随机初始化 |
| 运行目录 | outputs/baseline | outputs/model_depth4 | outputs/optimizer_lr3e4 |

### 模型侧假设：深度与计算成本

问题：在相同数据、宽度、头数、上下文和 token 预算下，减少深度能否保留接近 baseline 的预测质量，同时降低运行时间和显存？

预先假设：4 层模型应具有更低的计算和存储需求，但表达能力可能下降，完整验证集 CE/PPL 可能变差，也可能因更容易优化而接近 baseline。实际测量时间、显存和质量，不按层数推算加速比例。

参数量下降是减少层数的直接后果，不是第二个独立因素。此比较固定 token 预算，不固定 FLOPs、参数量或训练秒数，不能分离纯深度效应与容量变化。

### 优化侧假设：较低峰值学习率的固定预算效果

问题：将峰值学习率减半，在相同 2500 步内是否改变收敛速度、完整验证集 loss 和生成表现？

Baseline 没有发散证据，不预设原学习率过高。较低峰值可能带来较平滑的更新，但也可能减慢固定预算内的学习，使最终 CE/PPL 变差；是否改善泛化由结果判断。

仅修改 learning_rate；min_lr 仍为 0.00006，warmup 仍为 100 步，调度算法不变。因此起始/峰值不同、最终下限相同，不把整条学习率曲线都描述为减半。

### 统一比较协议

- 两组实验各自从原始 baseline 复制，分别对照 baseline，不串联改动、不使用已有 checkpoint 初始化。
- 优先在同一 RTX 4080 SUPER allocation 中顺序运行，保持 Python 3.12.16 / PyTorch 2.10.0+cu130；环境如变化需记录。
- 固定训练 split、tokenizer、seed、batch、accumulation、context、词表、更新数及 BF16。相同 seed 不代表不同架构具有相同初始权重。
- best.pt 均按训练器的抽样验证 loss 选择；正式定量比较用完整验证集 FP32 CE/PPL（389,048 个目标 token）。
- 固定三条官方提示词，采样 seed=2026（每条依次 +0/+1/+2）、temperature=0.8、top_k=40、max_new_tokens=160，EOS 处理一致。
- 同时报告参数量、更新计算时间、训练器总运行时间、峰值显存。计时不含后处理、安装或传输。
- 单 seed 不支持统计显著性结论；相同 token 预算也不等于相同 FLOPs。

## Baseline 实际结果

已核对 8 项核心证据文件、best.pt、日志计数、配置及 6 个源码哈希。best.pt 与采样/评估的 SHA-256 一致，checkpoint 的数据/tokenizer 指纹与固定数据一致。

| 项目 | Baseline | 模型侧实验 | 优化侧实验 |
|---|---|---|---|
| 状态 | complete | complete | complete |
| 完成 updates / tokens | 2500 / 10,240,000 | 2500 / 10,240,000 | 2500 / 10,240,000 |
| 最佳 checkpoint step | 2500 | 2500 | 2500 |
| 最终 sampled train loss（eval 模式） | 2.295476 | 2.336726 | 2.439711 |
| 最佳/最终 sampled validation loss（BF16） | 2.233782 | 2.280669 | 2.374858 |
| 对应 sampled validation PPL | 9.335103 | 9.783226 | 10.749483 |
| Full FP32 validation CE | 2.299374 | 2.344976 | 2.442079 |
| Full FP32 validation PPL | 9.967941 | 10.433019 | 11.496918 |
| 完整评估目标 token 数 | 389,048 | 389,048 | 389,048 |
| 总训练运行时间 | 54.3983 s | 39.6823 s | 47.4575 s |
| 累计更新计算时间 | 46.9031 s | 33.6961 s | 41.0907 s |
| 峰值显存 | 1267.5483 MiB（约 1.238 GiB） | 1017.9849 MiB（约 0.994 GiB） | 1267.5483 MiB（约 1.238 GiB） |
| GPU | NVIDIA GeForce RTX 4080 SUPER | 相同 | 相同 |
| Python / torch / CUDA runtime | 3.12.16 / 2.10.0+cu130 / 13.0 | 相同 | 相同 |

runtime_seconds 包含循环内评估与 checkpoint 开销，不含脚本初始模型/数据准备及后处理；train_seconds 累加更新步骤计时。train.csv 每 50 步记录的 seconds 仅为当次更新耗时，不应求和作为总时间。

### 曲线与生成分析

- eval.jsonl 有 11 个评估点（0、250、…、2500），train.csv 有 50 条训练记录。
- 抽样验证 loss 从 9.095184 降至 2.233782，每个记录点都下降，末期下降速度放缓。最后 500 步仍改善 0.062068；未观察到验证曲线回升，但不证明继续训练一定有效或不存在过拟合。
- 抽样 train loss 从 9.096303 降至 2.295476。这条曲线也在 eval 模式下测量，不能用训练时 dropout 直接解释图中 val loss 低于 train loss。不同 split 内容/难度和采样窗口可能影响它们。
- train.csv 的 loss 是训练模式下当次 batch 的 loss；eval.jsonl 的 train/val loss 都在 eval 模式下对固定随机窗口求均值，不能混为同一统计量。
- Full FP32 CE=2.299374 与 sampled BF16 validation loss=2.233782 在数据覆盖、窗口组织和精度上不同，两者不直接作为性能升降比较。
- 女孩与龙：开头承接龙，随后转为云、鸟、猫，缺少过渡，并有 “The bird and the bird” 重复。
- Tom：维持朋友/帮助主题，但从帮助朋友转为向朋友求助，末尾 “sometimes, and friends” 表达不完整。
- Lily：shiny rock 被称作 new ball，Lily 对自己发出邀请，物体指代和说话者不清楚。
- 三条样本仅支持案例观察，不估计普遍错误率；后续逐条对照相同提示词，检查角色、物体、事件衔接、重复和结束情况。

## 精确命令

以下保留为三组已完成运行的复现命令。在服务器项目根目录、已激活 dasc_2a 且分配到 CUDA GPU 后执行；重现时需使用新的输出目录，避免覆盖本次证据。

```bash
# 已完成的 baseline
python -m src.train --config configs/baseline.yaml --out_dir outputs/baseline --device cuda

# 模型侧：独立随机初始化
python -m src.train --config configs/model_depth4.yaml --out_dir outputs/model_depth4 --device cuda

# 优化侧：独立随机初始化
python -m src.train --config configs/optimizer_lr3e4.yaml --out_dir outputs/optimizer_lr3e4 --device cuda
```

对每个新 run 分别后处理（替换 RUN_DIR）：

```bash
python scripts/plot_run.py RUN_DIR
python -m src.sample --checkpoint RUN_DIR/best.pt --out RUN_DIR/samples.json --device cuda --seed 2026 --temperature 0.8 --top-k 40 --max-new-tokens 160
python -m src.evaluate --checkpoint RUN_DIR/best.pt --tokens data/val.bin --max-tokens 0 --out RUN_DIR/evaluation.json --device cuda
```

代码来自本地工作区上传包，model.py SHA-256 为 `c8d8187a86fdedbf135a9d4b797da41725d20db5e4d4b31214449b2df2f388b2`。上传包没有 Git 历史，不虚构远端 commit；完整源码指纹见 outputs/baseline/environment.json。

## 可选扩展（尚未决定）

先完成三次核心 run。以上两项不重复作为扩展计分；如做扩展，另列研究问题、额外实现、匹配预算的证据及独立报告，参见 EXTENSION.md。
