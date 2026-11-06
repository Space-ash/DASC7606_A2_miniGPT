# 实验记录

先填写假设，再运行。以下为模板，尚未确定实验改动，也没有实验结果。

## 运行前

| 项目 | Baseline | 模型侧实验 | 优化侧实验 |
|---|---|---|---|
| 配置 | configs/baseline.yaml | 待创建 | 待创建 |
| 唯一主要改动 | 无 | 待定 | 待定 |
| 假设与依据 | 官方参照 | 待填写 | 待填写 |
| seed | 1337 | 1337 | 1337 |
| target token budget | 10,240,000 | 10,240,000 | 10,240,000 |
| 初始化 | 随机 | 随机 | 随机 |
| 运行目录 | outputs/baseline | outputs/model_experiment | outputs/optimizer_experiment |

## 运行后

| 项目 | Baseline | 模型侧实验 | 优化侧实验 |
|---|---|---|---|
| 参数量 | 待运行 | 待运行 | 待运行 |
| 完成 updates / tokens | 待运行 | 待运行 | 待运行 |
| 最佳 checkpoint step | 待运行 | 待运行 | 待运行 |
| Trainer sampled validation loss | 待运行 | 待运行 | 待运行 |
| Full FP32 validation loss | 待运行 | 待运行 | 待运行 |
| Full FP32 perplexity | 待运行 | 待运行 | 待运行 |
| 运行时间 / 硬件 | 待运行 | 待运行 | 待运行 |
| 实际命令 / Git commit | 待记录 | 待记录 | 待记录 |

## 分析待填

- 曲线、收敛与假设是否得到支持：
- 统一提示词与采样设置下的代表性续写：
- 重复、情节矛盾、人物一致性和 EOS 等失败案例：
- 相同 token budget 下的性能与计算成本权衡：
- 单 seed 等局限：
- 最终模型选择及公开验证集依据：
