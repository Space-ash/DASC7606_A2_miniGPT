# 作业要求与进度

来源：官方 v1.0.0；报告采用项目所有者于 2026-11-06 提供的更新：**4–5 页**。PDF 与官方原始 README 中的两页规定已过时。

## 固定要求

- 截止：2026-11-22 23:59 HKT；迟交关闭：2026-11-29 23:59 HKT。
- 评分：代码与模型表现 80 分；报告与可复现性 20 分。
- 仅使用给定训练 split 和固定 tokenizer；不能在验证集上训练或增加外部文本。
- 显式实现 attention；不能用 SDPA、MultiheadAttention 或完整预训练模型替代。
- 保留公开接口、checkpoint 键名和支持的 GPT 架构；参数不超过 20M。
- baseline：6 层、6 头、宽度 384、context 256、vocab 8192、batch 16、梯度累积 1、2500 updates。
- 三次完整训练：baseline、模型侧实验、优化侧实验；各自随机初始化，各处理 10,240,000 target tokens。
- 两个实验分别与原始 baseline 对比，每次仅改变一个主要因素；固定数据、词表、context、seed 和 token budget。
- 实验不必优于 baseline；需要解释结果、生成质量和局限，避免单个 seed 的统计显著性结论。
- 最终 checkpoint 仅依据公开验证集选择；只提交一个 checkpoint，但保留三次训练的证据。
- 提交学号命名 ZIP，压缩后不超过 100 MB；Moodle 只有一次最终提交机会。

## 进度

- [x] 获取官方 starter 和数据，验证 SHA-256
- [x] 建立源码、配置、文档、报告与本地输出目录
- [ ] 实现四个 TODO，通过公开测试和 smoke run
- [ ] 完成原始 baseline
- [ ] 完成模型侧实验
- [ ] 完成优化侧实验
- [ ] 三个 run 均生成曲线、样本与完整验证集评估
- [ ] 写完 4–5 页报告，并更新实际复现命令
- [ ] 选择 checkpoint、打包、检查大小及格式
- [ ] 在 Moodle 正式提交并确认状态

源码、baseline、公开测试与原始脚本在初始化时保持官方内容。归档校验和见 `UPSTREAM_SHA256SUMS.txt`。
