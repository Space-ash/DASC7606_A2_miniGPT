# MiniGPT Notebook 工作台

主文件：`Assignment2_Workbench.ipynb`。它调用官方训练、采样、评估和打包脚本，并读取真实运行结果绘图；不包含另一份模型实现或预先生成的实验结果。

## 本地使用

在项目根目录激活 Python 3.12/3.13 环境，安装可选依赖：

```sh
source .venv/bin/activate
python -m pip install -r requirements-notebook.txt
```

用 VS Code 打开 Notebook，在右上角选择此环境作为内核，按顺序运行。独立使用 JupyterLab 时可自行安装 `jupyterlab` 后从项目根目录启动 `jupyter lab`。

本地 `.venv` 可以复用已有 Python 环境（本地符号链接不加入 Git）。移动项目后应确认环境链接仍有效；换机器后请重新建立自己的环境。

## GPU / Colab

1. 在 GPU Notebook 环境打开此文件，选择支持 BF16 的 CUDA GPU。
2. 没有项目时，在第 1 节设置 `CLONE_IF_MISSING = True`，或填写已有项目的 `PROJECT_DIR`。
3. 按 GPU 硬件安装适配的 PyTorch 2.10.0；按需打开依赖安装、固定数据下载和 setup 检查。
4. 先在 `src/model.py` 完成四个 TODO，再运行测试和 smoke run。
5. 填写实验假设和单一改动，准备实验 YAML；分别开启三次训练开关。
6. 完成后选择 `POSTPROCESS_RUNS`，生成曲线、统一提示词续写和完整 FP32 验证评估。
7. 运行展示单元格，查看学习曲线、学习率、记录步的吞吐、评估对比及故事续写；它们只读取输出文件。

所有有成本或会写入结果的动作默认关闭。没有训练文件时显示提示，不使用假数据。`SAVE_FIGURES` 可把报告图片保存到 `outputs/notebook_figures/`。

临时 GPU 运行时结束前备份 `outputs/` 和实验配置。恢复中断训练使用同一 run 的 `last.pt`；新实验不能从别的 run 权重开始。

## 提交与版本管理

报告仍为 **4–5 页 PDF**；Notebook 是辅助工具，官方打包脚本不自动包含 `.ipynb`。可复现模型代码应保存在 `src/`，实际命令写进根目录 README。

公开 Notebook 前清空输出和真实学号。Notebook 当前所有代码单元的 `execution_count` 为 `null`，`outputs` 为空；未执行测试、采样或训练。
