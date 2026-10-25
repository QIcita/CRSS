# R38 发布包（2026-10-25）

R38 在 R34 的 CMM 三路组合处加入有界低秩两两交互，保留 Utonia、SPoTr 与 R34 路径条件消息。密集邻域版扩大 SPoTr 的计算邻域，再将增量回投至原细支撑；最新 DRW 试验只改训练损失的类别权重日程。R28 仍为总体研究主参照，本次发布不代表 R38 已在全部任务胜出。

[结构与公式](../../docs/R38_ARCHITECTURE_ZH.md) · [结果与状态](../../docs/R38_RESULTS_ZH.md) · [机器可读结果](RESULTS.json)

| 实验目录 | 内容 | 发布时状态 |
|---|---|---|
| `experiments/original` | R38 pair fusion，H01 局部/整体 | 两项完成，保留两份 final.pt |
| `experiments/dense` | 密集邻域修复，H01 整体/FlyWire 粗分类 | 两项完成，保留两份 final.pt |
| `experiments/microns` | 完整观测神经突树的区域输入突触计数 | 完成，保留验证集选择的 best.pt 和 val/test 预测 |
| `experiments/fine` | FlyWire 148 类细分类；H01 局部等价性审计 | 细分类已暂停，无最终成绩；H01 局部复用原 R38 |
| `experiments/drw` | FlyWire 粗分类延迟温和重加权 | 训练中快照，无最终成绩或运行中权重 |
| `experiments/parents` | 父版本原字节协议 | 用于追溯输入与继承关系 |

`source_map.json` 映射 560 份运行源码至本仓库历史文件或本版 `overlay/`，避免改写已发布版本。`server_export.json` 记录本次服务器来源、SHA-256、时间和排除项。`release_manifest.json` 覆盖本版代码、证据、说明及脚本。

## 组装与检查

在仓库根目录运行（哈希核验只需 Python 标准库）：

```bash
python scripts/verify_r38.py
python scripts/materialize_r38.py /absolute/path/to/new/r38 --variant drw
```

目标目录必须不存在；可选 `original`、`dense`、`fine`、`drw`、`microns`。所有变体始终保留于组装目录的 `experiments/`；选中变体的协议与记录另外复制到根目录。源码映射共用同一冻结依赖集合，任务身份以各自协议和权重为准。

在兼容的 PyTorch 及 Utonia/SPoTr 依赖环境中可复核发布证据：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/validate_r38_runtime.py \
  /absolute/path/to/new/r38 --output /absolute/path/to/validation.json
```

已在 PyTorch 2.4.0+cu121 环境中通过 CPU 模型构造、五份权重 strict load、保存预测的指标复算；导入的项目和上游模型源码均来自组装目录。见 [验证记录](evidence/publication_validation.json)。本次没有重新前向推理、重新训练或重新选择测试结果；验证脚本需兼容运行依赖，标准库检查不替代它。

完整训练仍需原数据、缓存、兼容环境及父版本的绝对路径和验收文件。冻结协议中的路径与哈希保留原貌，组装脚本不是免配置训练安装器，不绕过验收门槛。未分发原始几何、数据缓存、外部预训练、运行中的优化器 checkpoint 或凭据。训练入口分别见 `tools/cpsf_cmm_pair_fusion/worker.py`、`tools/cpsf_r38_dense/worker.py`、`tools/cpsf_r38_dense_extension/worker.py`、`tools/cpsf_r38_delayed_reweight/worker.py` 和 `tools/cpsf_microns/train.py`（组装后路径）。

官方 Utonia/SPoTr 源码及许可证按历史映射保留；本次没有授予新的统一开源许可。见 [来源与许可](../../RIGHTS_AND_PROVENANCE.md)。
