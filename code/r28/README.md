# R28冻结实现、三项已完成权重及五任务结果

R28 = R9固定共同区域编码器 + 官方SPoTr局部通道注意力。在原局部均值/方差之后加入Utonia RMS0.1上下文和局部注意力校正，再进行骨架—表面共同区域融合。阅读[实际架构与逐模块论文来源](../../docs/R28_ARCHITECTURE_ZH.md)、[五任务及baseline结果](../../docs/R28_RESULTS_ZH.md)。

## 目录与版本

- `reference/`：原`CPSF_RAW_POINT_ATTENTION_20261019_R28_A1`的逐字节源文件、协议、验收、四项H01结果。保留主配置`r9_lpa_attention`两项H01的final.pt；uniform只保留结果与证据。
- `extension/`：原`CPSF_R28_TASK_EXPANSION_20261020_A2`的三任务适配、协议、验收和带时间运行快照。适配配置名为`r28_lpa_attention`；它没有引入新主干。已完成TLP的final.pt随包发布。
- `RESULTS.json`：五任务状态及值，未完成任务值严格为null。
- `BASELINE_COMPARISON.json` / `.csv`：本轮27项外部baseline记录、R28与单列的历史R9成绩。
- `server_export.json`：原服务器来源、逐文件SHA、extension与reference的共享字节解析及每项明确省略的非运行文件。
- `live_export.json`：本轮现场源/权重、H01预测和五任务manifest完全等同R9的审计。
- `evidence/`：独立末轮指标、基线权重、官方来源与发布模型加载检查。

R28没有启用历史源中的GeoTransformer、Polynormer、PointMamba或Exphormer。它们不应被根据目录名加入架构图。未包含R29–R33改动。

## 精确模型入口

在完整H01冻结运行根中：

```python
from tools.cpsf_raw_point_attention.model import Model
model = Model('h01_local', 'r9_lpa_attention')
# 'h01_global' uses the same backbone with a regression head.
```

扩展根先覆盖同字节reference依赖，再用：

```python
from tools.cpsf_r28_expansion.model import Model
model = Model('flywire_coarse')  # or 'ted', 'tlp'
```

`extension/`只保存新增或不同的文件，同字节共享源通过`server_export.json/shared_source_resolution`映射到`reference/`。不能直接把两个protocol.json互换。原源文件用自身位置推导运行根；在临时目录将reference和extension的tools/vendor覆盖合并后，五个模型都已在原服务器环境完成构造，三个发布权重strict=True加载通过；详见`evidence/publication_model_smoke.json`。这项检查未训练、未分配GPU，也不等同新机器端到端复现。

## 核验

从仓库根执行，仅需Python标准库：

```bash
python scripts/verify_r28.py
python scripts/verify_r9.py
python scripts/verify_export.py
```

实际训练环境沿用R9：Linux/RTX4090D、PyTorch2.4.0+cu121、spconv-cu120 2.3.6、cumm-cu120 0.4.11、torch_scatter2.1.2+pt24cu121、FlashAttention2.7.4.post1、numpy1.26.4、scipy1.13.1、scikit-learn1.5.2、einops/addict/IPython。官方环境文件在vendor/Utonia。导入原服务器的runtime依赖路径仍可能需要环境配置；代码没有被改写为新机器的一键包。

保留的服务器训练入口为：

```bash
python -m tools.cpsf_raw_point_attention.train --task h01_local --kind r9_lpa_attention
python -m tools.cpsf_raw_point_attention.train --task h01_global --kind r9_lpa_attention
python -m tools.cpsf_r28_expansion.train --task flywire_coarse
python -m tools.cpsf_r28_expansion.train --task ted
python -m tools.cpsf_r28_expansion.train --task tlp
```

上述是历史/现有运行入口说明，**不要据此重复启动仍在运行的实验**。原始数据、几何缓存、样本manifest及全部父运行目录未分发；固定协议中的绝对路径和验收门原样保留。全量训练需要合法数据与原支持文件，不要删除verify或伪造contracts来绕过依赖。

## 来源、许可与结果边界

直接组件：Utonia（ICML2026，commit `da776a0bd3a48c6df83ac2ae0e27b26141cc7e31`，Apache-2.0），SPoTr（CVPR2023，commit `69ea023d3781372df80e23326ef56b58a9b60cd6`，MIT）。第三方原LICENSE/NOTICE保留。RoFormer为项目自写区域3D RoPE的技术参照；其他思想参照与非直接依赖在架构页逐项注明。本发布不对整个研究包追加统一开源许可。

发布的三个final.pt均为本项目从零训练的末轮权重，不是外部预训练。没有发布运行中latest.pt/优化器；FlyWire/TED无最终成绩。全表为seed42固定末轮开发集，开发集已用于历史设计迭代，不能称盲测试或五任务SOTA。旧版论文和旧代码/权重不被覆盖。
