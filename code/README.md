# 当前完整模型代码包

版本：2026-10-11。保留固定残差clean core；TED使用lr_half，TLP使用directional_xyz。此包不是新模型版本，不改变正在训练的外部基线。

## 从哪里开始

- `model_full.py`：可直接导入的完整网络，661行。将11个历史模块的依赖合并到一个文件，保留数值表达式、继承、参数名及初始化顺序。只依赖PyTorch，不需要历史项目路径。
- `example.py`：五个任务的人工输入、权重加载和CPU前向示例。人工输出不代表效果。
- `FIXED_RESIDUAL_ZH.md`：固定残差的实际公式、作用、工程属性及证据。
- `checkpoints/<task>/final.pt`：五个当前默认完整模型的原始权重，以及各自identity、metrics、history。
- `snapshots/`：从服务器冻结协议导出的完整来源文件，包含数据构造、训练、评测、损失与历史依赖。三套快照分开存放，禁止用混合目录替代原版本。
- `validation/`：五任务的初始化、检查点、前向和反向一致性检查结果。
- `server_export.json`、`consolidation_sources.json`、`MANIFEST.sha256.json`：来源及文件哈希。

主模型的单文件没有用自写小网络替代原模型，而是通过AST合并原定义、消除跨目录导入；合并脚本是`build_standalone.py`。为了保留构造顺序，内部仍保留部分历史继承和消融分支；对外`build_model`只提供经过验证的full版本，不承诺内部任意历史分支组合有效。

## 快速运行

Python 3.10+；本次核验环境为PyTorch 2.4.0+cu121，但核验与示例均在CPU上执行。单文件网络没有自定义CUDA编译依赖。

```bash
python -m pip install -r requirements-model.txt
python verify_package.py
python example.py --task h01_local
python example.py --task h01_global
python example.py --task flywire_coarse
python example.py --task ted
python example.py --task tlp
```

在自己的代码中：

```python
import torch
from model_full import build_model, load_model

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)  # TLP保留确定性关联聚合的约束

model = load_model("h01_local", "checkpoints/h01_local/final.pt")
# g是与对应任务一致的区域输入字典，不能直接把原始XYZ张量传入。
# model.eval()
# with torch.no_grad():
#     logits = model(g)

# 从头训练时：
# model = build_model("h01_local")
```

CUDA运行沿用原实验的确定性设置（进程启动前`CUBLAS_WORKSPACE_CONFIG=:4096:8`，关闭TF32等）；本次代码整理没有重新跑全量训练或GPU指标评测。

## 五任务的准确身份

| task | 输出 | 当前适配 | 有效参数量 |
|---|---|---|---:|
| h01_local | 3个logits：AIS／axon／dendrite的实际索引以原协议为准 | 查询邻域读出，minimum-four层级 | 255875 |
| h01_global | 标量，sigmoid后的兴奋性输入比例 | 整细胞质量加权读出，minimum-four层级 | 255617 |
| flywire_coarse | 6个logits | 冻结V23层级、表面关联分块及激活重计算 | 256262 |
| ted | 2个反对称logits `[-s,s]` | 观测骨架查询读出；当前训练lr0.0005 | 255617 |
| tlp | N×N对称节点对logits | 真实邻域XYZ＋原第四维；端点及走廊区域读出 | 304897 |

FlyWire类序为ascending、central、optic、sensory、visual_centrifugal、visual_projection。TLP评测在上三角未观测边集合计算AP，不把对角线或已观测边当作待预测负例。

这些是同一主干设计的分任务训练模型，不是一份共享训练后权重。H01/FlyWire物理单位约定与NREDD原生坐标限制、两套层级构造、TLP独有描述均保留。

## 模型输入字段

本表描述full前向所需的区域缓存。每个样本独立处理，不在模型内部混合不同细胞。R为共同区域数，S为骨架节点数，P为表面点数，J为表面patch数。

| 字段 | 含义／形状 |
|---|---|
| `sf`, `pf` | S×4骨架描述、P×4表面描述；含义由对应冻结构造代码定义 |
| `point_owner`, `point_weight`, `patch_mass`, `patches` | 点所属patch、采样质量、patch总质量与J |
| `sr`, `sn`, `sw` | 骨架节点向区域聚合的关联行、列、弧长权重 |
| `pr`, `pj`, `pw`, `phi` | 表面patch向区域的软关联、权重及每关联5维几何关系 |
| `graph_mass`, `surface_mass` | 区域内两模态各自的质量，长度R |
| `has_g`, `has_p` | 区域模态存在掩码，bool，长度R |
| `s`, `pos`, `length`, `radius`, `nregions` | 骨架坐标、区域坐标／长度、支持半径及R |
| `adjacency` | 细区域邻接边，E×2整数索引 |
| `coarse_owner`, `ncoarse` | 细区域所属粗组与粗组数 |
| `query` | H01局部查询坐标，3维 |
| `query_xyz` | TED单点查询或TLP全部端点查询，N×3 |
| `pf_directional` | TLP真实邻域描述；当前full只取前三维，第四维来自原pf |
| `bridge_row`, `bridge_col`, `bridge_weight`, `pair_count`, `pair_lookup` | TLP节点对走廊读出与N×N节点对索引 |

动态分块、骨架弧长区域、软关联和层级构造发生在输入准备侧，不能因单文件forward接收g就省略它们。原始构造代码均保留在snapshots中。标签仅进入损失／评测，不进入模型输入。

## 原始实验脚本与重训边界

| 快照 | 保留来源 | 训练入口 |
|---|---|---|
| snapshots/h01_flywire | CPSF_FIXED_CORE_ABLATION_20261005 | `tools/cpsf_fixed_core/train.py` |
| snapshots/ted | CPSF_NREDD_CONFIG_ONLY_20261009 | `tools/cpsf_config_only/train.py`，TED `lr_half` |
| snapshots/tlp | CPSF_TLP_DIRECTIONAL_STAGE6_20261009 | `tools/cpsf_directional/train.py`，TLP `directional_xyz` |

输入构造重点：`cpsf_h01_topology_repair/adapter.py`、`cpsf_v22_shared_region/{data,support}.py`、`cpsf_v24_physical_conditioning/data.py`、`cpsf_flywire_v23/data.py`、`cpsf_fixed_nredd/data.py`、`cpsf_pair_region/data.py`、`cpsf_directional/{data,descriptor}.py`。

原训练脚本保留了服务器绝对路径、父协议校验、原始数据和NREDD上游工程依赖。这些历史入口不是迁到任意电脑后即可一键全量重训的脚本。代码包提供完整网络、对应源代码和配置，但不包含原始数据集及所有几何缓存；迁移训练数据／外部上游项目需另行配置。不要关闭哈希校验或随意合并三套快照来绕过依赖。

单文件前向可独立运行；复现原实验使用原服务器数据与冻结入口。没有将“接口样例成功”称为“新环境全量复现成功”。

## 检查点中的历史门控张量

H01的两个full结果原来保存在V26的`fixed_residual`对照目录。其checkpoint包含三个**未被该计算分支使用**的零张量：`gate_alpha`、`region_gate.weight`、`region_gate.bias`。

`load_model`仅在H01、身份明确为fixed_residual、三项完整存在且均为零时删除这些非活动项，再对完整有效参数执行`strict=True`加载。没有丢弃其他参数，没有把门控full当作固定残差结果。原始checkpoint字节保持不变。已另外使用历史V26的fixed_residual类加载未清理权重，确认真实训练样本输出完全一致。

## 已完成的核验

服务器冻结源文件538条记录逐项哈希核验，五任务checkpoint和结果身份分别保留。单文件与冻结源码比较包括：

1. seed42初始化参数与键名逐项完全一致。
2. 五任务已训练权重严格加载。
3. 五任务人工输入及每任务一个真实训练样本，eval／train模式前向、反向梯度逐项一致，最大差均为0。
4. FlyWire额外覆盖超过32768条表面关联的分块／重计算路径。
5. H01原历史fixed_residual模型与单文件在所核验真实样本上的输出一致。

核验只用CPU、人工输入和训练侧缓存，没有读取测试集，也没有重新计算全验证集成绩。`validation/*.json`记录样本来源、软件版本、模型SHA256和具体比较结果。

可自行运行`python validate_export.py --task h01_local`检查人工样例；加`--real`需要原服务器数据路径和numpy/scipy。每任务单独进程运行，以隔离历史Python模块命名空间。
