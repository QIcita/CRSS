# C：渐进式 SMM–CMM–RCM

模型标识：`regional_scaled_pi`，2026-10-31。发布实际用于 H01、FlyWire、MICrONS 的 C 版模型源码与构造依赖。共同主干结构与初始化保持相同，各任务从头训练，只有任务读出范围与输出头不同。

这是模型代码发布，包含来源映射、原许可说明和源码组装工具。数据、几何缓存、权重、优化器状态、训练／下载／调度入口不在本次提交中。FlyWire 在本次发布准备时仍在训练，不能把源码验收作为最终效果。

## 模型入口

| 部分 | 实际源码 |
|---|---|
| C 的区域处理、交互幅度与轻量 PI | [model.py](sources/CPSF_H01_PROGRESSIVE_REPAIR_20261030_A1/tools/cpsf_h01_progressive_repair/model.py) |
| RCM 区域信息保留、区域传播和读出 | [regional.py](sources/CPSF_H01_PROGRESSIVE_REPAIR_20261030_A1/tools/cpsf_h01_progressive_repair/regional.py) |
| FlyWire／MICrONS 任务适配 | [model.py](sources/CPSF_C_TRANSFER_20261031_A2/tools/cpsf_c_transfer/model.py) |
| SMM 内部交互与同／跨尺度交互定义 | [渐进式模型](sources/CPSF_PROGRESSIVE_SMM_20261029_A1/tools/cpsf_progressive_smm/model.py) |
| TreeLSTM 分支状态与 SMM/CMM 模块组织 | [模态模块](sources/CPSF_SMM_JOINT_20261028_A1/tools/cpsf_smm_joint/models.py) |
| 不相连树的批量计算 | [runtime model](sources/CPSF_SMM_RUNTIME_20261029_A1/tools/cpsf_smm_runtime/model.py) |
| CMM 的细 patch、层级／邻域视图 | [既有 JointContext](../r38_joint_context/README.md) |

```text
观测骨架 ─ TreeLSTM 分支状态 ─┐
持久性图 ─ 轻量 PI 编码 ─────┴─ SMM：模态内局部／全局交互

真实表面 ─ 细 patch 基底 ─────┐
         └ 层级／邻域计算视图 ┴─ CMM：同一细 patch 上联合上下文

                 SMM + CMM
                     ↓
RCM：支撑关联注意力 → 内容门控 → 同尺度／跨尺度交互
                     ↓
         模态直接基底 + 交互增量 + 区域传播
                     ↓
          区域／整体读出 → 任务头
```

PI 使用同一份三通道112×112输入：固定8×16池化及LayerNorm形成384维基底，三层卷积形成有界残差。C不使用DINOv2作为最终PI编码器。冻结构造过程仍先实例化原GraPHFormer/DINOv2模块，再替换为轻量PI；保留这段构造依赖以保持随机数消耗、初始化和参数身份。

交互残差补偿特征维度导致的幅度衰减；RCM保留原始模态区域信息、内容门控及同尺度／跨尺度交互，并恢复区域处理器和96→128非线性读出。MICrONS从真正的区域状态读出，不把整体预测复制给所有区域。

| task | 输出 | 参数量 |
|---|---|---:|
| `h01_local` | 查询邻域三分类logits | 5,262,969 |
| `h01_global` | sigmoid标量 | 5,262,711 |
| `flywire_coarse` | 六分类logits | 5,263,356 |
| `microns` | 各查询区域的期望输入突触数 | 5,262,711 |

MICrONS保留原`length * exp(clamp(log_rate, -12, 8))`，`length`为区域长度。FlyWire训练调用`forward_group(items)`，保持各细胞的树彼此不连通；单样本推理调用`model(x)`。

## 源码组装

在仓库根运行，只需Python标准库：

```bash
python code/c_smm_cmm_rcm/materialize.py
python code/c_smm_cmm_rcm/materialize.py /absolute/path/to/new/c_model
```

第一条核对全部来源哈希和Python语法；第二条拒绝覆盖已有目录，将已有发布源码与本次新增源码组装到一个目录。仓库中的冻结文件逐字节保留，组装时仅将Python字符串内的原服务器源码根路径换成目标目录，不改计算表达式。

在兼容CUDA环境及组装目录中：

```python
from model import build_model

model = build_model('flywire_coarse')
# model.load_state_dict(checkpoint['model'], strict=True)
# logits = model(prepared_cell)

count_model = build_model('microns', initial=training_synapses_per_um)
# expected_counts = count_model(prepared_cell)
```

从头训练时，H01整体`initial`使用训练集目标均值，MICrONS使用训练集总计数／总区域长度；示例默认值仅供构造。训练数据统计不由发布包拟合，不读取开发／测试集。保存权重用`state_dict`，不承诺历史完整模型pickle可跨环境加载。

运行依赖包括兼容的PyTorch/CUDA、DGL、NumPy、SciPy、einops、timm、torch-scatter、spconv和flash-attn等；原实验为Python3.10、PyTorch2.4.0+cu121。原算子与官方组件保留，不能用CPU示例替代完整CUDA运行环境。`source_map.json`覆盖项目／官方模型源码，不分发Python环境或编译扩展。

输入不是原始XYZ。每个`x`包含`batch`中的DGL树、五维标准化骨架特征与根索引，分量质量`mass`、持久性图`image`、区域—分支映射`jr/jb/jw`，以及R38密集修复输入字典`g`。MICrONS另需`query_index`和`length`。`g`保留原细patch、质量、层级／邻域视图、骨架关联和区域几何。突触标签、类型标签不传入模型。准备输入和复现完整训练仍需要冻结数据流水线及对应协议。

## 来源与验证

`source_map.json`登记126份实际构造依赖、动态加载源码及上游说明。75份直接引用仓库既有相同字节文件，51份作为新增冻结源码保存；没有覆盖R28/R34/R38历史文件。

共同主干的正式初始化摘要为：

`68b8d979971eca3094965c30b9b922d8d1544909c019d9fb0b18660a94eb1bcb`

四个任务均在隔离组装目录构造成功，参数量、共同主干摘要和完整初始状态摘要均与原冻结源码一致；没有回退导入原工程源码。H01局部／整体的最终检查点与MICrONS的best检查点均通过`strict=True`加载，缺失与多余键均为0。122份Python源码的AST对比确认仅替换了源码根路径字符串。具体记录见[validation.json](validation.json)；未重新训练或新增开发／测试集评估。

所有源路径和哈希保留在映射中；入口对应`CPSF_H01_PROGRESSIVE_REPAIR_20261030_A1`与`CPSF_C_TRANSFER_20261031_A2`。原`common.py`中的训练验收和调度辅助函数是冻结导入依赖；本包未携带它们所需的训练运行文件，模型构造入口不调用这些函数。

## 第三方组件

- Utonia和SPoTr沿用既有源码与许可，分别为Apache-2.0和MIT；映射中保留许可证。
- GraPHFormer来源：[Uzshah/GraPHFormer](https://github.com/Uzshah/GraPHFormer)。保留原模型源码、上游README以及实验所用TreeLSTM兼容版本；其骨架编码与C自己的PI编码是不同部分。本次冻结上游根目录未见独立LICENSE，不将其声明为本项目统一授权代码。
- DINOv2保留官方源文件头与Apache-2.0许可证，仅承担冻结构造兼容性，不加载外部预训练权重。

本包继续遵循仓库[来源与权利说明](../../RIGHTS_AND_PROVENANCE.md)，不新增统一开源许可。
