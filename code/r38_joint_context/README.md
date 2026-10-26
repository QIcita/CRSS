# R38 JointContext：联合上下文模块代码（2026-10-26）

本目录将原 R38 密集邻域修复版的 CMM 重组为形态基底和联合上下文单元，保留原 H+D+C 数值计算。它是代码结构重构，不新增模型参数，也不构成新的训练结果。

```text
CMM / ContextualMorphology
├── morphology                     原细 patch 形态基底 H
└── context / JointContext
    ├── hierarchy                  原 Utonia 路径 C
    ├── neighborhood               原密集 SPoTr 路径 D
    └── interaction                原 H-D、H-C、D-C 成对交互
```

主模型通过 `self.cmm(px, g)` 调用上述真实注册子模块，再执行原区域聚合、RCM、区域传播和任务头。外围中间特征捕获的临时 forward hooks 与缓存已移除；官方组件的内部实现与 checkpoint 机制保留。

## 代码入口

- [unit.py](overlay/tools/cpsf_r38_joint_context/unit.py)：形态统计、D/C 联合上下文与原融合算术。
- [model.py](overlay/tools/cpsf_r38_joint_context/model.py)：显式 CMM → 区域聚合 → RCM 接入，以及原模型模块迁移。
- [compatibility.py](overlay/tools/cpsf_r38_joint_context/compatibility.py)：检查点命名双向转换和原优化器参数遍历顺序。

四个 Python 文件与通过真实 H01 验收的工作版本逐字节一致。本次仅发布模型代码及本说明；运行审计脚本依赖原服务器数据和实验后端，未作为可移植运行入口分发。

## 基于已有 R38 发布包使用

从仓库根目录先组装已有 R38，再放入本模块。目标目录必须不存在：

```bash
python scripts/materialize_r38.py /absolute/path/to/new/r38_joint_context --variant dense
cp -R code/r38_joint_context/overlay/tools/cpsf_r38_joint_context \
  /absolute/path/to/new/r38_joint_context/tools/
```

在组装目录的原 R38 环境中，完成原模型和任务头构造后、创建优化器前调用：

```python
from tools.cpsf_r38_joint_context.model import install

backend.model = install(backend.model)
backend.model.load_state_dict(checkpoint['model'])  # 原版或新版 state_dict，严格加载
```

`install` 接受原 `H01Dense` / `FlyDense` 完整模型，原地转移已有参数对象，不重新初始化或消耗随机数。H01 局部可先使用原 `H01Model('h01_local')` 配合 `cpsf_r38_dense.model.install` 构造；其输入不触发密集邻域扩张。

| 原参数路径 | 新参数路径 |
|---|---|
| `patch.*` | `cmm.morphology.*` |
| `native_point.*` | `cmm.context.hierarchy.*` |
| `raw_refine.*` | `cmm.context.neighborhood.*` |
| `cmm_pair_fusion.*` | `cmm.context.interaction.*` |

`state_dict()` 保存新结构，`legacy_state_dict()` 导出原命名及顺序。顶层 `parameters()` / `named_parameters()` 保留原参数遍历顺序，以兼容优化器状态与梯度裁剪。原顶层模块属性已迁移，诊断工具应改用上表的新路径。

## 已验证的等价性及范围

H01 局部与整体分别完成初始/原最终权重前向、非零交互权重下全部梯度、两步 AdamW 更新、梯度裁剪、检查点迁移、优化器恢复及 A-B-A 输入隔离检查，比较均逐位一致。整体验收实际触发密集邻域路径。验收优化更新已丢弃。

使用原最终权重核对完整开发集的已保存预测，局部 214 个样本、整体 24 个样本的最大输出绝对差均为 0。原局部 Macro-F1 `0.6544237405106971`、整体 MAE `0.050025708973407745` 保持一致；这不是新的训练成绩。原权重、结果及上游组件来源见 [R38 发布包](../r38/README.md)。

真实数据验收仅覆盖 H01 两任务。FlyWire 安装接口已提供，尚未完成同等验收。MICrONS 等外层包装模型的 encoder 前缀及外层参数遍历尚需单独适配，不能直接套用顶层兼容结论。

本包继承原 R38 的依赖和输入要求，不含原始数据、缓存、额外权重或独立训练环境。现有冻结训练入口未切换到此代码；新实验应单独记录源码身份，不能覆盖原冻结源文件。已有分支消融结果保持原义，模块重组本身不证明内部路径独立有效。
