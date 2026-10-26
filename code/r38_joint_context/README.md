# R38 JointContext：统一形态载体与区域耦合

以细 patch 为统一形态载体，通过层级与邻域两种内部计算视图提取上下文，再回投到同一形态载体进行联合融合。RCM 随后以观测区域为共同支撑，组织骨架与表面的区域背景、条件差分消息和联合区域表示。

本目录是原 R38 密集修复版的等价模块与接口重构。保留原点集、权重、分组规则、Utonia/SPoTr、条件差分、融合算术和任务头，不新增可训练参数。原完整实验包仍见 [R38](../r38/README.md)。

## 代码入口

- [support.py](overlay/tools/cpsf_r38_joint_context/support.py)：`SurfaceSupport`、Hierarchy／Neighborhood 视图及原细 patch 加权回投。
- [data.py](overlay/tools/cpsf_r38_joint_context/data.py)：从一份细 patch 点级观测构造原 R38 的两种计算视图，可按样本缓存。
- [unit.py](overlay/tools/cpsf_r38_joint_context/unit.py)：形态基底 H、联合上下文及成对交互。
- [regional.py](overlay/tools/cpsf_r38_joint_context/regional.py)：`RegionalSupport`、独立几何编码及 `RegionalCoupling`；返回区域特征、质量和有效掩码。
- [model.py](overlay/tools/cpsf_r38_joint_context/model.py)：显式 CMM → RCM → 原区域处理与任务读出，以及原模型模块迁移。
- [compatibility.py](overlay/tools/cpsf_r38_joint_context/compatibility.py)：检查点命名迁移与优化器参数顺序兼容。

`SurfaceSupport` 保留 patch 内的点、权重及逐点编码。两个计算视图借用原张量，原编码器保留组→点→细 patch 的计算顺序。`RegionalSupport` 借用原关联、质量、几何和掩码，不重新建立骨架—patch 对应。

```text
cmm
├── morphology                 H
└── context
    ├── hierarchy              Utonia
    ├── neighborhood           密集 SPoTr
    └── interaction            H-D、H-C、D-C 交互
rcm
├── geometry                   原 path_position / relation / position / scale
├── message                    原 PathMessage，含独立消息几何编码
└── joint                      原联合区域融合
processor                      原区域传播与粗区域上下文
```

RCM 的外层接口是 `regions = rcm(nodes, patches, RegionalSupport(g))`，返回 `features / mass / active`。它内部保留骨架、表面各自的质量归一化；基础表面汇聚包含 relation 编码，消息中心化均值只包含 patch 内容。消息 query 仍读取加入共享位置／尺度编码之前的骨架内容。16 维 signed-tanh 调制、16384 关联分块、checkpoint 和原 matched-only joint 均不变。

两个骨架汇聚调用与 CMM 的两条等值输出仍保留各自的计算图，以保持梯度累加顺序。几何模块统一所有权，但原独立权重不共享。`message_enabled=False` 仍仅关闭条件差分消息，保留基础区域汇聚与原 joint；区域传播和任务读出保持在 RCM 外。

## 使用

先组装已有 R38；目标目录必须不存在：

```bash
python scripts/materialize_r38.py /absolute/path/to/new/r38_joint_context --variant dense
cp -R code/r38_joint_context/overlay/tools/cpsf_r38_joint_context \
  /absolute/path/to/new/r38_joint_context/tools/
```

完成原模型和任务头构造后、创建优化器前调用：

```python
from tools.cpsf_r38_joint_context.model import install

backend.model = install(backend.model)
backend.model.load_state_dict(checkpoint['model'], strict=True)
```

`install` 也支持上一版仅重构 CMM 的 JointContext 模型原地升级，并可重复调用。主模型与任务头调用方式不变，`stems(g)` / `joint_regions(graph, point, g)` 诊断接口保留。

已有 R38 dense 加载器产生的 `g` 可直接使用。如需从统一输入构造视图，在 CPU 数据准备阶段调用：

```python
from tools.cpsf_r38_joint_context import prepare_surface_inputs

g = prepare_surface_inputs(base_geometry, original_patch_pos)
# 按原加载器将 g 中的张量移到设备；模型调用方式不变。
```

`base_geometry` 必须保留点级观测且不包含监督标签；`original_patch_pos` 为原缓存质心。适配器保持原坐标重建、网格原点、边界偏移、排序、统计和浮点转换，仅支持本包对应的 H01/FlyWire 输入路径。原 `self.cmm(px, g)` 调用仍兼容。

## 检查点与参数兼容

`install` 原地迁移已有 `H01Dense` / `FlyDense` 参数对象，不消耗初始化随机数。`state_dict()` 使用当前 CMM／RCM 命名；`legacy_state_dict()` 导出原 R38 命名及顺序。原 R38、上一版 CMM-only JointContext、本版 CMM＋RCM 三种检查点均可严格加载。顶层参数遍历顺序保留，可恢复原优化器状态。

RCM 新增的命名迁移如下；CMM 已有的命名迁移保持不变：

| 原名称 | 当前名称 |
|---|---|
| `path_position` | `rcm.geometry.path_position` |
| `relation` | `rcm.geometry.relation` |
| `position` | `rcm.geometry.position` |
| `scale` | `rcm.geometry.scale` |
| `region_message` | `rcm.message` |
| `joint` | `rcm.joint` |

直接访问这些旧顶层属性的外部代码需改用新路径。保存与加载均使用 state dictionary；本包不承诺整模型 pickle 的跨版本兼容。

## 本版验证范围

- H01 两项：初始／原最终权重前向、完整有效 batch 的两步更新、全部梯度、裁剪、权重回载、优化器恢复及 A-B-A 输入隔离均逐位一致。
- H01 完整开发集：214 个局部与 24 个整体样本的预测与原保存结果逐位一致，最大输出绝对差为 0；局部 Macro-F1 `0.6544237405106971`、整体 MAE `0.050025708973407745` 保持不变。
- 区域接口：真实模型的区域两路输出、条件消息、joint 和进入 processor 的张量一致；关闭消息后的结果也一致。
- 迁移与边界：上一版模型原地升级、三种检查点严格加载、参数对象／顺序／随机状态／eval 模式及旧 state metadata 验证；覆盖双模态、仅骨架、孤立表面 patch、单 patch 区域、无效区域，以及 16383／16384／16385 关联数的分块边界，前向和梯度均逐位一致。
- FlyWire：12 个形状极值、中位规模及六类别代表样本的输入与保存预测一致。MICrONS：相同确定性环境下一个验证细胞的 encoder 接入前后输出一致。两者不构成本版完整队列重新评测。

CMM 输入适配器本版未修改；上一版已验证 H01 全部 930 个局部与 104 个整体输入字段与原加载器逐位一致，本版真实模型验收亦逐样本重建并核对所用输入。

验收更新全部丢弃，不产生新训练成绩。MICrONS 外层检查点和优化器适配仍不在本包范围内；不要直接套用顶层模型的迁移结论。

本次发布仅包含七个模型／输入适配 Python 文件及本说明，不含数据、新权重或服务器运行脚本。现存冻结训练目录未被替换。原依赖和官方组件来源、许可要求保持不变。
