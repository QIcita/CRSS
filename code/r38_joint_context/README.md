# R38 JointContext：统一细 patch 形态载体

以细 patch 为统一形态载体，通过层级与邻域两种内部计算视图提取上下文，再回投到同一形态载体进行联合融合。

本目录是原 R38 密集修复版的等价接口重构。保留原点集、权重、分组规则、Utonia/SPoTr、融合算术、RCM 和任务头，不新增可训练参数。原完整实验包仍见 [R38](../r38/README.md)。

## 代码入口

- [support.py](overlay/tools/cpsf_r38_joint_context/support.py)：`SurfaceSupport`、Hierarchy／Neighborhood 视图及原细 patch 加权回投。
- [data.py](overlay/tools/cpsf_r38_joint_context/data.py)：从一份细 patch 点级观测构造原 R38 的两种计算视图，可按样本缓存。
- [unit.py](overlay/tools/cpsf_r38_joint_context/unit.py)：形态基底 H、联合上下文及成对交互。
- [model.py](overlay/tools/cpsf_r38_joint_context/model.py)：CMM → 区域聚合 → RCM，以及原模型模块迁移。
- [compatibility.py](overlay/tools/cpsf_r38_joint_context/compatibility.py)：旧／新检查点命名与优化器参数顺序兼容。

`SurfaceSupport` 保留 patch 内的点、权重及逐点编码，不把点级信息压缩成无法恢复的 patch 摘要。主模型调用 `self.cmm(SurfaceSupport(g, px))`；原 `self.cmm(px, g)` 调用仍兼容。两个视图借用原张量，原编码器保留组→点→细 patch 的计算顺序。

注册模块路径与首版 JointContext 相同：

```text
cmm
├── morphology                 H
└── context
    ├── hierarchy              Utonia
    ├── neighborhood           密集 SPoTr
    └── interaction            H-D、H-C、D-C 交互
```

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
backend.model.load_state_dict(checkpoint['model'])
```

已有 R38 dense 加载器产生的 `g` 可直接使用。如需从统一输入构造视图，在 CPU 数据准备阶段调用：

```python
from tools.cpsf_r38_joint_context import prepare_surface_inputs

g = prepare_surface_inputs(base_geometry, original_patch_pos)
# 按原加载器将 g 中的张量移到设备；模型调用方式不变。
```

`base_geometry` 必须保留点级观测且不包含监督标签；`original_patch_pos` 为原缓存质心。该适配器保持原坐标重建、网格原点、边界偏移、排序、统计和浮点转换，仅支持本包对应的 H01/FlyWire 输入路径。

## 兼容与验证

`install` 原地迁移已有 `H01Dense` / `FlyDense` 参数对象，不消耗初始化随机数。`state_dict()` 的键名与首版 JointContext 一致；`legacy_state_dict()` 导出原 R38 命名及顺序。原版与首版 JointContext 检查点均可严格加载，顶层参数遍历和优化器状态保持兼容。

本次验证包括：

- H01 全部 930 个局部和 104 个整体输入：统一适配器重建的所有字段与原加载器逐位一致。
- H01 两项：初始／原最终权重前向、完整有效 batch 的两步更新、全部梯度、裁剪、权重回载、优化器恢复及 A-B-A 输入隔离均逐位一致。
- H01 完整开发集：214／24 个样本与原保存预测逐位一致，最大输出绝对差为 0；局部 Macro-F1 `0.6544237405106971`、整体 MAE `0.050025708973407745` 保持不变。
- FlyWire：形状极值、中位规模及各类别代表样本的输入与保存预测核对。MICrONS：固定确定性设置下的单细胞 encoder 接入检查。两者不构成本次完整队列重新评测。

验收更新全部丢弃，不产生新训练成绩。MICrONS 外层检查点和优化器适配仍不在本包范围内；不要直接套用顶层模型的迁移结论。

本次发布仅包含六个模型／输入适配 Python 文件及本说明，不含数据、新权重或服务器运行脚本。现存冻结训练目录未被替换。原依赖和官方组件来源、许可要求保持不变。
