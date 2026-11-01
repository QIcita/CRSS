# C版PI公共密度／加权残差候选

2026-11-01。最新实现的独立候选 `pi_basis_residual`，接入已发布C版SMM的PI编码器。**尚未正式训练、没有预测性能结果；不是当前三任务成绩所对应的默认C版。** 原C及其入口保留在[父发布包](../c_smm_cmm_rcm/README.md)。

## 计算与接口

原三通道为无权重密度R、持久性加权图G、半径加权图B。对W∈{G,B}：

`a_W = <W,R>/<R,R>`，`D_W = W - a_W R`。

R全零时系数为0。编码前保留`(R,D_G,D_B)`和`(a_G,a_B)`，因此可重建`W=D_W+a_W R`。残差有正负，不截断或重新min–max缩放。MICrONS的B=R对应a_B=1、D_B=0；其他数据集的有效加权差异保留。

分解图经过原8×16池化及原卷积编码器。两个系数通过零初始化、无bias的2→384投影加入统计基底，再LayerNorm；保留原0.1有界卷积残差。PI输出仍384维，新增768参数。其余SMM、CMM、RCM和任务头的原始参数保持一致。

可逆性仅针对输入分解，不能推广到池化和神经网络压缩；D_G与D_B仍可能相关。该候选重组公共信息，不宣称消除了所有冗余或已提升准确率。它不改PI条带、根条带、栅格、核宽或输入缓存，不读取突触标签构造输入。

| 任务 | 参数量 | 输出 |
|---|---:|---|
| H01整体 | 5,263,479 | sigmoid标量 |
| FlyWire粗分类 | 5,264,124 | 六类logits |
| MICrONS | 5,263,479 | 各查询区域期望计数 |

代码：[model.py](sources/tools/cpsf_pi_basis_residual/model.py)。来源与父包哈希：[source_map.json](source_map.json)。

## 组装与使用

在仓库根目录执行：

```bash
python code/c_pi_basis_residual/materialize.py
python code/c_pi_basis_residual/materialize.py /absolute/path/to/new/c_candidate
```

第一条检查候选及父C发布包的源码哈希与Python语法；第二条拒绝覆盖已有目录，组装父C依赖后增加候选。组装目录中的`model.py`仍构造原C，`candidate.py`明确构造新候选，两种入口不互相替换。

```python
from candidate import build_model

model = build_model('flywire_coarse')
# logits = model(prepared_cell)
count_model = build_model('microns', initial=training_synapses_per_um)
```

H01整体的initial应为训练集目标均值，MICrONS应为训练集总突触数／总长度；默认值仅供构造。完整CUDA依赖、输入字段和任务读出沿用[父C包](../c_smm_cmm_rcm/README.md)，不是原始XYZ直接输入。候选需要自己训练的权重；不能严格加载原C权重后就声称完成新模型评估。

## 验证与发布范围

原工程验收覆盖原C及根条带两套共17,206份PI，分解重建最大绝对误差5.96×10⁻⁸；三个任务真实输入的前向／反向与原参数保留检查通过。验收借用原C已训练权重验证接口，新投影保持零初始化，没有optimizer或参数更新，不是候选训练成绩，也不保证从头初始化首步全部PI参数都有任务梯度。

发布包额外验证组装目录中的三个任务构造、初始化身份和PI算子，检查不会回退导入原工程源码；详见[validation.json](validation.json)。本次没有全量重训或新增开发／测试集性能评价。

仅发布模型源码、组装器和说明；不包含数据、缓存、权重、优化器或服务器调度器。来源与许可沿用[父包说明](../c_smm_cmm_rcm/README.md#第三方组件)和[仓库权利说明](../../RIGHTS_AND_PROVENANCE.md)。
