# 实现导航与版本关系

完整网络见[model_full.py](../code/model_full.py)。该文件由原始定义合并而来，保留计算、继承和初始化顺序，不是新写的简化网络。

| 计算环节 | 真实符号 | 作用 |
|---|---|---|
| 模型入口 | [`build_model`](../code/model_full.py#L628) | 选择五任务当前完整模型 |
| 严格权重加载 | [`load_model`](../code/model_full.py#L651) | 读取原始检查点；H01非活动门控张量按身份条件处理 |
| 骨架/表面编码 | [`ScaleAwareEncoder`](../code/model_full.py#L250) | stems：patch一二阶矩、分别带权聚合、显式位置与尺度 |
| FlyWire内存路径 | [`FlyWireEncoder`](../code/model_full.py#L296) | 表面关联分块及激活重计算，保持冻结实现 |
| 当前H01/FlyWire融合 | [`FixedCoreMixin`](../code/model_full.py#L394) | embedding：区域均值＋0.1交互残差，再进入processor |
| 共享局部/粗上下文 | [`PositionProcessor`](../code/model_full.py#L207) | 细区域消息传递、粗区域聚合与位置注意力 |
| 上下文消融控制 | [`ContextProcessor`](../code/model_full.py#L373) | 保留fine传播，控制是否启用coarse上下文 |
| 三轴位置编码 | [`rotary`](../code/model_full.py#L22) | 在attention中应用3D rotary坐标 |
| TED适配 | [`NreddModel`](../code/model_full.py#L453) | 局部观测查询读出与两个反对称logits |
| TLP节点对读出 | [`PairRegionModel`](../code/model_full.py#L528) | 端点及走廊区域信息；对称节点对输出 |
| TLP确定性聚合 | [`DeterministicModel`](../code/model_full.py#L594) | 匹配确定性区域累加 |
| TLP方向描述 | [`DirectionalModel`](../code/model_full.py#L611) | directional_xyz：新XYZ＋原pf第四维 |

## 避免误读继承

`RegionEncoder.forward`是历史基础实现；不能读完它就把其中拼接MLP当作当前full的全部融合。请从`build_model`选择实际类，沿其方法解析关系阅读。H01/FlyWire使用`FixedCoreMixin.embedding`；TED/TLP使用各自保留的查询路径。`code/consolidation_sources.json`列出每个定义的原文件与哈希。

## 输入准备也是方法的一部分

`model_full.py`接收已构造好的区域字典`g`。动态采样、折线区域、软关联和层级构造发生在模型前向之前，不能在图中省略。

以下均为`code/snapshots/h01_flywire/tools/`下的冻结来源：

- [cpsf_v22_shared_region/data.py](../code/snapshots/h01_flywire/tools/cpsf_v22_shared_region/data.py)：`whole_geometry`、`local_geometry`和`regions`的组合。
- [cpsf_v22_shared_region/support.py](../code/snapshots/h01_flywire/tools/cpsf_v22_shared_region/support.py)：`chains`、`build_units`构造共同支持。
- [cpsf_h01_topology_repair/adapter.py](../code/snapshots/h01_flywire/tools/cpsf_h01_topology_repair/adapter.py)：保留观测骨架拓扑的输入修复。
- [cpsf_v24_physical_conditioning/data.py](../code/snapshots/h01_flywire/tools/cpsf_v24_physical_conditioning/data.py)：相关层级构造来源，名称不代表全部历史编码变更都启用。
- [cpsf_flywire_v23/data.py](../code/snapshots/h01_flywire/tools/cpsf_flywire_v23/data.py)：FlyWire冻结支持和区域构造。

NREDD必须使用对应任务快照：

- [TED data.py](../code/snapshots/ted/tools/cpsf_fixed_nredd/data.py) 与 [TED train.py](../code/snapshots/ted/tools/cpsf_config_only/train.py)。
- [TLP pair-region data.py](../code/snapshots/tlp/tools/cpsf_pair_region/data.py)、[directional descriptor.py](../code/snapshots/tlp/tools/cpsf_directional/descriptor.py) 与 [TLP train.py](../code/snapshots/tlp/tools/cpsf_directional/train.py)。

H01/FlyWire的保留训练入口为[固定主干train.py](../code/snapshots/h01_flywire/tools/cpsf_fixed_core/train.py)。入口依赖原数据、路径与父协议，不代表可在任意目录直接全量训练。各快照中的历史依赖应隔离使用。

## 输入与标签

完整字段表在[code/README.md](../code/README.md)：`sf/pf`为两模态描述；`sr/sn/sw`为骨架关联；`pr/pj/pw/phi`为表面软关联与几何关系；`pos/length/radius`为区域几何；`adjacency/coarse_owner`为上下文；`query/query_xyz`为任务查询。标签只进入训练损失与评测。

当前消融应读取各原训练入口和协议，不能把`build_model`改个字符串后默认获得匹配预算、匹配输入的正确对照。`no_position`并未删除几何支持构造，`no_scale`只关闭显式尺度MLP，`no_context`仍保留细区域传播。
