# CRSS: Common-Region Skeleton–Surface Encoding

神经元骨架—表面联合表征研究。**最新发布为R34阶段一（2026-10-21）**：在冻结R28上增加骨架与路径条件的区域差分消息。R28仍作为匹配参照保留。先读[START_HERE.md](START_HERE.md)。

| 内容 | 入口 |
|---|---|
| R34架构、实际公式和模块边界 | [架构说明](docs/R34_ARCHITECTURE_ZH.md) |
| 六组H01结果与机制对照 | [结果说明](docs/R34_RESULTS_ZH.md)、[JSON](code/r34/RESULTS.json) |
| 冻结源码、两项完整模型权重与组装方法 | [R34代码包](code/r34/README.md) |
| 新增模型代码 | [PathMessage / Model](code/r34/overlay/tools/cpsf_path_message/model.py) |
| R28继承主干与历史外部对照 | [R28架构](docs/R28_ARCHITECTURE_ZH.md)、[R28历史结果](docs/R28_RESULTS_ZH.md) |

## R34已完成结果

原冻结H01输入，seed42、固定最终轮开发集，各任务分别从零训练：

| 配置 | H01局部Macro-F1↑ | H01整体MAE↓ |
|---|---:|---:|
| R28参照 | 0.663894 | 0.048891 |
| **R34路径条件差分消息** | **0.681712** | **0.048690** |
| 去新增项骨架内容条件 | 0.649236 | 0.048770 |
| 普通条件注意力 | 0.654174 | 0.048830 |

**版本边界：**以上不是新几何预算＋全源FPS输入的结果；R34尚未在该输入及FlyWire/TED/TLP上完成验证。后续输入研究已把表面取样降为普通预处理，但本发布保留原冻结输入和实际计算，不追溯改写历史协议。阶段二/三仍是提案，未纳入模型。

研究主线是：CMM保留局部形态证据并补充跨尺度上下文，RCM在骨架共同区域内组织表面证据。R34新增有符号通道差分消息，不是新动态超图算子；保留R28的Utonia、SPoTr、物理位置/尺度编码与区域传播。论文草稿仍为历史版本，未自动改写。

## 校验和使用

```bash
python scripts/verify_r34.py
python scripts/materialize_r34.py /absolute/path/to/new/r34
python scripts/verify_r28.py
python scripts/verify_r9.py
python scripts/verify_export.py
```

R34发布源码经过六模型构造与对应末轮权重严格加载、六组开发预测指标复算。原数据、几何缓存和父运行路径未分发，新机器全量训练仍需配置原依赖；不绕过冻结验收。

官方来源和许可见[RIGHTS_AND_PROVENANCE.md](RIGHTS_AND_PROVENANCE.md)。[R28首页快照](docs/README_SNAPSHOT_20261020_R28.md)、R9和更早模型/论文保留，可追溯但不能代替当前版本证据。
