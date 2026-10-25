# CRSS: Common-Region Skeleton–Surface Encoding

神经元骨架—表面联合表征研究。**最新代码发布为 R38（2026-10-25）**：在 R34 的 CMM 三路组合中加入有界低秩两两交互，包含密集邻域修复与延迟温和重加权训练分支。**R28 仍是总体研究主参照**；版本发布不代表全任务效果升级。

| 内容 | 入口 |
|---|---|
| R38 结构、计算公式与实现 | [架构说明](docs/R38_ARCHITECTURE_ZH.md) |
| R38 已完成结果与未完成任务 | [结果与状态](docs/R38_RESULTS_ZH.md)、[机器可读记录](code/r38/RESULTS.json) |
| 冻结源码、五份完成权重、协议和验收 | [R38 发布包](code/r38/README.md) |
| R34 密集邻域历史发布 | [结构](docs/R34_DENSE_SUPPORT_ZH.md)、[代码包](code/r34_dense/README.md) |
| R28 / 原输入 R34 / R9 历史包 | [R28](code/r28/README.md)、[R34](code/r34/README.md)、[R9](code/r9/README.md) |
| 2026-10-23 统一 FPS 结果快照 | [历史汇总](docs/UNIFIED_FPS_RESULTS_20261023.md) |

R38 密集版 H01 整体开发集 MAE **0.05002571**，FlyWire 粗分类开发集 Macro-F1 **74.2621%**；MICrONS 测试集 cell-macro Poisson deviance **0.30222369**。H01 局部通过等价性审计复用原 R38。FlyWire 细分类已暂停，DRW 粗分类在导出时训练中，二者无最终成绩。

```bash
python scripts/verify_r38.py
python scripts/materialize_r38.py /absolute/path/to/new/r38 --variant drw
```

已核验冻结源码哈希、五份权重严格加载与保存预测复算。原始数据/缓存和运行中优化器断点不分发；完整训练仍需原依赖、数据及父版本验收文件。官方 Utonia/SPoTr 来源与许可保留。见 [START_HERE](START_HERE.md) 和 [来源与许可](RIGHTS_AND_PROVENANCE.md)。
