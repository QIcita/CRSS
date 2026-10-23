# CRSS: Common-Region Skeleton–Surface Encoding

神经元骨架—表面联合表征研究。**最新发布为统一 FPS 输入下的 R34 密集邻域修正版（2026-10-23）**：扩大 SPoTr 的计算邻域，保留原细粒度证据支撑和骨架关联。

| 内容 | 入口 |
|---|---|
| 当前结构、计算公式与边界 | [R34 密集邻域说明](docs/R34_DENSE_SUPPORT_ZH.md) |
| 冻结源码、最终权重、协议和验收 | [R34 dense 发布包](code/r34_dense/README.md) |
| 所有已完成统一 FPS 实验 | [结果汇总](docs/UNIFIED_FPS_RESULTS_20261023.md)、[逐项审计 JSON](results/unified_fps_20261023/completed_results_audit.json) |
| 历史 R34 阶段一（不同输入） | [旧首页快照](docs/README_SNAPSHOT_20261021_R34.md)、[旧代码包](code/r34/README.md) |
| 历史 R28/R9 | [R28](code/r28/README.md)、[R9](code/r9/README.md) |

本版 H01 整体 MAE **0.050024**，原统一 FPS R34 为 0.051537；seed42、50轮/1000更新、固定最终轮开发集，无外部预训练。H01 局部仅完成规则不触发与两样本前向兼容性检查，不能写成本版局部正式重训成绩。R38＋密集邻域为独立候选训练，不归入本版。

CMM 保留局部表面形态并补充上下文，RCM 通过骨架共同区域及路径条件差分消息组织表面证据。官方 Utonia/SPoTr 贡献和许可继续保留，表面 FPS 取样视为输入预处理。

```bash
python scripts/verify_r34_dense.py
python scripts/materialize_r34_dense.py /absolute/path/to/new/r34_dense
```

发布包核验了源文件哈希、模型构造、最终权重严格加载和保存预测的指标复算。原始数据/缓存不分发，完整训练仍需原依赖及父版本验证文件；不绕过冻结门槛。历史发布均独立保留。见 [START_HERE](START_HERE.md) 和 [来源与许可](RIGHTS_AND_PROVENANCE.md)。
