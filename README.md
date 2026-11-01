# CRSS: Common-Region Skeleton–Surface Encoding

神经元骨架—表面联合表征研究。**当前默认方案为 C版渐进式 SMM–CMM–RCM（`regional_scaled_pi`）**，已完成H01整体、FlyWire粗分类和MICrONS训练。2026-11-01新增PI公共密度／加权残差候选：代码与工程验收已完成，尚未正式训练，不能继承原C成绩。

| 内容 | 入口 |
|---|---|
| 当前C版架构、源码与组装 | [C模型包](code/c_smm_cmm_rcm/README.md) |
| 三任务当前结果与评价边界 | [结果说明](docs/C_CURRENT_RESULTS_20261101.md)、[结果身份](code/c_smm_cmm_rcm/RESULTS.json) |
| 最新PI候选：公共密度／加权残差 | [候选源码与组装](code/c_pi_basis_residual/README.md) |
| 三张机制消融表与两组可视化设计 | [论文实验方案](docs/C_PAPER_EXPERIMENT_PLAN_20261101.md) |
| R38 结构、计算公式与实现 | [架构说明](docs/R38_ARCHITECTURE_ZH.md) |
| R38 已完成结果与未完成任务 | [结果与状态](docs/R38_RESULTS_ZH.md)、[机器可读记录](code/r38/RESULTS.json) |
| 冻结源码、五份完成权重、协议和验收 | [R38 发布包](code/r38/README.md) |
| R34 密集邻域历史发布 | [结构](docs/R34_DENSE_SUPPORT_ZH.md)、[代码包](code/r34_dense/README.md) |
| R28 / 原输入 R34 / R9 历史包 | [R28](code/r28/README.md)、[R34](code/r34/README.md)、[R9](code/r9/README.md) |
| 2026-10-23 统一 FPS 结果快照 | [历史汇总](docs/UNIFIED_FPS_RESULTS_20261023.md) |

C版H01整体开发集MAE **0.05413037**、FlyWire粗分类开发集Macro-F1 **83.5322%**、MICrONS验证选模后测试集cell-macro Poisson deviance **0.28042970**。三项主指标均优于当前已完成外部baseline；H01仍不及内部历史R38密集修复版，辅助指标也未全部领先。这里是本项目匹配协议下的比较，不是公开跨论文榜单SOTA。

```bash
python code/c_smm_cmm_rcm/materialize.py
python code/c_pi_basis_residual/materialize.py /absolute/path/to/new/c_candidate
```

C与PI候选发布模型源码、组装器和必要说明，不新增数据、缓存、权重或运行中优化器断点。三张消融表去重后是36项计划训练，尚未执行，不将计划填作结果。历史R28/R34/R38及其已发布文件保留；官方组件来源与许可见 [START_HERE](START_HERE.md) 和 [来源与许可](RIGHTS_AND_PROVENANCE.md)。
