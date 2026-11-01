# 当前C版：结果与发布边界

结果快照：2026-11-01 15:12（北京时间）。默认方案为`regional_scaled_pi`；三个任务分别从头训练，共享模型设计，不是同一份训练权重跨任务迁移。

| 任务及主指标 | 原C | 最强已完成外部baseline | 原C改善 |
|---|---:|---|---:|
| H01整体 MAE↓，固定末轮开发集 | 0.05413037 | SGFormer 0.05845043 | 误差降低7.39% |
| FlyWire粗分类 Macro-F1↑，固定末轮开发集 | 83.5322% | GraPHFormer官方特征／分类头适配 80.3874% | +3.1448个百分点 |
| MICrONS细胞平均Poisson deviance↓，验证选模后测试集 | 0.28042970 | Utonia scratch R2 0.28867030 | 降低2.85% |

使用统一FPS、冻结划分、seed42和既定预算。H01整体50轮／1,000更新，FlyWire粗分类50轮／62,200更新，MICrONS 50轮／4,750更新；MICrONS按验证deviance选第33轮。机器可读指标、checkpoint摘要和协议身份见[RESULTS.json](../code/c_smm_cmm_rcm/RESULTS.json)。

三项主指标领先的是**当前已完成外部baseline**，不是所有指标或公开跨论文榜单全面SOTA：

- H01原C仍不及内部R38密集修复MAE 0.05002571；原C RMSE 0.07294311也高于GraPHFormer仅官方PI的0.07135071。
- FlyWire原C Accuracy 95.3694%，低于GraPHFormer仅官方PI的96.2019%。
- MICrONS原C总计数MAE 220.1372，高于Utonia的183.0251；deviance、密度MAE和JSD则更低。
- PTv3–MICrONS缺正式结果，已于本日启动独立补跑；LitePT-S–FlyWire和Polynormer–H01有训练态／推理态落差，尚无已验收修复结果。不能将异常低分当作这些模型本身的能力上限。

本日重新核对47条既有结果哈希；新完成的C版FlyWire保存预测独立复算覆盖1,922个开发样本，标签、混淆矩阵、Macro-F1和Accuracy一致，最终检查点及50轮采样顺序摘要匹配。未额外模型推理或新增测试集评价。

## 独立候选与实验计划

PI根条带／统一栅格候选已有H01 MAE 0.05617504、MICrONS deviance 0.29272880；快照时FlyWire尚在训练。这项输入对照不自动替换原C。

[PI公共密度／加权残差候选](../code/c_pi_basis_residual/README.md)是本次新发布的模型代码，已经通过分解与工程接口验收，**尚未正式训练，没有新的性能结果**。原C分数不转移给候选。

[论文实验方案](C_PAPER_EXPERIMENT_PLAN_20261101.md)包含三张机制表和两组可视化。12个新变体覆盖三任务，共36项计划训练，Full按身份核验后复用；新消融和图尚未完成，不提供占位成绩。
