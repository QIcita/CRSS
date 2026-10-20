# R28五任务结果与基线对比

截至 **2026-10-20 09:41 北京时间**。R28主配置为`CPSF_RAW_POINT_ATTENTION_20261019_R28_A1/r9_lpa_attention`，三任务执行根为`CPSF_R28_TASK_EXPANSION_20261020_A2/r28_lpa_attention`。两个名字指同一个R28局部注意力机制，不含R33。所有数字为固定seed42、固定最终轮的**开发集**成绩；无外部预训练，不使用test反馈。五任务分别从零训练，不共享一份已训练权重。

## 完整比较

| 方法 | 输入 | H01局部F1↑ | H01整体MAE↓ | FlyWire粗类F1↑ | TED正类F1↑ | TLP缺边Macro-AP↑ |
|---|---|---:|---:|---:|---:|---:|
| SGFormer（NeurIPS’23） | 骨架图 | 0.486490 | 0.058459 | 0.652894 | 0.442886 | 0.551245 |
| Polynormer（ICLR’24） | 骨架图 | 0.532291 | 0.119943 | 0.761603 | 0.336623 | 0.505180 |
| PTv3（CVPR’24） | 表面点云 | 0.587514 | 0.070318 | 0.584003 | — | — |
| LitePT-S（CVPR’26） | 表面点云 | 0.634839 | 0.082825 | 0.655238 | — | — |
| OctFormer（ACM TOG / SIGGRAPH’23） | 表面点云 | 0.472227 | 0.084962 | 0.583756 | — | — |
| Utonia scratch R2（ICML’26） | 表面点云 | 0.620823 | 0.076477 | 0.785536 | — | — |
| GraPHFormer-adapted（CVPR’26） | 骨架＋衍生拓扑图像＋半径 | 0.637270 | 0.065709 | 0.766696 | — | — |
| TED-Net（Pattern Recognition’26） | 骨架＋SPE图像 | — | — | — | 0.569248 | — |
| TLP-Net（Pattern Recognition’26）† | 骨架＋SPE图像 | — | — | — | — | 0.108438 |
| R9（历史参照） | 骨架＋真实表面点云 | 0.667901 | 0.050277 | 0.783810 | 0.490982 | 0.566934 |
| R28（当前参照） | 骨架＋真实表面点云 | 0.663894 | 0.048891 | 训练中 | 训练中 | 0.569710 |

† TLP-Net为**缺边目标适配版、无Rule**。正式来源是Pattern Recognition 2026，不能将其0.108438与论文完整邻接AP 0.7751直接比较。恢复官方损失、保持同一空间划分和训练预算的独立对照，开发集缺边Macro-AP为0.010333，完整邻接Macro-AP为0.780942；详见[TLP评测专项审计](TLP_NET_EVALUATION_AUDIT_ZH.md)。

出处已按作者仓库、CVF正式页面及Crossref核查：[来源清单](../code/r28/evidence/baseline_publications.json)。SGFormer是**NeurIPS’23**；OctFormer是**ACM TOG / SIGGRAPH’23**。会议/期刊标注说明方法来源，scratch和adapted标注仍保留，不表示复现了原论文排行榜。2026-10-20本次仅更新出处与审计说明；表中数值和“训练中”沿用上述09:41快照，不冒充实时训练状态。


“—”表示该baseline没有对应正式实验；“训练中”表示R28尚无固定末轮成绩，不能用R9或旧完整模型填入。PointMamba和Exphormer按用户决定不作为baseline。表中基线是官方主干＋项目任务适配，不是重现各论文原始排行榜。Utonia只纳入从零训练R2；GraPHFormer-adapted不用外部DINO预训练。点云基线没有正式TED/TLP训练结果，不能用内部单模态消融替代。

## 当前能作出的判断

- H01局部：R28 **0.6638936372**，高于当前最强纳入外部基线GraPHFormer-adapted **0.6372700855**，提高 **2.662个百分点**；相对R9 **0.6679012346**，降低 **0.401个百分点**。
- H01整体：R28 MAE **0.0488909427**，优于最强外部基线SGFormer **0.0584590187**，下降 **0.0095680760，约16.37%**；也优于R9 **0.0502774163**，下降 **0.0013864736**。
- TLP：R28 Missing-edge Macro-AP **0.5697102714**，高于SGFormer **0.5512450621**，提高 **1.847个百分点**；相对R9补评 **0.5669343012**，提高 **0.278个百分点**。Micro-AP为 **0.3878511334**，62张开发图、1,146,183个未观察无序对、283条缺失正边。
- FlyWire、TED：R28还在训练，暂不能判断相对baseline是否有收益。R9该两项成绩只是历史参照，不是R28结果。

**已完成三项均优于当前纳入的外部基线；R28没有全面超越R9，也尚不具备五任务领先证据。** H01整体仅24个开发细胞，规模本身也应随分数一起展示。

## R28自己的现有对照

| 配置 | H01局部Macro-F1↑ | H01整体MAE↓ |
|---|---:|---:|
| R9，无新增SPoTr局部支路 | 0.6679012346 | 0.0502774163 |
| R28 uniform，固定点质量汇聚 | 0.6495901318 | 0.0493367792 |
| R28 attention，通道注意力 | 0.6638936372 | 0.0488909427 |

attention两项优于同接口uniform，但相对R9只有整体提升。uniform注册同样模块，attention评分参数不参与有效梯度，因此不是严格同有效容量对照。位置、尺度、采样、单模态等R28专属完整消融未完成，不能拿旧版本矩阵作为R28消融。原始精确数值以发布metrics为准。

## 五任务进度

| 任务 | 状态 | 当前轮/最终轮 | 已记录优化器更新 |
|---|---|---:|---:|
| h01_local | completed | 25/25 | 2250 |
| h01_global | completed | 50/50 | 1000 |
| flywire_coarse | training | 17/50 | 19931 |
| ted | training | 10/30 | 3541 |
| tlp | completed | 80/80 | 12480 |

表为带时间快照，训练仍会继续；未发布运行中的latest.pt或优化器状态。完整机器可读信息见[RESULTS.json](../code/r28/RESULTS.json)。

## 可比性与独立复核

本轮直接核对五任务R28与原R9的**全部manifest records逐项相同**：H01局部930例（716/214）、整体104例（80/24）、FlyWire7961例（6039/1922），TED34729事件（23967/10762）、TLP218图（156/62）。R9与外部基线的输入谱系/标签/划分审计继续沿用，R28再次确认没有改观察支撑和标签。不是重新生成全量原mesh的审计；相同观测支撑也不能证明R28采样优于baseline采样，因为点云baseline复用该支撑。

27项外部基线的正式指标文件和末轮权重SHA本轮重新核对，均与保留审计一致。H01三种含BN点云基线沿用run3单样本BN复评；FlyWire沿用已确认的单样本BN口径；没有混入旧running-BN成绩。其他模型依各自声明的规范运行，不声称输入张量或适配器完全相同。

TED主表保留原正式适配结果（sigmoid-round-argmax）；另外提供统一logit argmax口径：SGFormer **0.442442**、Polynormer **0.336449**、TED-Net **0.568330**。R28按argmax评测，完成后必须同时展示该统一口径，不按分数挑规则。精确值见BASELINE_COMPARISON.json中的common_argmax_f1。

TLP评价全部未观察无序节点对，不是候选或全矩阵AP。R28完成80轮12480更新；独立重算62张图的Macro/Micro-AP，target/observed逐图与原源一致，final/latest末轮模型与优化器步数一致。保留原FP32 i<j分数，仅镜像未用于主指标的下三角。R9对应有效结果来自独立补评根`CPSF_R9_TLP_EVAL_REPAIR_20261018_A1`，原训练目录的评测失败未被抹去。TLP-Net是匹配缺边目标的任务适配，不能称原论文原协议复现。

## 证据入口

- [R28完整结构](R28_ARCHITECTURE_ZH.md)、[R28源代码说明](../code/r28/README.md)。
- [完整对照JSON](../code/r28/BASELINE_COMPARISON.json)、[CSV](../code/r28/BASELINE_COMPARISON.csv)。
- [本轮现场来源/输入谱系及H01预测复核](../code/r28/live_export.json)、[27项基线指标/权重复核](../code/r28/evidence/baseline_live_audit.json)。
- [H01完整独立审计](../code/r28/evidence/h01_independent_final_audit.json)、[TLP末轮/预测/监督复核](../code/r28/evidence/completed_status_audit.json)。

已发布H01两项和TLP三个本项目训练的final.pt，不含外部预训练或运行中优化器权重。FlyWire/TED以后完成时另行更新；当前发布不更改现有训练或恢复小时自动化。
