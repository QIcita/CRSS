> **版本提示（2026-10-17）：本页描述历史版本。当前R9请先读 [架构](R9_ARCHITECTURE_ZH.md)、[结果](R9_RESULTS_ZH.md)和[代码](../code/r9/README.md)，不要混用模型与成绩。论文草稿尚未按R9更新。**

# 实验索引：冻结主模型与后续基线复评

主结果来源为2026-10-10的[原始审计JSON](../evidence/five_task_20261010/results_audit.json)。以下表格直接从该文件生成，没有重新计算或刷新训练成绩。全部使用seed42开发验证，`test_used=false`。

## 三个数据集、五个任务

| 任务 | 预测对象 | train / val | 主指标 |
|---|---|---:|---|
| H01局部 | AIS/axon/dendrite区室三分类 | 716 / 214事件，按细胞划分 | Macro-F1↑ |
| H01整体 | 整细胞兴奋性输入比例 | 80 / 24细胞 | MAE↓ |
| FlyWire粗类 | 六个super_class | 6039 / 1922细胞 | Macro-F1↑ |
| NREDD TED | 重建错误节点/误连定位事件 | 23967 / 10762事件 | 正类F1↑ |
| NREDD TLP | 未观测节点对连边 | 156 / 62图 | Missing-edge macro AP↑ |

FlyWire六类为ascending、central、optic、sensory、visual_centrifugal、visual_projection。当前不是细cell_type分类或检索主表。TLP为图输入、节点对输出，不是整细胞分类。

## 当前默认配置的完整与消融

除H01整体MAE外，以下以百分数展示；缺项为此审计没有对应当前版本的结果，不用旧配置回填。

| arm | H01局部F1 | H01整体MAE | FlyWire F1 | TED F1 | TLP AP |
|---|---:|---:|---:|---:|---:|
| full | 65.911 | 0.051048 | 84.916 | 49.552 | 58.365 |
| mean_only | 63.996 | 0.051701 | 76.021 | 49.520 | 57.241 |
| no_fusion | 61.625 | 0.056338 | 76.126 | 48.852 | 55.206 |
| skeleton | 54.837 | 0.064829 | 76.380 | 46.933 | 55.979 |
| point | 57.660 | 0.089705 | 72.545 | 36.416 | 26.878 |
| no_scale | 60.752 | 0.063896 | 75.141 | 48.708 | 56.816 |
| no_position | 62.596 | 0.064304 | 67.681 | 48.170 | — |
| no_position_scale | 65.472 | 0.056835 | 73.315 | — | — |
| thumbnail | — | 0.080953 | 73.071 | — | — |
| uniform_budget | — | 0.053257 | 75.010 | — | — |
| shuffle_regions | 60.204 | 0.054544 | 74.109 | — | — |
| no_context | 63.025 | 0.077107 | 81.223 | — | — |

## 身份与比较边界

H01 full来自V26目录的`fixed_residual`对照，不是V26门控full。H01/NREDD保留minimum-four层级修复；FlyWire沿用冻结层级。TED采用`lr_half`，TLP采用`directional_xyz`＋原训练配置。准确checkpoint及协议SHA256见[DEFAULTS.json](../code/DEFAULTS.json)。

- TED full相对mean_only仅约0.032个百分点；slow_decay骨架配置F1为53.415%，高于所选默认full。不能声称等搜索后full最优。
- H01局部位置/尺度双去除优于单去任一项但低于full，说明作用不是独立线性可加。
- NREDD表面分支仍保留任务定义所需的观测查询位置和样本支持；不能说完全脱离骨架。
- `mean_only`保留共同区域及上下文，只移除学习交互；`no_fusion`是两路独立处理后汇合。
- 对同点预算采样、区域错配等机制检验的详细解读见[10月10日完整报告](source_reports/CPSF_CURRENT_RESULTS_AND_METHOD_REVIEW_20261010.md)。

## 10月11日官方基线复评：独立于上表

官方基线逐细胞训练与默认running-BN推理存在统计不一致，原低分不能作为可靠的主方法优势证据。固定原权重、输入和验证集，使用当前样本统计进行独立协议复评，没有增加训练步或读取测试集，也没有覆盖原成绩。

| 模型 | H01局部 cell-BN Macro-F1 | H01整体 cell-BN MAE |
|---|---:|---:|
| Exphormer | 0.442838 | 0.060130 |
| PTv3 | 0.587514 | 0.070318 |
| LitePT-S | 0.634839 | 0.082825 |
| OctFormer | 0.472227 | 0.084962 |
| PointMamba | 0.618989 | 0.073551 |
| 保留完整模型，无BN | 0.659113 | 0.051048 |

证据：[H01审计报告](source_reports/CPSF_H01_BN_AND_INPUT_SCOPE_AUDIT_20261011.md)、[复评summary.json](../evidence/h01_bn_20261011/run3/summary.json)、[final_audit.json](../evidence/h01_bn_20261011/final_audit.json)。

FlyWire在已完成复评的三个基线上，cell-BN Macro-F1为Exphormer 0.809131、LitePT-S 0.655238、PointMamba 0.852888；完整模型参考0.849163。**PointMamba高约0.37个百分点，不能声称本方法全面领先官方基线。** 见[FlyWire审计报告](source_reports/CPSF_FLYWIRE_BASELINE_QUALITY_AUDIT_20261011.md)。这些是报告时间的快照，不代表之后所有队列状态。

复评保留原生官方数值实现。cell-BN是明示的任务适配，不能写成作者原默认推理结果；H01局部只用当前事件的局部输入统计，不使用隐藏完整细胞。未来独立测试前需冻结统一规则。

## 论文尚未完成的部分

当前LaTeX尚无完整实验章节、正式主表和架构图。主机制消融与不同任务适配已具备记录，但当前默认版本下的缺项、外部基线覆盖和最终独立测试仍须按实际证据补齐，不能因已有引言而视作实验完成。
