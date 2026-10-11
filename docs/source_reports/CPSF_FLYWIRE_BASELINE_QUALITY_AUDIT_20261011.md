# FlyWire 官方基线质量排查与BN修复诊断（2026-10-11）

完成核查时间：2026-10-11T21:00:46.964955+08:00。

主要问题已经定位：逐细胞前向／梯度累积时，官方BatchNorm在训练中使用当前细胞的节点／点统计；默认eval却使用跨细胞累计running统计。FlyWire细胞形态、规模和类型差异显著，这个切换破坏了训练时学到的表征。此前基线适配遗漏了对应的训练／推理统计一致性检查，原低分不能用作方法能力或主方法优势的可靠证据。

## 固定权重、全验证集的对照结果

全部使用同一份各模型的正式最终权重、相同1922个验证细胞、seed42、原输入、原标签与任务头。没有新增优化步，未改官方数值源码，未读取测试集。每一列是完整验证集结果，不是冒烟子集。

| 官方模型＋任务适配 | 原running-BN Macro-F1↑ | 仅训练集重校准running-BN | 单细胞BN统计 Macro-F1↑ | 单细胞BN Accuracy↑ |
|---|---:|---:|---:|---:|
| Exphormer | 0.162882 | 0.137520 | **0.809131** | 0.960978 |
| LitePT-S | 0.187768 | 0.242081 | **0.655238** | 0.911030 |
| PointMamba | 0.540980 | 0.580808 | **0.852888** | 0.958897 |

训练集重校准使用固定随机512个训练细胞，不使用验证数据或标签重建统计。每个BN实际更新计数均验收到512，学习参数保持不变。校准仍明显弱于单细胞统计，因此不能把问题归结为最后几批样本造成的EMA过时，也不宜仅调整momentum或学习率来解决。

我们当前完整模型的FlyWire固定残差参考Macro-F1为0.849163（协议锁定的`CPSF_FLYWIRE_V23_CE_EXPANDED_20261003/results/coarse_full`）。PointMamba单细胞BN诊断复评为0.852888，高0.003724，即约0.37个百分点。当前证据不支持声称主方法在FlyWire上全面领先所有官方基线。诊断复评需要以独立协议记录，不能静默覆盖原正式成绩。

## 排除数据与复现错误

- 7961个train/val细胞的super_class与原始source_annotations.tsv逐一一致，coarse_target、split和legacy/added来源与候选表及冻结manifest一致。
- train/val细胞ID无交叉；骨架与表面坐标均无精确几何重复；文件哈希、几何有限性、合法边索引和正表面质量均通过。
- 输入文件字段只含几何及查询信息，未夹带类别标签。骨架nm→µm、点云µm和固定物理坐标缩放路径核对一致；仍保留原采样支持及任务输入。
- 三个模型均完成真实GPU复评；原成绩全量复现，1922个预测argmax与原文件均100%一致。数值小差异处于原生CUDA浮点执行范围，未改变预测。
- 源码／父协议哈希、权重SHA256和版本身份通过；学习参数在诊断及校准前后不变。官方主干与算子沿用上一阶段的固定Git版本。

输入差异的一个实测例子：训练集中legacy sensory的表面包围盒对角线中位数约55.7µm，而added visual_centrifugal约379.0µm；训练细胞总体对角线从约11.5µm到709.9µm。这些差异放大了逐细胞统计与跨细胞累计统计的落差，但本身是数据的真实形态特征。

## 训练集也存在推理崩落

为区分过拟合和训练／推理不一致，固定抽取每类最多12个训练细胞，共72个；验证诊断子集68个（稀有类别验证集仅8个）。下面是训练子集，不能替代上面的完整验证指标。

| 方法 | 原推理模式训练子集Macro-F1 | 单细胞统计训练子集Macro-F1 | 原生BN层数 |
|---|---:|---:|---:|
| Exphormer | 0.139957 | 1.000000 | 25 |
| LitePT-S | 0.178963 | 0.668124 | 13 |
| PointMamba | 0.612275 | 1.000000 | 2 |

Exphormer、PointMamba在已经训练过的样本上，仅切换统计方式就恢复到1.0，因此原落差的主要部分不能用验证集过拟合来解释。LitePT单细胞统计后训练子集仍只有0.668124，说明归一化修复之外仍有少数类拟合／模型适配空间，不能宣称全部质量问题已解决。

## 单细胞统计下的类别表现

| 类别 | Exphormer F1 | LitePT-S F1 | PointMamba F1 |
|---|---:|---:|---:|
| ascending | 0.785714 | 0.528302 | 0.885246 |
| central | 0.921630 | 0.846262 | 0.924528 |
| optic | 0.980998 | 0.951409 | 0.978910 |
| sensory | 0.977413 | 0.909804 | 0.963563 |
| visual_centrifugal | 0.363636 | 0.000000 | 0.571429 |
| visual_projection | 0.825397 | 0.695652 | 0.793651 |

例如sensory：Exphormer由原0.013559恢复到0.977413，PointMamba由原0.028490恢复到0.963563。这进一步支持原sensory→optic混淆的主要部分来自归一化不一致。LitePT的visual_centrifugal仍为0，长尾能力仍需合理调参／适配验证。

## 可复用的修复适配

已实现`tools/cpsf_native_cell_bn_policy_20261011.py`：先严格加载原权重，再对原生BatchNorm关闭running统计、清空对应运行缓冲；正常model.eval()仍关闭dropout，而BN使用当前细胞自身的节点／点统计。没有替换成自写归一化层，也没有移除官方注意力、SSM、稀疏卷积或层级模块。

此规则是明确声明的本任务归一化适配，不冒充作者的默认eval配置。单细胞统计仅在当前样本内部计算，未混合验证细胞，也未使用验证标签更新参数。

实际官方模型契约验证已通过：

- 修复适配的普通eval与诊断BN模式前向一致，最大误差不超过约7.2e-7。
- 原训练方式与修复后的前向、参数梯度一致；148／201／156组参数梯度逐项核对，最大差约2.6e-6。
- 原正式checkpoint文件哈希不变，seed、输入、模型维度、任务头和更新预算不变。

下一步应在独立的cell-BN评测协议下，对H01与FlyWire的各官方基线统一复评；OctFormer、PTv3完成后也需检查同类风险。保留原running-BN结果作为原适配记录；不能只对验证成绩较差的模型选择性修复，也不能用诊断最高分选择checkpoint或统计规则。

本次未改动正在运行的正式队列，核查时9项完成、OctFormer FlyWire运行、PTv3 FlyWire排队、0项失败。新规则以独立适配文件及复评身份保存，没有覆写当前队列的冻结协议。

## 审计记录与版本边界

- 原正式协议SHA256：`13a24eda2637eb327582a04fc7414f125542ac95a9178b084439c56e24e9866b`。
- 数据manifest SHA256：`59dd3ad58641dfb4dcfdcd47255084730b800111088e415be2d7bb77bea215b6`。
- 原始记录：`CPSF_EXTERNAL_BASELINES_PHASE2_20261010/results/`。
- 诊断记录：`CPSF_EXTERNAL_BASELINES_DIAGNOSTICS_20261011/run1/`，仅使用attempt_validity.json中列出的有效模式。
- 首轮校准测试发现重置顺序错误：恢复track_running_stats之前调用reset导致计数未清零。该列作废，现场保留；本文校准列全部来自修正后的run2，并验收各BN计数恰为512。其他模式不受此校准错误影响。
- 有效校准记录：`CPSF_EXTERNAL_BASELINES_DIAGNOSTICS_20261011/run2/`。
- 最终审计：[final_audit.json](/Users/fengqi/Documents/ChatGPT/神经元形态分析/artifacts/CPSF_EXTERNAL_BASELINES_DIAGNOSTICS_20261011/final_audit.json)。
- 修复复评身份：[cell_bn_replay_protocol.json](/Users/fengqi/Documents/ChatGPT/神经元形态分析/artifacts/CPSF_EXTERNAL_BASELINES_DIAGNOSTICS_20261011/cell_bn_replay_protocol.json)。
- 可复用适配：[cpsf_native_cell_bn_policy_20261011.py](/Users/fengqi/Documents/ChatGPT/神经元形态分析/tools/cpsf_native_cell_bn_policy_20261011.py)。

本地artifacts目录保存三种统计方式的全量预测、训练／验证诊断子集、数据与原始注释审计、原生前向／梯度契约及日志。新的cell-BN复评身份明确记录父协议、权重和预测哈希；没有将其归类为原始running-BN正式分数。
