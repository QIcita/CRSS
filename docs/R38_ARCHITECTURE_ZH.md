# R38：CMM 三路特征的有界两两交互

R38 继承统一 FPS 输入下的 R34 主干，修改 CMM 的特征组合接口。Utonia 提供上下文，SPoTr 提供原始点邻域增量；R34 的路径条件差分消息继续在共同区域内连接表面与骨架。这里描述的是已冻结代码，不是尚未验证的新机制结论。

## 三路组合

记每个细支撑上的 96 维特征为：基础表面描述 $H$、SPoTr 增量 $D$、按 $H$ 的 RMS 校准的 Utonia 上下文 $C$。父版本的组合是 $H+0.1D+0.1C$；R38 在保留该计算的基础上增加 $HD$、$HC$、$DC$ 三项交互。

对 $x=(H,D,C)$，使用三个无偏置投影 $A_i\in\mathbb{R}^{16\times96}$，以及三个输出矩阵 $B_{ij}\in\mathbb{R}^{96\times16}$：

$$
z_i=\tanh(A_i x_i),\qquad
v_{ij}=\frac{z_i\odot z_j}{\sqrt{16}},\qquad
\widehat B_{ij}=\frac{B_{ij}}{\sqrt{1+\|B_{ij}\|_F^2}}.
$$

令 $w=(1,0.1,0.1)$、$\epsilon=10^{-6}$，计算

$$
n_i=\sqrt{\|w_i x_i\|_2^2+\epsilon^2}-\epsilon,\qquad
\Delta_{ij}=\frac{0.1}{2}(n_i+n_j)\widehat B_{ij}v_{ij}.
$$

最终为 $H+0.1D+0.1C+\Delta_{HD}+\Delta_{HC}+\Delta_{DC}$。由于 $\|v_{ij}\|_2\le1$、$\|\widehat B_{ij}\|_2<1$，可得

$$
\left\|\sum_{i<j}\Delta_{ij}\right\|_2
\le0.1(n_H+n_D+n_C)
\le0.1\sum_i\|w_i x_i\|_2.
$$

这是新增分支的幅度上界，不是泛化保证。$B_{ij}$ 从零初始化，新增 9,216 个参数；初始输出保持父模型，随后学习低秩特征交互。已有算子/梯度验收不能替代交互分支的效果消融。

实际实现：[PairFusion](../code/r38/overlay/tools/cpsf_cmm_pair_fusion/operator.py)、[H01 接入](../code/r38/overlay/tools/cpsf_cmm_pair_fusion/model.py)。组合结果同时供原表面聚合与 R34 路径消息使用，避免两个消费者读取不同版本的表面特征。

## 密集邻域修复

原 0.5µm 细支撑平均点数不足 10 时，按倍增尺度构造更密集的 SPoTr 计算分组。点坐标仍以原物理单位表达，原细支撑、点质量、Utonia 输入和骨架关联保持不变。SPoTr 输出按点所属计算分组取回，再按点质量聚合到原细支撑，之后进入上述 PairFusion 和两处消费者。

这一区分使“注意力计算覆盖的邻域”与“跨模态对齐的支撑区域”可以不同。它是计算邻域与细支撑的实现修复，不意味着所有任务必然提升。见 [数据构造](../code/r38/overlay/tools/cpsf_r38_dense/data.py)、[回投与接入](../code/r38/overlay/tools/cpsf_r38_dense/model.py)。H01 局部所有训练/开发样本均不触发扩组，已有等价性审计支持复用原 R38 结果。

## 延迟温和重加权（DRW）

DRW 不改变以上模型。FlyWire 粗分类仍用 seed42、原划分、原初始化、原样本顺序、50 轮/62,200 更新、有效 batch4、AdamW、余弦学习率和固定最终轮评估。Macro-F1 为主指标，Accuracy 为辅助指标。

仅依据训练类频次计算 $u_c=(N/(6n_c))^{1/4}$，再除以训练频率加权均值获得 $w_c$。前 30 轮权重为 1；第 31–35 轮令 $a_e=\min((e-30)/5,1)$，使用 $1+a_e(w_c-1)$，第 35 轮达到全权重，此后保持。全权重依类别顺序为 `[2.0653, 1.1785, 0.8279, 1.2476, 2.8532, 1.6835]`。

实现先计算每细胞无权重、label smoothing 0.05 的 CE，再乘目标类权重，按组内细胞数归一化；不使用会在单细胞 mean reduction 中抵消权重的写法。见 [objective.py](../code/r38/overlay/tools/cpsf_r38_delayed_reweight/objective.py) 和 [冻结协议](../code/r38/experiments/drw/protocol.json)。发布时仍在训练，CPU 损失/梯度检查和真实模型更新兼容通过，尚无效果结论。

## MICrONS 任务适配

使用完整观测神经突树，保留轴突和树突，不用基于突触的区室标签裁剪形态。R38 dense 编码器输出共同区域表征，替换全局读出为区域计数率头，以区域长度作为 exposure，预测该区域输入突触数量。训练采用每细胞平均 Poisson NLL，主指标为每细胞宏平均 Poisson deviance。见 [CountModel](../code/r38/overlay/tools/cpsf_microns/models.py)。

该任务的验证集选模及一次最终测试独立于 H01/FlyWire 的固定最终轮开发评估，不能混用评估口径。源码中的 SGFormer/Utonia 任务适配保留用于解释共享接口，本次不发布它们的结果或权重。上游官方模块贡献及许可见 [RIGHTS_AND_PROVENANCE](../RIGHTS_AND_PROVENANCE.md)。
