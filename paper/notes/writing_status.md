# 当前写作状态与证据范围

更新：2026-10-11，文献扩充与三节修订。当前正文为Abstract、Introduction、Related Work、Preliminaries、Method；实验章节和正式图表尚未加入。

## 本轮完成

- 三节实际引用43篇2023–2026正式高质量会议/期刊文献（19期刊、24会议），均有可复核来源；Method额外引用1篇InternVL技术报告，总参考文献44条。
- 26个中英文稳定编号块同步；`paragraph_manifest.json`记录源文件及双方SHA256。
- Introduction、Related Work、Preliminaries均在原CVPR模板第1–3页内结束，Method在第3页开始；参考文献单独排版。没有缩小字号或页边距。
- 方法文件为 `sec/4_method.tex`，共享定义迁入 `sec/3_preliminaries.tex`，避免重述。
- 精确引文清单、来源、阅读层级和借鉴逻辑见 `frontmatter_research_zh.md`、`neuron_literature_zh.md`、`writing_exemplars_2026_zh.md`。

## 当前论证

总问题为不同空间范围下的几何证据保留与关联。CRSS以有限带权几何及归一化区域聚合为统一数学表述，包含带权多分辨率几何编码、拓扑引导共同区域融合两个机制。既有残差、消息传递及RoPE是实现组件；没有新训练目标或尺度不变性定理。

NEURD已有mesh/skeleton分解；GraPHFormer探索late fusion和cross-attention；近期连接预测结合全局形态和局部EM。本文差异限定到已共坐标骨架与独立表面支持的共同区域联合编码，不能泛称首次多模态或首次局部—整体分析。

## 数据与结果边界

实验仍采用2026-10-10冻结审计 `CPSF_CURRENT_FIVE_TASK_20261010/results_audit.json`；本轮没有刷新训练。全部为开发验证、seed42和任务内匹配协议，`test_used=false`。五任务分别训练，非共享一套预训练权重。

H01 full为fixed_residual对照，不能混入V26门控结果；FlyWire沿用冻结V23层级；H01/NREDD保留层级修复；TED默认lr_half，TLP默认方向XYZ及原配置。H01/FlyWire为微米约定，NREDD原生单位限制保留。

同点预算：FlyWire Macro-F1提高9.91个百分点，H01整体MAE相对下降4.15%。full对mean_only：H01局部+1.92个百分点、FlyWire+8.90个百分点。区域错配分别降低5.71和10.81个百分点。这些是对应对照的结果，不能称无条件最优或所有机制独立相加。

须保留：TED full49.552%与mean_only49.520%差距极小；另一slow_decay骨架配置53.415%高于当前默认full。H01局部双去位置/尺度65.472%高于单去任一项但低于full65.911%。

## 历史记录与后续工作

`introduction_logic_zh.md`保留上一轮八段构思，最新正文已压缩为六段加贡献条目，以本状态和 `frontmatter_research_zh.md` 为准；`contribution_rationale_zh.md`保留代码解释与完整边界。

后续需要补实际主实验和外部基线表、架构图及任务适配细节；本次文献写作不能代替这些实验。CVPR2026为继承的模板标识，不代表已确定投稿届次。
