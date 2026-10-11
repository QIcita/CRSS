# 三节正文文献基础与写作修订记录

日期：2026-10-11。交付范围：Introduction、Related Work、Preliminaries，英文Overleaf及中文逐段同步。当前模型、实验版本与训练协议不变。

## 检索范围与纳入标准

“近四年”按2023–2026正式发表／官方会议录用记录核验。共核验47篇不同论文，正文三节实际引用43篇：19篇期刊、24篇会议；此外Method保留InternVL技术报告作为采样组织启发，不计入43篇正式论文门槛。没有使用nocite或只放BibTeX的方法凑数量。

检索分为神经元任务与数据背景、点几何与采样预算、2026区域/位置/跨模态机制三组。访问CVF官方2023–2025目录筛选论文，通过正式论文页读取摘要与BibTeX；神经元组用EuropePMC、Crossref、Nature/Science等正式来源；2026组使用会议官网逐项核验。原始响应及全文阅读片段保存在本目录。部分来源只读摘要，以记录的阅读级别约束论断，不将“摘要未提”写成“全文不存在”。

ICCV为双年会议，不存在ICCV2026届；因此ICCV检索采用近四年已举行届次，2026对应ECCV。官方NeurIPS2026页面目前已可核验，先前“尚未公布”的旧笔记不再适用。仅确认为官方Oral/Spotlight/Highlight的条目带此标签。

## 知识沉淀：哪些认识改变了稿件

1. **任务意义要建立在重建后分析活动上。** H01、FlyWire和MICrONS等提供丰富几何与注释；校对、补全、区室判断和细胞属性是不同粒度的分析需求。MICrONS和其他物种资源只用于背景引用，没有加入当前三数据集训练。
2. **领域已有局部—整体、多模态和骨架—网格关联。** NEURD已有非分叉段图及mesh-skeleton分解，TMI2026段连接方法已有全局形态与局部EM的cross-attention；GraPHFormer明确测试late fusion与cross-attention。缺口必须限定到观测骨架与原生表面、有限预算、共同区域联合编码，不能泛称“前人没有融合”。
3. **覆盖、密度、物理尺寸、感受野是不同变量。** Utonia区分域间几何差异；APES讨论点分配；PTv3/LitePT等讨论算子与上下文。本文先问哪些观测进入模型，再问它们如何交互；没有把更大注意力范围等同于更完整的输入。
4. **数学预备知识用来定义计算，而非给现有模型套理论保证。** CAJAL的度量几何、VesselTok的中心线tokens、LEAP的结构编码有各自输入与目标。本文有限测度只精确描述实际带权区域求和；不能借用它们的结论宣称采样不变、物理尺度不变或区域最优。

## 2026官方荣誉核验

| 论文 | 官方标识 | 本文使用方式 | 官方来源 |
|---|---|---|---|
| LitePT: Lighter Yet Stronger Point Transformer | CVPR 2026 Highlight | 正文引用 | [官方页面](https://cvpr.thecvf.com/virtual/2026/poster/37647) |
| MSPT: Efficient Large-Scale Physical Modeling via Parallelized Multi-Scale Attention | CVPR 2026 Highlight | 正文引用 | [官方页面](https://cvpr.thecvf.com/virtual/2026/poster/38086) |
| SuP: Sub-cloud Driven Point Cloud Registration | CVPR 2026 Highlight | 正文引用 | [官方页面](https://cvpr.thecvf.com/virtual/2026/poster/38606) |
| Multi-Domain Riemannian Graph Gluing for Building Graph Foundation Models | ICLR 2026 Oral | 保留检索知识，未强行纳入正文 | [官方页面](https://iclr.cc/virtual/2026/oral/10010538) |
| Learning with Dual-level Noisy Correspondence for Multi-modal Entity Alignment | ICLR 2026 Oral | 保留检索知识，未强行纳入正文 | [官方页面](https://iclr.cc/virtual/2026/oral/10007523) |
| URoPE: Universal Relative Position Embedding across Geometric Spaces | ECCV 2026 Spotlight | 正文引用 | [官方页面](https://eccv.ecva.net/virtual/2026/spotlight/6035) |
| VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation | ECCV 2026 Spotlight | 正文引用 | [官方页面](https://eccv.ecva.net/virtual/2026/spotlight/6052) |
| Graph Cascades: Contagion-Based Mesoscopic Rewiring for Structure-Aware Graph Machine Learning | NeurIPS 2026 Oral | 保留检索知识，未强行纳入正文 | [官方页面](https://neurips.cc/virtual/2026/oral/155098) |

Graph Cascades、GraphGlue和RULE虽有官方荣誉，但与当前输入假设距离较远，未为凑会议覆盖而加入正文。正文相关性优先于奖项；普通Poster不升级为Spotlight。

## 实际全文阅读及写作借鉴

本次读取LitePT、MSPT、Utonia、FusionCell的完整Introduction；神经元侧读取NEURD的Main/Graph decomposition和MorphRep的Introduction及相关Methods。另沿用用户指定HDF论文的已读§1作为总体逻辑参照。阅读来源与逐段归纳见 `writing_exemplars.md` 和 `neuron_synthesis.md`。

| 写作位置 | 原文实践与定位 | 本稿具体修订 | 适用边界 |
|---|---|---|---|
| Introduction的问题 | Utonia §1分开尺度/密度/通道；MSPT §1解释局部与长程需求；NEURD Main先讲分析资源的用途 | 从任务价值进入证据保留、证据关联两个递进问题，说明遗漏和对应缺失的后果 | 不照搬跨域预训练、PDE或软件平台主张 |
| 框架承接与贡献 | LitePT §1先说明组件分工；FusionCell §1说明每路提供什么及交互位置 | 两机制分别改变采样/编码与区域内交互；联合token再进上下文 | 不把残差、RoPE列独立创新，不声称最优分工已获证明 |
| Related Work / Preliminaries | 相关方法的具体输入和计算假设；oral-skill“条件跟随结论” | Related Work按资源、形态表示、预算、对应与先验比较；Preliminaries仅定义G/P/q与有限测度聚合 | 不将参考文献当作我们模型的实验证明；未知全文机制不作排除判断 |

## 三节分工与长度控制

- Introduction：任务价值→已有进展→两个问题→统一表示→CRSS两机制→实验与贡献。
- Related Work：比较输入对象、交互位置和适用假设，尤其明确NEURD、GraPHFormer、段连接预测的已完成内容。
- Preliminaries：输入、查询粒度、坐标单位、目标信息隔离，以及归一化带权区域聚合。
- 具体tile数量、软关联高斯核、MLP、残差及上下文保留到Method；旧Method中的定义移至Preliminaries，避免重复。
- 页数使用原CVPR 10pt双栏review模板实测。三节在PDF第1–3页内完成，Method在第3页开始。未缩小字号、页边距或修改行距；参考文献不计入三节正文。

## 本轮纳入的43篇正式文献

年份分布：{2023: 12, 2024: 9, 2025: 9, 2026: 13}。表中阅读层级明确“摘要”和“全文片段”的区别。所有BibTeX键都出现在三节正文。

| 引文键 | 论文 | 出版 | 证据来源与阅读级别 |
|---|---|---|---|
| `schlegel2024flytyping` | Whole-brain annotation and multi-connectome cell typing of Drosophila | Nature 2024 | [来源](https://doi.org/10.1038/s41586-024-07686-5)；indexed abstract and publisher-deposited Crossref metadata |
| `dorkenwald2024flywire` | Neuronal wiring diagram of an adult brain | Nature 2024 | [来源](https://doi.org/10.1038/s41586-024-07558-y)；indexed abstract and publisher-deposited Crossref metadata |
| `shapsoncoe2024h01` | A petavoxel fragment of human cerebral cortex reconstructed at nanoscale resolution | Science 2024 | [来源](https://doi.org/10.1126/science.adk4858)；indexed abstract and publisher-deposited Crossref metadata |
| `microns2025functional` | Functional connectomics spanning multiple areas of mouse visual cortex | Nature 2025 | [来源](https://doi.org/10.1038/s41586-025-08790-w)；indexed abstract and publisher-deposited Crossref metadata |
| `govek2023cajal` | CAJAL enables analysis and integration of single-cell morphological data using metric geometry | Nature Communications 2023 | [来源](https://doi.org/10.1038/s41467-023-39424-2)；indexed abstract and publisher-deposited Crossref metadata |
| `celii2025neurd` | NEURD offers automated proofreading and feature extraction for connectomics | Nature 2025 | [来源](https://doi.org/10.1038/s41586-025-08660-5)；indexed abstract + full text Europe PMC introduction/method sections |
| `winding2023insect` | The connectome of an insect brain | Science 2023 | [来源](https://doi.org/10.1126/science.add9330)；indexed abstract and publisher-deposited Crossref metadata |
| `schneider2025inhibitory` | Inhibitory specificity from a connectomic census of mouse visual cortex | Nature 2025 | [来源](https://doi.org/10.1038/s41586-024-07780-8)；indexed abstract and publisher-deposited Crossref metadata |
| `gou2024gapr` | Gapr for large-scale collaborative single-neuron reconstruction | Nature Methods 2024 | [来源](https://doi.org/10.1038/s41592-024-02345-z)；indexed abstract and publisher-deposited Crossref metadata |
| `shah2026graphformer` | GraPHFormer: A Multimodal Graph Persistent Homology Transformer for the Analysis of Neuroscience Morphologies | CVPR 2026 | [来源](https://cvpr.thecvf.com/virtual/2026/poster/36424)；official event abstract; previously inspected author full-text sections 3.1–3.2 and fusion supplement |
| `zhao2024macgnn` | Graph Representation Learning for Large-Scale Neuronal Morphological Analysis | IEEE Transactions on Neural Networks and Learning Systems 2024 | [来源](https://doi.org/10.1109/tnnls.2022.3204686)；indexed abstract and publisher-deposited Crossref metadata |
| `fan2024morphrep` | Learning meaningful representation of single-neuron morphology via large-scale pre-training | Bioinformatics 2024 | [来源](https://doi.org/10.1093/bioinformatics/btae395)；indexed abstract + full text Europe PMC introduction/method sections |
| `xiong2024dsm` | DSM: Deep sequential model for complete neuronal morphology representation and feature extraction | Patterns 2024 | [来源](https://doi.org/10.1016/j.patter.2023.100896)；indexed abstract and publisher-deposited Crossref metadata |
| `sheng2025sgtmorph` | Self-Supervised Neuron Morphology Representation With Graph Transformer | IEEE Transactions on Medical Imaging 2025 | [来源](https://doi.org/10.1109/tmi.2025.3590484)；indexed abstract and publisher-deposited Crossref metadata |
| `nern2025visualinventory` | Connectome-driven neural inventory of a complete visual system | Nature 2025 | [来源](https://doi.org/10.1038/s41586-025-08746-0)；indexed abstract and publisher-deposited Crossref metadata |
| `zhu2023morphognn` | Data-Driven Morphological Feature Perception of Single Neuron With Graph Neural Network | IEEE Transactions on Medical Imaging 2023 | [来源](https://doi.org/10.1109/tmi.2023.3275209)；indexed abstract and publisher-deposited Crossref metadata |
| `dorkenwald2025cave` | CAVE: Connectome Annotation Versioning Engine | Nature Methods 2025 | [来源](https://doi.org/10.1038/s41592-024-02426-z)；indexed abstract and publisher-deposited Crossref metadata |
| `sheng2026neurogt` | NeuroGT: Biophysically grounded graph transformers for self-supervised representation learning of neuronal morphology | Medical Image Analysis 2026 | [来源](https://doi.org/10.1016/j.media.2026.104065)；indexed abstract and publisher-deposited Crossref metadata |
| `chen2026segmentconnectivity` | Neuron Segment Connectivity Prediction With Multimodal Features for Connectomics | IEEE Transactions on Medical Imaging 2026 | [来源](https://doi.org/10.1109/tmi.2026.3658169)；indexed abstract and publisher-deposited Crossref metadata |
| `liao2025neunet` | Building connectome analysis tools with representation learning on neuronal skeleton and circuit topology | Neural Networks 2025 | [来源](https://doi.org/10.1016/j.neunet.2025.107603)；indexed abstract and publisher-deposited Crossref metadata |
| `matr2023` | 2D3D-MATR: 2D-3D Matching Transformer for Detection-Free Registration Between Images and Point Clouds | ICCV 2023 | [来源](https://openaccess.thecvf.com/content/ICCV2023/html/Li_2D3D-MATR_2D-3D_Matching_Transformer_for_Detection-Free_Registration_Between_Images_and_ICCV_2023_paper.html)；official proceedings abstract |
| `spt2023` | Efficient 3D Semantic Segmentation with Superpoint Transformer | ICCV 2023 | [来源](https://openaccess.thecvf.com/content/ICCV2023/html/Robert_Efficient_3D_Semantic_Segmentation_with_Superpoint_Transformer_ICCV_2023_paper.html)；official proceedings abstract |
| `sonata2025` | Sonata: Self-Supervised Learning of Reliable Point Representations | CVPR 2025 | [来源](https://openaccess.thecvf.com/content/CVPR2025/html/Wu_Sonata_Self-Supervised_Learning_of_Reliable_Point_Representations_CVPR_2025_paper.html)；official proceedings abstract |
| `flash3d2025` | Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality | CVPR 2025 | [来源](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Flash3D_Super-scaling_Point_Transformers_through_Joint_Hardware-Geometry_Locality_CVPR_2025_paper.html)；official proceedings abstract |
| `kpconvx2024` | KPConvX: Modernizing Kernel Point Convolution with Kernel Attention | CVPR 2024 | [来源](https://openaccess.thecvf.com/content/CVPR2024/html/Thomas_KPConvX_Modernizing_Kernel_Point_Convolution_with_Kernel_Attention_CVPR_2024_paper.html)；official proceedings abstract |
| `ptv32024` | Point Transformer V3: Simpler Faster Stronger | CVPR 2024 | [来源](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_Point_Transformer_V3_Simpler_Faster_Stronger_CVPR_2024_paper.html)；official proceedings abstract |
| `geomae2023` | GeoMAE: Masked Geometric Target Prediction for Self-Supervised Point Cloud Pre-Training | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Tian_GeoMAE_Masked_Geometric_Target_Prediction_for_Self-Supervised_Point_Cloud_Pre-Training_CVPR_2023_paper.html)；official proceedings abstract |
| `roitr2023` | Rotation-Invariant Transformer for Point Cloud Matching | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_Rotation-Invariant_Transformer_for_Point_Cloud_Matching_CVPR_2023_paper.html)；official proceedings abstract |
| `peal2023` | PEAL: Prior-Embedded Explicit Attention Learning for Low-Overlap Point Cloud Registration | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_PEAL_Prior-Embedded_Explicit_Attention_Learning_for_Low-Overlap_Point_Cloud_Registration_CVPR_2023_paper.html)；official proceedings abstract |
| `pointmeta2023` | Meta Architecture for Point Cloud Analysis | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Lin_Meta_Architecture_for_Point_Cloud_Analysis_CVPR_2023_paper.html)；official proceedings abstract |
| `pointvector2023` | PointVector: A Vector Representation in Point Cloud Analysis | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Deng_PointVector_A_Vector_Representation_in_Point_Cloud_Analysis_CVPR_2023_paper.html)；official proceedings abstract |
| `pimae2023` | PiMAE: Point Cloud and Image Interactive Masked Autoencoders for 3D Object Detection | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_PiMAE_Point_Cloud_and_Image_Interactive_Masked_Autoencoders_for_3D_CVPR_2023_paper.html)；official proceedings abstract |
| `apes2023` | Attention-Based Point Cloud Edge Sampling | CVPR 2023 | [来源](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Attention-Based_Point_Cloud_Edge_Sampling_CVPR_2023_paper.html)；official proceedings abstract |
| `litept2026` | LitePT: Lighter Yet Stronger Point Transformer | CVPR 2026 | [来源](https://cvpr.thecvf.com/virtual/2026/poster/37647)；official abstract and actual full-text Introduction (Section 1) |
| `mspt2026` | MSPT: Efficient Large-Scale Physical Modeling via Parallelized Multi-Scale Attention | CVPR 2026 | [来源](https://cvpr.thecvf.com/virtual/2026/poster/38086)；official abstract and actual full-text Introduction (Section 1) |
| `sup2026` | SuP: Sub-cloud Driven Point Cloud Registration | CVPR 2026 | [来源](https://cvpr.thecvf.com/virtual/2026/poster/38606)；official abstract only |
| `hgi2p2026` | Hg-I2P: Bridging Modalities for Generalizable Image-to-Point-Cloud Registration via Heterogeneous Graphs | CVPR 2026 | [来源](https://cvpr.thecvf.com/virtual/2026/poster/39176)；official abstract only |
| `utonia2026` | Utonia: Toward One Encoder for All Point Clouds | ICML 2026 | [来源](https://icml.cc/virtual/2026/poster/62613)；official abstract and actual full-text Introduction (Section 1) |
| `fusioncell2026` | FusionCell: Cross-Attentive Fusion of Layout Geometry and Netlist Topology for Standard-Cell Performance Prediction | ICML 2026 | [来源](https://icml.cc/virtual/2026/poster/61564)；official abstract and actual full-text Introduction (Section 1) |
| `mogen2026` | MoGen: Detailed Neuronal Morphology Generation via Point Cloud Flow Matching | ICLR 2026 | [来源](https://iclr.cc/virtual/2026/poster/10010362)；official abstract only |
| `urope2026` | URoPE: Universal Relative Position Embedding across Geometric Spaces | ECCV 2026 | [来源](https://eccv.ecva.net/virtual/2026/spotlight/6035)；official abstract only |
| `vesseltok2026` | VesselTok: Tokenizing Vessel-like 3D Biomedical Graph Representations for Reconstruction and Generation | ECCV 2026 | [来源](https://eccv.ecva.net/virtual/2026/spotlight/6052)；official abstract only |
| `leap2026` | LEAP: Local ECT-Based Learnable Positional Encodings for Graphs | ICLR 2026 | [来源](https://iclr.cc/virtual/2026/poster/10011190)；official abstract only |

## 证据与复现

- `selected_papers.json`：正文43篇，完整原始作者及来源。`neuron_papers.json`、`geometry_papers.json`、`awarded_papers.json`保留全部检索结果。
- `neuron_synthesis.md`：直接任务知识、六项最近神经元方法对照。
- `writing_exemplars.md`：官方荣誉及实际引言阅读逻辑。
- `source_check.json`：正文引用、年份、重复、同步hash与凭据扫描。
- `layout_check.json`：实际编译页数及各节页码；`sync_status.json`：Overleaf提交、远端一致性及PDFhash。
- 正文数字仍来自2026-10-10冻结审计：开发验证、seed42、任务内匹配预算；非新训练、非独立测试集结论。TED微小交互增益、训练配置反例与H01非加性结论保留在当前写作状态说明。

本报告路径相对工作区 `artifacts/CPSF_FRONTMATTER_RESEARCH_20261011/`。Overleaf中同步的本报告位于 `notes/frontmatter_research_zh.md`，其原始证据保留在本地工作区。
