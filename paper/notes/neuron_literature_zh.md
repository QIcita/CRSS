# 与神经元具体任务相关的正式文献：问题知识与写作边界

检索日期：2026-10-11。范围：2023–2026正式会议与期刊；已核验20篇（19篇期刊及1篇CVPR）。不以preprint冒充会议论文，不将Poster自动标成Highlight。原始Crossref/EuropePMC及官方会议页存于 `neuron_sources/`；结构化元数据和逐篇可支持论断见 `neuron_papers.json`，可引用条目见 `neuron_papers.bib`。

academic-search已读取并执行API优先、先筛后查。安装目录缺失该技能引用的 `references/metadata-schema.md`、`references/disciplines/biomedicine.md`、check-deps脚本，因此没有伪称执行；采用字段完整的JSON并记录阅读层级。未批量下载PDF；NEURD与MorphRep通过EuropePMC全文XML读取Introduction及相关方法。其他主要使用正式元数据＋PubMed/EuropePMC摘要，不依据摘要缺项断言全文没有某机制。所有论文有正式来源支持2023–2026出版身份；MACGNN DOI含2022但正式卷期为2024，DSM DOI含2023但卷期为2024；CAVE DOI含2024但正式发表2025。

## 1. 可以直接服务当前五任务的知识

**重建校对不等于只有局部图像分割。** NEURD（Nature2025）全文“Main”“Graph decomposition”及Fig.2–3表明，mesh已被分解为非分叉段图，并包含node/subgraph/graph级特征、骨架/mesh关联及按领域规则校对。其校对明确主要指merge error removal，不包含补全false split。Chen等TMI2026则针对split：全局3D morphology＋高分辨率局部EM图像，先连接点proposal，再multimodal cross-attention与connectivity prediction。因此，不应笼统写“前人仅局部图像、缺乏全局形态”；应定位当前任务所用**观测骨架与三维表面共同输入的区域编码**，区别于启发式分解工具或形态—体图像管线。相应引用：`celii2025neurd,chen2026segmentconnectivity,zhu2023morphognn,gou2024gapr`。

**局部区室与整体细胞类型是不同的预测粒度，而非人为凑齐任务。** NEURD展示从spine/bouton等细部到branch和cell的形态测量；Schneider-Mizell等Nature2025以树突区室靶向与完整细胞输入组织类型；H01表明高分辨率mesh能支持细胞形态及皮层组织研究。可支撑“同一细胞表示需保留局部证据并支持整体分析”，但不证明我们的任务标签绝对真实。相应引用：`celii2025neurd,schneider2025inhibitory,shapsoncoe2024h01`。

**形态有分类价值，但cell type不是纯几何定义。** FlyWire annotation论文（Nature2024）建立hierarchical classes、types、hemilineages，并通过跨connectome一致性讨论细胞类型；Nern等Nature2025的视觉系统inventory联合形态、连接、递质和专家标注。故我们的FlyWire分类是“根据形态预测已发布细胞标签”，不能说监督标签仅由形态独立产生。也不能把hemibrain加入我们的训练叙述。相应引用：`schlegel2024flytyping,dorkenwald2024flywire,nern2025visualinventory`。

**形态表示的最新相关先例已经充分利用拓扑、几何与位置。** MACGNN（TNNLS2024）研究classification/retrieval；MorphoGNN（TMI2023）包含点级表示、重建质量和与其他模态组合；MorphRep（Bioinformatics2024）用超过250k重建做图Transformer预训练；SGTMorph（TMI2025）明确local topology＋global transformer＋random-walk位置；NeuroGT（MedIA2026）以电缆方程EPE与k-SPE编码物理/层次先验；GraPHFormer（CVPR2026）结合骨架和同源persistence images。不能制造“已有方法忽略局部/全局、位置或多模态”的问题。

## 2. 与当前稿件最近的边界表

| 最近工作 | 输入与明确已做内容 | 当前可以论证的区别 | 不应声称 |
|---|---|---|---|
| NEURD Nature2025 | mesh和synapse，skeleton/mesh分解映射、多层特征、启发式校对和类型分析 | 不以手工特征为最终表示，而在共同区域上学习双分支特征并传播 | 首次mesh与skeleton对应、首次区域分析 |
| Chen et al. TMI2026 | 全局3D形态＋局部高分辨率EM图像；proposal-based跨尺度采样和cross-attention | 直接使用已共坐标的骨架与表面，不需要体图像编码，区域支持内分别归一化 | 首次多尺度多模态proofreading |
| GraPHFormer CVPR2026 | skeleton＋由同树导出的persistence image；官方摘要明确late fusion与cross-attention | 第二路是实际表面几何，关联是原生三维局部支持 | 该文仅late fusion或不含局部交互 |
| SGTMorph TMI2025 / NeuroGT MedIA2026 | 形态图的局部-全局编码、结构/物理位置 | 两种异质几何支持共同区域聚合，显式覆盖预算组织 | 首次位置/尺度/局部-全局图表示 |
| NeuNet Neural Networks2025 | 单细胞skeleton＋细胞间connectome graph | 单细胞内部skeleton-surface对应 | 所有“topology＋geometry”均是同一输入设定 |

## 3. 从原文学习叙述逻辑（不借内容）

1. **NEURD，Nature2025，Main前四段＋Summary of large-scale reconstructions。** 原文先给领域资源的发展与机会，再指出资源可用性依赖分析工具，明确mesh为不同成像管线共有的轻量数据产品，然后定义分解图及其使能的具体活动。借鉴：我们Intro第一段应从重建后几何产物及局部/整体分析需求切入，而非先报模型名。限制：NEURD是软件资源/分析工作，不能把它的高层multi-task utility当作我们的模型竞争结果。
2. **MorphRep，Bioinformatics2024，Introduction§1及Methods§2.1–2.3。** 先解释形态表型价值，再列noise、3D complexity和数据规模三个约束，随后逐类讨论表示策略，最后逐项对齐large-scale pretraining、augmentation、adapter。借鉴：问题应与已实现机制一一对应，避免两个机制共同声称解决全部问题。限制：其问题之一是标注/规模，而我们当前不是预训练创新，故不能照抄它的数据稀缺主线；其Introduction对其他方法规模的概括也不应未经独立核验照搬。
3. **NeuroGT，MedIA2026，正式收录摘要（全文未读）。** 从computationally efficient/biologically faithful张力转到缺乏biophysical bias与local-global objectives，然后点名cable equation → EPE、hierarchy → k-SPE、global contrast/local denoising。借鉴：我们应明确加权经验几何表示如何导出采样权重、区域归一化，并将此称为数学建模原则。限制：电缆方程有明确物理含义；不能仅用“measure”一词把我们包装成新的物理理论或精确不变性。

## 4. 可直接采用的相关工作段落（英文素材，待主代理压缩）

**Neuronal morphology and reconstruction.** Recent connectomes provide detailed reconstructed geometries and annotations across human cortex and the adult fly brain, while functional connectomics links such reconstructions to neuronal activity. These resources motivate analysis at several levels: correcting reconstruction errors, characterizing compartments, and predicting cell identity. NEURD organizes neuronal meshes into annotated branch graphs for proofreading and morphology analysis, whereas multimodal segment-connection prediction combines global three-dimensional morphology with local EM appearance. These approaches establish that topology and geometry already provide complementary evidence; our focus is a learned encoder for co-registered skeleton and surface inputs, with correspondence represented before regional context propagation.

**Learned neuronal representations.** Morphology-aware graph learning, point-level morphological embeddings, and large-scale graph pretraining support neuronal classification and retrieval. Recent graph Transformers explicitly combine local topology with global relations or inject biophysical and hierarchical position encodings. GraPHFormer further combines skeletons with persistence-image representations, exploring both late fusion and cross-attention. We therefore do not identify a lack of multimodality or local–global modeling as the gap. Instead, we address how an observed skeletal graph and a sampled three-dimensional surface, which have different discretization supports, can form common local units under a finite sampling budget.

## 5. 引文使用建议与排除

正文优先引用与任务直接相关的前16–18篇；幼虫全脑和雄蝇视叶inventory可在篇幅紧张时删减，不能仅为40篇门槛堆砌。所有20篇均是2023–2026正式论文；已有GraphDINO旧笔记和Point Affinity Transformer仅preprint身份未在本轮取得新正式证明，故未计入20篇。NREDD关键词EuropePMC未找到，不据此判断数据不存在，应该沿项目数据源另查其原始出版物。

MICrONS、幼虫和雄蝇视叶仅作为领域文献背景，不改变当前NREDD/H01/FlyWire三数据集协议。论文原始资源规模与当前训练子集规模必须分开。所有领域机制描述均不等于我们已经完成强cross-attention/late-fusion对照或未见测试评估。

## 6. 对照主稿应进一步收紧的六项区别

| 工作 | 核心表示与任务（阅读证据范围） | 相对当前实际实现的区别 | 写作禁区 |
|---|---|---|---|
| MACGNN, TNNLS2024 | 形态感知图对比学习，classification/retrieval；摘要 | 目标在形态图的无监督与哈希检索；我们在共坐标观测骨架和真实表面上作分任务监督 | 不说其不利用几何或只全局池化 |
| MorphoGNN, TMI2023 | 邻域内点级图特征，多任务，也可合并显微图像/手工特征；摘要 | 当前显式构造两种支持的共同区域，按骨架长度/表面采样权重分别归一化 | 不说其没有局部表示、不能多模态或不做质量分析 |
| SGTMorph, TMI2025 | 图局部建模＋Transformer全局关系，RW位置与空间不变编码；摘要 | 当前刻意保留区域长度/位置并联合表面支持；不主张编码空间不变性 | 不说前人没有local-global或位置 |
| NeuroGT, MedIA2026 | 电缆方程EPE＋k-SPE，图对比＋坐标去噪；摘要 | 当前scale features是几何标识，不是电生理模型；两模态region learning不是预训练objective贡献 | 不包装成新的生物物理理论 |
| GraPHFormer, CVPR2026 | 同源skeleton与persistence images；late fusion和cross-attention；正式摘要及既有作者全文 | 当前第二路来自实测三维表面，已配准几何有固定局部incidence，而非投影拓扑摘要 | 不写只有late fusion |
| Segment Connectivity Prediction, TMI2026 | 3D形态＋局部EM图像；proposal sampling改善跨尺度cross-attention；摘要 | 当前不需要体图像分支，也不以生成connection proposal为核心；学习区域编码后接point/pair/whole读出 | 不宣称首次跨尺度神经元多模态连接预测 |

新增CAJAL（Nature Communications2023）已经替换相对边缘的MoMo，最终保持20篇。它是用明确数学对象统一形态表征的适合先例：其理论基础为metric geometry；我们只可借鉴解释对象—操作—意义的逻辑，不能把它的Gromov–Wasserstein意义或保证当作我们的权重聚合理论。正式元数据和摘要均来自Crossref/EuropePMC。
