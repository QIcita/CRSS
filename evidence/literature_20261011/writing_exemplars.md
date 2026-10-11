# 2026高质量论文核验与引言写作借鉴

核验日：2026-10-11。已加载 academic-search 和 oral-paper-skill，并读取后者 abstract-derived-practices 与 reading-level 来源。academic-search 安装中只有SKILL.md及README等，所指向的scripts/check-deps.sh、metadata-schema.md、CS profile、arXiv站点文件在两个标准路径均缺失；本次使用requests/curl可用的官方网页，输出透明JSON字段，不以缺失附件阻断已授权工作。未下载批量PDF。

## 会议范围与重要更新

- 官方页面明确标注的荣誉共8篇：**CVPR2026 Highlight**：LitePT、MSPT、SuP；**ICLR2026 Oral**：GraphGlue、RULE；**ECCV2026 Spotlight**：URoPE、VesselTok；**NeurIPS2026 Oral**：Graph Cascades。
- 其余7篇是相关性优先的普通Poster，不将它们升级为Highlight/Spotlight。CVPR的Highlight文章仍使用poster URL，标签证据来自`.award-name`，而非URL。
- **此前2026-09-29“NeurIPS2026荣誉尚不可核验”的笔记现在已过时**。当前官方[2026 Orals](https://neurips.cc/virtual/2026/events/oral)及[Graph Cascades](https://neurips.cc/virtual/2026/oral/155098)可访问。只描述当前已核验页面，不根据日历推断是否存在论文。
- [ICCV官网](https://iccv.thecvf.com/)自述biennial，页面当前为ICCV2027且年份菜单列2027/2025；2026应检索ECCV，不杜撰ICCV2026。
- [ECCV2026 Spotlights](https://eccv.ecva.net/virtual/2026/events/spotlight)已公开。VesselTok是本轮新增、与管状生物结构及link prediction高度相关的工作。
- 15篇完整字段及原始摘要见`awarded_papers.json`，可用BibTeX见`awarded_papers.bib`。保存了官方HTML及提取文本；未核验DOI/页码的字段为空，不能补猜。

## 实际阅读级别

本次重新获取并完整读取以下4篇作者全文的**§1 Introduction**，而非仅用会议摘要推断段落逻辑。没有声称重新审计其所有实验或代码。

### LitePT（CVPR2026 Highlight）

官方：https://cvpr.thecvf.com/virtual/2026/poster/37647
正文：https://arxiv.org/html/2512.13689v2#S1
本地原文：`official_sources/litept_full_intro.txt`

**原文组织**：
1. 应用重要性→已有多种算子，但缺少各自作用及如何组合的原则。
2. 用一个具体结构事实收窄问题：PTv3有67%参数用于卷积，混合结构使“卷积与注意力分别做什么”成为可回答的问题。
3. 给出作者分析得到的分工：高分辨率阶段局部几何适合卷积，低分辨率阶段长程语义适合注意力。
4. 指出移除卷积后位置功能仍需承担→PointROPE。
5. 原则→LitePT架构→参数/内存/速度及准确率证据。

**可迁移的写法**：不要从“我们用了采样、图、注意力”进入贡献；先界定不同计算阶段各需要什么证据，再说我们改变输入预算分配与融合位置。**边界**：LitePT有分析支撑其分工，不能将我们尚未证明的“最优分工”照写；其算子效率结论也不能替代我们的采样消融。

### MSPT（CVPR2026 Highlight）

官方：https://cvpr.thecvf.com/virtual/2026/poster/38086
正文：https://arxiv.org/html/2512.01738v2#S1
本地原文：`official_sources/mspt_full_intro.txt`

**原文组织**：
1. 工业物理域规模增大带来准确率与可扩展性需求。
2. 通过不同物理例子把需求拆成局部场变化与长程耦合，并指出全对交互的二次代价。
3. 比较相关解法的具体取舍：spectral、global slice/latent、local partition各解决一部分，保真与传播仍需兼顾。
4. 以parallel local/coarse attention给出计算对象及信息通路；贡献区分PMSA机制、MSPT架构、实验。

**可迁移的写法**：先用神经元形态任务说明“局部突变”和“跨分支整体组织”为何都必要，再区分输入覆盖与跨模态关联两层。**边界**：我们的编码不是PDE算子学习；没有其near-linear scaling与网格无关保证。不能把“有限测度聚合”写成已经验证的神经算子理论。

### Utonia（ICML2026 Poster，未核验Spotlight）

官方：https://icml.cc/virtual/2026/poster/62613
正文：https://arxiv.org/html/2603.03283v2#S1
本地原文：`official_sources/utonia_full_intro.txt`

**原文组织**：
1. 点云作为稀疏几何表示的价值。
2. 与规则图像比较，列出尺度、密度、采样模式、辅助通道的域间差异。
3. 具体后果：模型可能按域先验聚类而非可迁移几何；用toy car/real car相似度说明。
4. 既有backbone和SSL成熟后，“简单混数据为何仍失效”成为下一步问题。
5. 三类不匹配逐一对应minimal fixes，同时以“first step/exploration”控制范围。

**可迁移的写法**：当前引言应明确区分物理范围、采样点数、密度和区域长度，避免“任意尺度”包办所有含义。**边界**：我们五任务分别训练，同一个架构不等于Utonia式共享预训练权重；不承诺跨域统一表征空间。

### FusionCell（ICML2026 Poster，未核验Spotlight）

官方：https://icml.cc/virtual/2026/poster/61564
正文：https://arxiv.org/html/2605.20287#S1
本地原文：`official_sources/fusioncell_full_intro.txt`

**原文组织**：
1. 任务重要性与现有精确仿真成本。
2. 说明预测由布局几何和网表共同决定；删任一路各丢什么信息。
3. 相关工作按缺失信息与结合层次比较，指出电气拓扑需要解释布局效应。
4. 用“schematic map / satellite image”直观描述结构引导几何的原则。
5. 明确交互操作：graph query、image key cross-attention，并区分输入编码、融合、资源。

**可迁移的写法**：我们的骨架提供观测连接路径，表面提供横截面/局部形状；重点是这些证据何时、在哪个区域形成联合特征。**边界**：神经元骨架并不拥有网表的电气因果权威。原文中的防止modality collapse/vision shortcut也不是我们已完成验证的性质。

## 最多三个优先修改（对应当前稿件）

| 位置 | 来源实践 | 建议改写 | 适用原因与边界 |
|---|---|---|---|
| Introduction第2–3段：核心问题 | Utonia §1把差异拆成可检验的不匹配；MSPT §1解释局部/长程后果 | 将总问题写为“有限采样预算下，如何保留跨范围形态证据，并在相同物理区域关联异质支持”；随后只保留“证据保留”和“证据关联”两层问题。明确前一层是后一层的必要输入条件。 | 对应实际动态采样/共同区域；不是单点数、尺度不变性或跨域预训练的泛化主张。 |
| Introduction方法承接与贡献列表 | LitePT §1原则先于模块；FusionCell §1点明交互操作 | 用有限加权几何集合与区域条件聚合作为统一表述，再命名两个机制：带权多分辨率输入编码、共同区域联合编码。写清确定性预处理和学习模块分别发生在哪里。 | 数学表述解释设计，不作为新定理。避免claim首次使用位置编码或图—几何融合；不把残差系数列独立创新。 |
| Related Work末段及Preliminaries | MSPT按取舍比较相关方法；oral-skill要求条件紧随结论 | Related Work按神经元形态学习、分辨率/几何编码、区域跨模态聚合三组；Preliminaries只定义观测G与P、采样权重、区域软关联和归一化算子。模型具体MLP、层数、tile公式进入Method。 | 3页预算需要功能分工，不应三节重复两模块故事。强调已有共坐标，非未知姿态registration；五任务独立监督，非共用一个预训练checkpoint。 |

## 可直接用于Related Work的事实片段（非最终段落）

- GraPHFormer将骨架与同源拓扑图像结合，MoGen关注神经元局部点云生成，VesselTok从centerline与pseudo radius构造管状结构latent：三者分别约束“多模态”“细粒度表面”“管状结构tokens”的泛泛新颖性。应比较实际输入支持及融合位置。
- LitePT/MSPT/Utonia说明高低分辨率处理、局部/全局交互与点云粒度已经被研究；我们需强调这些选择如何服务同坐标骨架—表面联合表示。
- Hg-I2P/SuP是需要寻找跨对象对应的配准；FusionCell是结构驱动几何检索。我们保留观测拓扑，以已有共坐标建立几何软关联，再学习区域内容交互。
- Graph Cascades的摘要明确给出低度规则图及结构瓶颈下的失败条件；引用时不能称图重连普遍有益。URoPE依赖相机射线/深度锚点，不应写成通用神经元任意尺度编码。

本报告所有借鉴都是作者写作组织层面的有界推论，不是Oral/Highlight产生原因的统计结论，也不替代本项目实验。
