# 最新代码发布：R38（2026-10-25）

[架构与公式](docs/R38_ARCHITECTURE_ZH.md) → [结果与状态](docs/R38_RESULTS_ZH.md) → [代码、五份完成权重和组装](code/r38/README.md)。

包含原 R38、密集邻域修复、MICrONS 区域计数适配及 DRW 训练分支。R28 仍是总体研究主参照。FlyWire 细分类已暂停，DRW 导出时仍训练；不得把两者的验收或中间状态写成最终结果。Macro-F1 是 FlyWire 默认主指标。

```bash
python scripts/verify_r38.py
python scripts/materialize_r38.py /absolute/path/to/new/r38 --variant drw
```

H01/FlyWire 采用固定最终轮开发集评估，MICrONS 采用验证集选模后最终测试；各任务原始协议保留。历史模型、权重、报告和论文继续保留，论文尚未按 R38 改写。以下为历史导航，以各节日期与版本为准。

---

# 最新发布：R34 密集邻域＋细支撑保留（2026-10-23）

[结构与结果](docs/R34_DENSE_SUPPORT_ZH.md) → [源码和最终权重](code/r34_dense/README.md) → [统一 FPS 全部完成实验](docs/UNIFIED_FPS_RESULTS_20261023.md)。

本版在统一 FPS 输入上完成 H01 整体重训，MAE 0.050024；其他任务的本版正式重训尚未完成。它扩大 SPoTr 计算邻域，保留原 0.5µm 细支撑、Utonia 输入、骨架关联和尺度编码。旧 R34 发布与旧输入结果继续保留，不能混为当前成绩。R38＋密集邻域是另外的候选运行。

```bash
python scripts/verify_r34_dense.py
python scripts/materialize_r34_dense.py /absolute/path/to/new/r34_dense
```

---

以下为历史导航，具体边界以各自日期/版本为准。

# 最新发布：R34阶段一（2026-10-21）

阅读顺序：[R34实际架构](docs/R34_ARCHITECTURE_ZH.md) → [H01结果与机制对照](docs/R34_RESULTS_ZH.md) → [代码、权重和组装](code/r34/README.md)。新增入口为`code/r34/overlay/tools/cpsf_path_message/model.py`；继承依赖由`source_map.json`逐字节映射至R28。`scripts/materialize_r34.py`可组装完整运行源码。

R34只实现路径条件差分消息，在原输入的H01双任务通过预定门槛。新FPS输入、其他任务和阶段二/三均未验证；不要把提案或其他版本结果写入当前模型。下面R28及更早导航为带日期的历史记录。论文正文仍未按R34改写。

---

# 当前参照：R28（2026-10-20）

先读 [R28整体架构与逐模块论文来源](docs/R28_ARCHITECTURE_ZH.md) → [R28五任务与baseline对照](docs/R28_RESULTS_ZH.md) → [R28冻结代码、三项权重和复现边界](code/r28/README.md)。机器可读状态是`code/r28/RESULTS.json`；未完成任务值为null。

主入口`code/r28/reference/tools/cpsf_raw_point_attention/model.py`，三任务适配`code/r28/extension/tools/cpsf_r28_expansion/model.py`，共享主干`reference/tools/cpsf_utonia_context/model.py`。H01主配置名r9_lpa_attention只是继承命名，不是原R9；扩展名r28_lpa_attention与其机制一致。不要混入R29–R33分支。

`python scripts/verify_r28.py`核对原源、协议、权重、结果和发布清单。H01两项与TLP已完成；FlyWire/TED的状态是带时间快照。当前论文尚未按R28重写。下方所有R9与更早文字均为历史导航，不覆盖本节。

---

# 当前入口：R9（2026-10-17）

先阅读 [R9架构](docs/R9_ARCHITECTURE_ZH.md)、[R9结果](docs/R9_RESULTS_ZH.md)和[R9代码包](code/r9/README.md)。共享主干入口为`code/r9/reference/tools/cpsf_utonia_context/model.py`，扩展入口为`code/r9/extension/tools/cpsf_r9_expansion/model.py`。

H01两项已有固定最终轮开发成绩；FlyWire/TED/TLP仍训练。不要把下方旧五任务结果、旧单文件模型或论文草稿改称R9。R9仅有已发布的幅度对照与重放证据，旧版消融不能证明R9核心模块贡献。

以下为2026-10-11历史研究快照的导航，继续保留供追溯。

---

# 从这里理解当前研究

这个仓库将“做了什么”和“准备如何写论文”放在一起。请先理解当前定义，再读历史来源文件；文件名包含 V21/V23/V26 不代表正在同时主张多套方法。

## 建议阅读顺序

1. [论文主线](docs/PAPER_STORY_ZH.md)：任务价值、两个递进问题、数学表述、两个机制和证据链。
2. [代码导航](docs/CODE_MAP_ZH.md)：定位到真实计算；随后阅读 [model_full.py](code/model_full.py) 的 `build_model`、对应任务类、`stems`、`embedding`、processor 和读出。
3. [原包说明](code/README.md)、[DEFAULTS.json](code/DEFAULTS.json)：输入字段、任务身份、环境和重训依赖。
4. [实验索引](docs/EXPERIMENTS_ZH.md)：冻结主结果与后续官方基线复评分别阅读，不把不同时间、配置、版本的结果混在一起。
5. [英文引言](paper/sec/1_intro.tex)、[Related Work](paper/sec/2_related_work.tex)、[Preliminaries](paper/sec/3_preliminaries.tex)、[Method](paper/sec/4_method.tex)，以及 [中英段落对照](paper/notes/paragraphs_zh.md)。
6. [领域知识](paper/notes/neuron_literature_zh.md)、[写作范例](paper/notes/writing_exemplars_2026_zh.md)、[文献依据](paper/notes/frontmatter_research_zh.md)。

## 遇到表述冲突时

实际实现以 `code/model_full.py`、对应 `code/snapshots/` 和配置为准；成绩以原始指标及冻结审计为准；最新论文意图以 `paper/notes/writing_status.md` 和本仓库论文主线为准。`docs/source_reports/` 是带日期的历史分析，不能直接当作当前论文全部结论。

尤其注意：历史报告曾讨论将区域解释为超边，当前论文不据此声称实现了新的超图网络；历史“分辨率编码”概括不等于当前尺度 MLP 动态接收每块分辨率，具体尺度输入见 Method 和代码。10月11日 BN 复评更新了外部基线比较认识，不改写10月10日的主方法冻结消融。

## 给阅读模型的任务边界

请引用具体仓库路径、类/函数、配置或指标，区分已经实现、已经验证和仍需补充。不要根据期望的消融规律制造结果，不把五次分任务训练说成共享权重迁移。完整模型 API 只暴露已核验 full；历史分支不自动构成通用有效消融 API。

GitHub 仓库权限和 ChatGPT 中的仓库读取授权需由对应账号提供。创建 GitHub 仓库本身不代表它已经被某个网页会话读取；可以用“列出 `build_model` 支持的任务并引用源码”确认该会话的实际访问情况。
