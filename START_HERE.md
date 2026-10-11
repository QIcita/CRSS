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
