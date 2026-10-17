> **版本提示（2026-10-17）：本页描述历史版本。当前R9请先读 [架构](R9_ARCHITECTURE_ZH.md)、[结果](R9_RESULTS_ZH.md)和[代码](../code/r9/README.md)，不要混用模型与成绩。论文草稿尚未按R9更新。**

# 供网页版 GPT 使用的阅读提示词

在该会话能够访问本仓库后，可以直接发送：

> 请阅读 QIcita/CRSS 仓库。先读 START_HERE.md、docs/PAPER_STORY_ZH.md、docs/CODE_MAP_ZH.md 和 docs/EXPERIMENTS_ZH.md，再读 code/model_full.py、code/DEFAULTS.json 以及 paper/sec/1_intro.tex、2_related_work.tex、3_preliminaries.tex、4_method.tex。请先通过列出 build_model 支持的五个任务及对应类名确认实际读取了代码；如果文件不可访问，请明确说明，不要根据仓库名称推测内容。
>
> 我正在撰写神经元骨架—表面联合表示论文，当前方案为CRSS固定残差共同区域框架。请按“任务价值—几何证据保留—几何证据关联—带权共同支持表述—两个机制—验证证据”理解。目标不是组合更多模块，而是让具体问题、计算改变和实验依据形成连贯体系。
>
> 请输出：①当前架构与真实数据流；②两项核心机制各自解决的具体问题及对应源码；③主张—实现—消融证据的对应表；④Introduction、Related Work、Preliminaries中存在的逻辑或措辞问题；⑤最优先需要补充的证据。请引用仓库文件路径和类/函数，区分已实现、已验证、待验证。不要混入V17超图或V26门控full；不要把五个分别训练的模型说成同一套权重迁移，也不要把开发验证结果说成独立测试结果。
>
> 最新基线复评见docs/source_reports中的20261011审计：FlyWire的PointMamba略高于当前full，不可声称全面领先。当前没有新训练目标、尺度不变性定理或重采样不变性定理。请在这些事实约束内帮助我完善论文，保持问题和机制聚焦。

第一轮建议只做代码与主张核对；确认读懂后，再逐段改写。需要查看中英文时，使用 `paper/notes/paragraphs_zh.md`；不要默认历史研究报告中的每条拟议机制已经实现。
