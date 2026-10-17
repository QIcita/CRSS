# R9 冻结代码与结果（2026-10-17）

当前保留配置为 **R9 patch_rms_010**。它在旧固定残差共同区域主干上加入真实官方Utonia缩小配置，先保留patch局部统计，再加入RMS0.1上下文。源码中的GeoTransformer与Polynormer分支未在此配置启用。

- `reference/`：服务器R9冻结文件的逐字节副本。`tools/cpsf_utonia_context/model.py`为模型入口，`data.py`是物理体素输入，`kernel_policy.py`固定spconv计算核选择。保留全部被协议记录的程序源文件；不把历史依赖名当作正在使用的独立模型。
- `reference/results/`：六组H01原始结果、历史和预测；只附保留RMS0.1的两个`final.pt`。重跑及0.3控制不重复发布权重。
- `extension/`：FlyWire/TED/TLP运行A2的新增适配包、冻结协议、验收与带时间的运行快照。它复用reference中的共享源码，不是另一套主干。没有发布运行中的latest.pt。
- `RESULTS.json`：五任务状态；未完成任务的`value`为null。
- `server_export.json`：服务器来源与逐文件哈希，以及明确省略的非必要上游数据/示意图与通用Python依赖源码。原协议保持字节不变，仍可能引用未打包的数据、父版本和上游示例文件。

阅读顺序：[实际架构](../../docs/R9_ARCHITECTURE_ZH.md) → [结果与边界](../../docs/R9_RESULTS_ZH.md) → [模型源码](reference/tools/cpsf_utonia_context/model.py) → [任务扩展](extension/tools/cpsf_r9_expansion/model.py)。

## 可执行检查

在仓库根目录运行（仅Python标准库）：

```bash
python scripts/verify_r9.py
python scripts/verify_export.py
```

两项保留checkpoint可由原冻结模型严格加载；它们是本项目训练得到的权重，不是外部预训练权重。H01模型构造接口为`tools.cpsf_utonia_context.model.Model(task, 'patch_rms_010')`，三任务适配入口为`tools.cpsf_r9_expansion.model.Model(task)`。

## 环境与训练入口

实际验证环境为Linux、RTX4090D/SM89、PyTorch2.4.0+cu121；numpy1.26.4、scipy1.13.1、scikit-learn1.5.2、spconv-cu120 2.3.6、cumm-cu120 0.4.11、torch_scatter2.1.2+pt24cu121、flash_attn2.7.4.post1、einops0.8.1、addict2.4.0。Utonia导入还需要IPython（服务器额外runtime路径为8.37.0）。完整安装版本见server_export.json；本包不提供新机器的一键编译脚本。

原服务器的冻结入口：

```bash
# R9 H01，仅说明已完成实验入口，不要重复启动
python -m tools.cpsf_utonia_context.train --task h01_local --kind patch_rms_010
# A2三任务，需在其完整冻结运行目录及既有环境中执行
python -m tools.cpsf_r9_expansion.train --task flywire_coarse
python -m tools.cpsf_r9_expansion.train --task ted
python -m tools.cpsf_r9_expansion.train --task tlp
```

这些脚本依赖原数据、manifest、侧文件、父版本哈希及验收门。GitHub不包含原始数据、几何缓存或全部父运行目录；不要关闭verify或伪造contracts来绕过依赖。R9与A2均以文件所在目录定位运行根；不能直接把两个根的protocol.json互换。新环境的全量训练复现尚未验收。

## 来源与许可

Utonia固定commit `da776a0bd3a48c6df83ac2ae0e27b26141cc7e31`，官方源码与许可证在`reference/vendor/Utonia/`；GeoTransformer与Polynormer作为冻结历史源依赖保留各自许可证，但未启用。所有第三方许可证保持原样，本次没有为整个研究代码额外授予统一开源许可。

原`code/model_full.py`及其五任务权重仍属于2026-10-11旧版本，不能作为R9模型或R9五任务成绩引用。
