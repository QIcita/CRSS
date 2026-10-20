# R34阶段一：冻结代码、H01双任务权重与机制对照

版本：`CPSF_PATH_MESSAGE_20261020_R34_A2`。协议SHA256：`eaf723615ae22df382e9db542a9c4bed94ccafe947d0378c221273641c4931e3`。

R34在R28的共同区域模型上加入**骨架与路径条件的区域差分消息**。请先读[架构与实际公式](../../docs/R34_ARCHITECTURE_ZH.md)和[六组结果及边界](../../docs/R34_RESULTS_ZH.md)。

## 目录与实际入口

- [overlay/tools/cpsf_path_message/model.py](overlay/tools/cpsf_path_message/model.py)：本版本新增模型，包含完整候选和两个机制对照。
- [source_map.json](source_map.json)：461个运行源码/依赖文件的逐文件SHA256及仓库路径；450个与R28完全相同的源直接引用`../r28/reference/`，11个新增源保存在overlay。
- [protocol.json](protocol.json)、[contracts.json](contracts.json)：原始冻结协议与训练前验收，保持原字节和绝对来源路径。
- [RESULTS.json](RESULTS.json)、`results/`：六组H01末轮指标、identity、history、开发预测和首次更新记录。
- `results/path_message_h01_local/final.pt`、`results/path_message_h01_global/final.pt`：两项完整候选的最终权重。控制组权重仍在原服务器，仅发布指标、预测和权重哈希。
- [server_export.json](server_export.json)：服务器导出文件哈希；[evidence/publication_validation.json](evidence/publication_validation.json)：发布源码构造、六组严格加载和指标复算结果。

69个非运行上游示例数据/图片等文件沿用R28的明确省略清单。不是遗漏Python模块；来源、许可证及官方模型源码均保留。原始神经元几何、缓存、样本manifest及运行依赖目录未分发。

## 组装与加载

在仓库根运行：

```bash
python scripts/verify_r34.py
python scripts/materialize_r34.py /absolute/path/to/new/r34
```

目标目录必须不存在。脚本校验共享源码哈希并复制原字节，不修改模型，不启动训练，也不绕过原训练验收。

在组装目录及兼容依赖环境中：

```python
import torch
from tools.cpsf_path_message.model import Model

task = 'h01_local'  # or 'h01_global'
model = Model(task, 'path_message')
checkpoint = torch.load(
    f'results/path_message_{task}/final.pt',
    map_location='cpu', weights_only=False,
)
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
```

输入字典沿用R28真实patch/区域几何；[data.py](overlay/tools/cpsf_path_message/data.py)提供额外的完整关联批处理布局。完整候选不需要普通注意力布局计算，但冻结加载器为三个对照共同保留它。

环境沿用R28：Linux、PyTorch2.4.0+cu121、spconv-cu120 2.3.6、cumm-cu120 0.4.11、torch_scatter2.1.2+pt24cu121、FlashAttention2.7.4.post1、numpy1.26.4、scipy1.13.1、scikit-learn1.5.2及einops/addict。完整环境/来源见[R28包](../r28/README.md)。本发布核验模型构造和加载，不宣称新机器端到端训练已复现。

原服务器的训练命令为：

```bash
python -m tools.cpsf_path_message.train --task h01_local --kind path_message
python -m tools.cpsf_path_message.train --task h01_global --kind path_message
```

控制配置是`constant_message`和`ordinary_attention`。在只有本发布包时，上述训练命令仍需要原manifest、几何缓存、父版本证据和协议列出的依赖文件；不要删除verify或伪造contracts来绕过这些条件。

## 来源与结果身份

完整R34执行继承的官方Utonia和SPoTr模块；新增PathMessage由项目实现。GeoTransformer衍生注意力仅用于ordinary_attention对照。许可见对应共享vendor目录和[来源声明](../../RIGHTS_AND_PROVENANCE.md)，不新增统一开源许可。

只包含阶段一。新输入FPS方案、FlyWire/TED/TLP扩展、层级贡献接口、SPoTr内部差分值映射都未被加入本版本。两项H01采用原冻结输入，seed42、固定最终轮开发集；不能与新FPS结果混表或宣称六任务已验证。
