# R34 dense neighborhood / preserved fine support

不可变版本：`CPSF_R34_AP10_SUPPORT_PRESERVED_20261023_A1`。
协议 SHA256：`e5ac48d2060320c307fea451c1e10236bb7394efdfd613349d54ccc13a0e74cd`。

先读[结构、来源和验证边界](../../docs/R34_DENSE_SUPPORT_ZH.md)。该版本仅发布完成的 H01 整体结果：MAE **0.0500239972**，50 轮、1000 更新，seed42，统一 FPS 输入。

- [model.py](overlay/tools/cpsf_r34_patch_repair/model.py)：`SupportPreserved`，密集 SPoTr 增量回投原细支撑。
- [data.py](overlay/tools/cpsf_r34_patch_repair/data.py)：原支撑与自适应计算邻域的分离。
- [source_map.json](source_map.json)：513 个逐文件哈希映射，复用已发布的相同源码，新冻结源保存在 overlay。
- [protocol.json](protocol.json)、[variant.json](variant.json)、`parents/`：原字节冻结协议。
- `results/r34_h01_global/`：最终权重、开发预测、指标、身份、历史和首次更新记录。
- `acceptance/`、`evidence/`：训练前验收、局部兼容性检查和发布加载验证。
- [server_export.json](server_export.json)：服务器来源与原文件哈希。未发布的非运行上游示例沿用明确省略清单。

```bash
python scripts/verify_r34_dense.py
python scripts/materialize_r34_dense.py /absolute/path/to/new/r34_dense
# 使用兼容 Linux/PyTorch/CUDA 依赖环境；此命令不训练：
python scripts/validate_r34_dense_runtime.py /absolute/path/to/new/r34_dense --output /absolute/path/to/validation.json
```

也可以在组装目录中直接构造并加载：

```python
import torch
from tools.cpsf_r34_patch_repair.model import Model
model = Model('h01_global')
state = torch.load('results/r34_h01_global/final.pt', map_location='cpu', weights_only=False)
model.load_state_dict(state['model'], strict=True)
model.eval()
```

依赖沿用[R34旧发布](../r34/README.md)记录的 Linux、PyTorch2.4.0+cu121、spconv/torch_scatter/FlashAttention 等版本。来源与许可见[RIGHTS_AND_PROVENANCE](../../RIGHTS_AND_PROVENANCE.md)。模型的原训练加载器仍需原数据、AP10派生布局与父版本验证文件；本包用于冻结代码检查/权重加载，不是新机器免配置完整训练发行版，不绕过原验证门槛。

旧 `code/r34/` 是不同输入协议的历史结果，继续原样保留。R38＋密集邻域的新训练不属于此包。
