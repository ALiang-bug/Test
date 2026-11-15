# 六维核查训练与检索修订（Test）

本仓库用于新的图文核查研究；原始 `MAR_KDD` 不作修改。

主流程：领域角色 → 六维 Draft → 网页/固定案例检索 → Critique → Refine → 决策。
事实判断、媒体操纵、误导机制、表达策略与数据集标签分别记录。

- [研究方案与代码对应](docs/SIXDIM_IMPLEMENTATION.md)
- [事件划分、教师质检及两阶段 LoRA 操作说明](docs/DATA_AND_TRAINING.md)
- [原 MAR 基线说明](docs/MAR_BASELINE_README.md)

## 默认安全的离线检查

```bash
python -m unittest discover -s tests -v
python -m sixdim.split --help
python -m sixdim.teacher --help
python -m sixdim.data --help
python -m sixdim.train --help
```

`teacher` 默认禁止联网，只能显式使用离线响应回放或主动开启网络。
`train` 默认仅检查数据和配置，不下载权重、不加载模型、不创建训练目录。
实际教师调用需要 `--allow-network`；实际训练需要 `--execute-training`，并要求冻结测试集清单。

原 MAR 的 `MAR_framework/`、`reverse_image/` 和原数据处理脚本保留为实验基线，
与新流程有关，因此未删除。新代码不导入原 MAR 模块。

- [Cloud finetuning and paired benchmark](docs/FINETUNING_CLOUD.md)

```bash
python -m sixdim.benchmark --help
```
