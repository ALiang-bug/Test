# 数据制作与 LoRA 训练入口

全部命令从 Test 根目录执行。本文的路径为示例，需要真实图像和人工审核资料。
本次实现过程只运行测试与 dry-run，没有调用教师 API、下载模型或开始训练。

## 1. 事件划分（先于 QA 生成）

```bash
python -m sixdim.split --train-input data_processing/MMFakeBench/train_set.json --test-input data_processing/MMFakeBench/test_set.json --image-root dataset/MMFakeBench --output-dir outputs/splits-provisional
```

输出 train/dev/test/excluded.jsonl、audit.json 和 event_map.template.jsonl。所有输出拒绝覆盖。
未提供人工事件映射时 grouping_status=provisional，不能直接进入教师生成。

审核模板示例：

```json
{"sample_id":"MMFakeBench_test/real/example.png","event_id":"reviewed-event-001","review_status":"approved"}
```

sample_id 默认是原 image_path，也可由输入显式提供；不得跨样本重复。
审核者需把相同真实事件的样本设置为同一 event_id，跨 train/test 全部覆盖。
不能仅将所有 pending 改为 approved；必须核对事件和近重复图像。
完成后重新划分，保留旧产物：

```bash
python -m sixdim.split --train-input data_processing/MMFakeBench/train_set.json --test-input data_processing/MMFakeBench/test_set.json --image-root dataset/MMFakeBench --event-map outputs/event_map.reviewed.jsonl --output-dir outputs/splits-reviewed
```

约 90/10 的训练/开发比例按事件组分配，类比例近似；与冻结测试集有事件、精确文本或图片哈希重叠的训练样本进入 excluded。
同一事件可能有不同真假变体，不能拆到不同划分。
未传 image-root 时不检查图片哈希，audit 会明确记录，不能宣称完成图像泄漏排查。

## 2. 教师生成与离线回放

输入使用已审核 train 或 dev 文件；test/excluded/provisional 禁止生成。
可在样本 evidence 字段附加核验资料：

```json
{"id":"E1","kind":"document","source":"https://source.example/report","excerpt":"原始来源摘录","published_at":"2026-01-01"}
```

kind 支持 web/web_snippet/document/analogy。程序不自动联网补证据；缺少证据时必须允许保留判断。
I0、T0 为保留 ID，分别表示原始图片和文本；外部证据不能使用它们。

离线响应文件格式为 `sample_id -> phase -> JSON响应`：

```json
{"sample-1":{"observation":{},"verification":{},"review":{"pass":true,"issues":[]}}}
```

observation、verification 必须填写真实六维报告结构（见 sixdim.core.REPORT）；空对象会被质检拒绝。
A 阶段使用 observation/verification/review；学生采集使用 student_draft；B 阶段使用 critique/refinement/review。

```bash
python -m sixdim.teacher generate --stage A --input outputs/splits-reviewed/train.jsonl --image-root dataset/MMFakeBench --responses /path/to/offline-responses.json --output outputs/A.raw.jsonl --limit 500 --seed 42
```

默认按 split/原始类别分层抽取至多 500 条，保存种子和来源指纹；每完成一条就写入并 flush。
失败也写出记录，不静默消失。不支持覆盖或自动续跑；重试应选择失败子集和新的输出路径。

真实调用必须显式指定 --allow-network、--base-url、--model，并从 MAR_TEACHER_API_KEY 读取密钥。
可选 --verifier-model/--verifier-base-url 和 MAR_VERIFIER_API_KEY 使用另外一个审查模型；否则只是同一模型的独立调用，不应声称为独立教师一致性。
代码不会判断一个 endpoint 是否收费，因此不加 --allow-network 就绝不调用。

生成阶段不把 fake_cls、gt_answers 传给教师；观察阶段不提供外部来源。
原始真假标签不能直接转换成 factual_verdict。模型分歧或来源不足必须留给人工，不强制生成符合标签的理由。

## 3. 自动 QC 与人工审核

```bash
python -m sixdim.teacher qc --input outputs/A.raw.jsonl --output outputs/A.qc.jsonl
```

检查六维完整性、观察/最终报告、引用 ID、修订动作和标签分歧。
自动 QC 与模型审查都不会把 review_status 变为 approved。

人工审核文件示例（annotation_hash 必须从待审核记录复制）：

```json
{"sample_id":"sample-1","annotation_hash":"hash-from-generated-row","decision":"approved","reviewer":"human-reviewer-id","reviewed_at":"2026-11-14","checks":{"evidence_grounded":true,"no_fabricated_facts":true,"dimension_boundaries":true,"citations_verified":true,"intent_restraint":true}}
```

拒绝时 decision=rejected，并逐项填写 true/false。标签或审查器有分歧时，需要 adjudication 说明。
修正内容后重新运行 QC，并使用新内容计算 annotation_hash，再人工确认；旧审核不能用于新答案。
审核覆盖观察报告、最终报告、图片和证据，不只是语言质量。

```bash
python -m sixdim.teacher review --input outputs/A.raw.jsonl --reviews outputs/A.human-reviews.jsonl --output outputs/A.reviewed.jsonl
```

summary 统计 500 条先导、至少 100 条人工复核、98% 结构通过、90% 人工检查通过、关键编造不高于 5% 的扩量门槛。
这些是审核表上的统计，不是无需人工就能测出的事实准确率；人工抽样必须有代表性。

## 4. SFT-A 导出

```bash
python -m sixdim.data --input outputs/A.reviewed.jsonl --image-root dataset/MMFakeBench --stage A --output outputs/A.train.sft.jsonl
```

开发集单独生成和审核，然后使用 --split dev 导出 A.dev.sft.jsonl。
只接受本划分、已审核事件、人工批准且内容哈希/图片哈希未变化的样本。
每条新闻输出六个证据问答和一条观察报告；模型实际收到所有被引用的证据。

## 5. 学生初稿与 SFT-B

对 A 阶段训练后的学生服务采集初稿。输入仍是训练/开发图文，不能使用测试新闻。
离线测试可用 --responses 回放；真实采集同样要求显式 --allow-network，学生密钥使用 MAR_API_KEY。

```bash
python -m sixdim.teacher drafts --input outputs/splits-reviewed/train.jsonl --image-root dataset/MMFakeBench --responses /path/to/student-responses.json --output outputs/student-drafts.jsonl
python -m sixdim.teacher generate --stage B --input outputs/student-drafts.jsonl --image-root dataset/MMFakeBench --responses /path/to/refinement-responses.json --output outputs/B.raw.jsonl
```

B 需要 student_model、draft 和可靠 evidence；refinement 返回 report 与 changed_claims。
对 B 重复 QC/人工审核，再使用 sixdim.data --stage B 导出训练和开发文件。
critique 与 refine 是两个独立监督任务，refine 的 assistant 目标直接是完整报告，与在线推理一致。

## 6. LoRA 默认 dry-run

```bash
python -m sixdim.train --train outputs/A.train.sft.jsonl --dev outputs/A.dev.sft.jsonl --stage A --model /models/Qwen3-VL-8B-Instruct --output-dir outputs/lora-A --heldout outputs/splits-reviewed/test.jsonl
```

不会 import torch/transformers/peft、加载或下载模型、创建训练目录。
检查输入消息、阶段、审核标记、图片哈希、train/dev/heldout 的样本和事件泄漏。
可不传 heldout 做初步检查，但实际训练强制要求它。

B 从 A adapter 继续，混入约 25% A 训练例子，取整后输出实际比例：

```bash
python -m sixdim.train --train outputs/B.train.sft.jsonl --dev outputs/B.dev.sft.jsonl --stage B --model /models/Qwen3-VL-8B-Instruct --init-adapter outputs/lora-A/adapter --replay outputs/A.train.sft.jsonl --output-dir outputs/lora-B --heldout outputs/splits-reviewed/test.jsonl
```

初始超参数（可通过命令行调整，未宣称最优）：rank=32、alpha=64、dropout=0.05、lr=1e-4、2 epochs、每卡 batch=1、accumulation=16、bf16、max_length=8192。
只选择 language_model 中 attention/MLP 投影，不给视觉模块加适配器。
B 的 rank/alpha 以已加载 A adapter 为准；新建 A 时才使用 CLI 值。

只有明确添加 --execute-training 才会进入真实训练路径。本次未执行。
真实执行前需安装适配 GPU 的 torch 与 requirements-training.txt，准备完整本地模型和图像。
local_files_only=True 禁止训练入口偷偷下载权重。

实际训练先检查所有样本的完整分词结果：prompt 和完整序列必须前缀一致，只对 assistant tokens 计算 loss；padding=-100。
过长样本直接报错，不截断图片 token 或监督答案。随后记录版本、配置、数据指纹、可训练参数量和 adapter。
当前只提供单机 Trainer/可选 torchrun DDP 路径，不包含自动显存配置、DeepSpeed 或 QLoRA。

## 验证边界

离线集成测试覆盖事件隔离 → 教师回放 → 人工审核 → A 导出 → 学生回放 → B 导出 → A/B dry-run。
它不代表真实教师质量、真实模型训练兼容性或性能提升。
训练依赖尚未在当前环境安装；GPU 上需要额外执行分词、前向与小步训练验证，再扩大规模。

官方接口参考：
- https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl
- https://huggingface.co/docs/peft/developer_guides/custom_models
- https://huggingface.co/docs/transformers/en/main_classes/trainer
