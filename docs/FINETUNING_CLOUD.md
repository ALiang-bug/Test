# 六维微调与配对评测：云端执行手册

开发仅在 Test。原始 MAR_KDD 是参考实现，不向其中写入训练代码。
本次新增功能只经过离线测试，不能据此声称 GPU 训练成功或性能提升。

## 研究与实现边界

学生为 Qwen/Qwen3-VL-8B-Instruct 的 BF16 原始权重，不使用 32B FP8 推理权重训练。
FakeAgent 的可借鉴部分是分维度分析、感知和判别分离、证据定位与中间产物缓存；
公开代码并非完整的教师 QA 微调流水线，不能把它的检测输出直接视为正确监督。
公共 POLICY 现在明确六维职责，限制“缺少证据就判假”、图像未出现某事就算矛盾、情绪直接推出事实真假。
最终强制四分类 benchmark_pred 与 factual_verdict=insufficient 分离。
模型自己给出的解释/自信度不等于事实依据，结构 QC 不等于语义正确。

## 0. 环境与资源

从 Test 根目录运行。单 A800 80GB 上训练前，用户应在原 vLLM 终端 Ctrl+C 停止 32B 服务，
然后用 nvidia-smi 确认进程已退出、显存释放；不要同时启动 32B 教师和 8B 训练。
容器可直接安装 requirements-training.txt，先 pip check，再实际 import 验证。
不在这里用无约束升级命令覆盖已经工作的 MAR 环境；记录 torch/transformers/peft/accelerate 版本。

```bash
cd /data/home/202422011133/MMFND/Test
export USE_TF=0
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir /data/home/202422011133/MMFND/models/Qwen3-VL-8B-Instruct
```

上述下载是用户执行的后续操作，本地代码不会自动下载权重。若 Test 无图片，可在命令的
--image-root 指向 /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench，读取即可。
生成的 SFT images 使用绝对路径，必须在云端重新导出，不能直接复制 Windows 导出路径。

## 1. 划分与冻结

按 DATA_AND_TRAINING.md 先生成 provisional 划分，再人工审核事件映射并重新划分。
人工事件审核不能用全量 pending 改 approved 代替。正式比较前冻结训练/开发/测试，
同一事件的不同真假变体不能跨集合。外部依据和教师生成仅用于 train/dev。

```bash
python -m sixdim.benchmark freeze --input outputs/splits-reviewed/test.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/benchmark.test.jsonl
```

freeze 固定测试文本、标签、图片哈希、事件 ID。不能在看过测试结果后修改提示词再选最好结果。
允许 provisional 测试用于功能测试，但 compare 会标记事件审核尚未完成，不能当作无泄漏正式实验。
训练执行要求 --heldout，拒绝空清单、缺 ID、非 test 记录及跨集合事件/样本重叠。

## 2. 教师数据：保留本地和付费两种端点

两者使用相同 OpenAI 兼容 Model、输入盲标和审核规则，--base-url/--model 指定模型。
本地教师可使用已有 32B 服务；付费教师的视觉能力、JSON 接口和预算需后续单独确认。
密钥从 MAR_TEACHER_API_KEY 读取，不写进脚本或版本库。本地可使用 EMPTY。
下列仅为命令形式，不表示授权立即请求付费端点。

```bash
# 本地教师：先启动已有 32B 服务。先生成少量样本验证视觉/JSON 能力。
export MAR_TEACHER_API_KEY=EMPTY
python -m sixdim.teacher generate --stage A --input outputs/splits-reviewed/train.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/A.train.raw.jsonl --limit 5 --allow-network --base-url http://127.0.0.1:19887/v1 --model Qwen3-VL-32B-Instruct-FP8
```

付费教师只替换 base-url/model/环境密钥；不要把占位符当作可执行配置。
先 5 条检查，再扩大到约 500 新闻的代表性 pilot；dev 单独生成、单独审核，禁止处理 test。
A 阶段包含：盲标 observation -> 有来源 verification -> 单独 verifier；所有结果 pending。
网络错误不能伪装成成功样本。验证模型可独立指定，默认同一模型的第二次调用不算独立模型共识。
质检、人工 review 的具体格式和命令见 DATA_AND_TRAINING.md。
若没有外部证据，不能为真实历史事件自动补造身份和日期；标记 insufficient，审核不通过的不要训练。
每个通过审核的 A 新闻导出 6 个维度 QA + 1 个无外部证据 draft 报告。
没有 claim-level 人工证据审核，不能仅凭与 fake_cls 一致就接受。

## 3. A 阶段 LoRA：dry-run -> 2 步冒烟 -> 正式训练

先将审核通过的 train/dev 各自导出 A.train.sft.jsonl、A.dev.sft.jsonl。
导出命令见 DATA_AND_TRAINING.md。先不加 --execute-training 做数据预检：

```bash
python -m sixdim.train --stage A --train outputs/A.train.sft.jsonl --dev outputs/A.dev.sft.jsonl --heldout outputs/benchmark.test.jsonl --model /data/home/202422011133/MMFND/models/Qwen3-VL-8B-Instruct --output-dir checkpoints/A-preflight
```

预检不加载模型、不分配显存、不创建输出目录。通过后加 --execute-training，先短跑：

```bash
python -m sixdim.train --stage A --train outputs/A.train.sft.jsonl --dev outputs/A.dev.sft.jsonl --heldout outputs/benchmark.test.jsonl --model /data/home/202422011133/MMFND/models/Qwen3-VL-8B-Instruct --output-dir checkpoints/A-smoke --max-steps 2 --max-length 4096 --max-pixels 262144 --execute-training
```

确认 loss 有限、无 OOM、adapter 正常保存后，用新的输出目录 A-full，去掉 --max-steps 2，
采用 2 epochs 起步。像素/长度预算是训练超参，记录在 manifest，不要为绕开失败静默截断监督。
显存不够时先降低像素预算/长度并检查样本长度，不改变测试集或标签。
只训练语言模型 attention/MLP 的 LoRA，视觉编码器和连接器冻结，loss 只覆盖 assistant。
A-smoke 只验证训练链路，不能作为正式模型。

## 4. 微调前后统一评测

基座和 adapter 用同一 vLLM 版本、同一 BF16 权重、同一图片设置分别服务。
先做基座 run；再加载语言 LoRA adapter 服务做 tuned run。加载命令需按云端 vLLM
实际支持的 Qwen3-VL LoRA 接口验证，不能用 32B FP8 教师当 8B 基线。
run-id 用可追溯权重/adapter 标识，另保存服务启动参数和版本。

```bash
python -m sixdim.benchmark run --input outputs/benchmark.test.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/base.test.jsonl --base-url http://127.0.0.1:19887/v1 --model Qwen3-VL-8B-Instruct --run-id base-bf16 --allow-network
# adapter 服务验证后，用其 served model 名重复同一命令，输出改为 A.test.jsonl、run-id 改为 A-full
python -m sixdim.benchmark compare --base outputs/base.test.jsonl --tuned outputs/A.test.jsonl --output outputs/A.comparison.json
```

本轮严格不调用 Web/记忆，比较的是学生本身，不是完整 MAR 框架收益。
报告四分类/二分类 accuracy、macro-F1、输出有效率、insufficient 数量和错->对/对->错。
失败也保留在总样本和 F1 的 FN 中，不能只比较成功子集。输入/协议不一致则拒绝比较。
当前 compare 为单次描述性对比，不自动宣称显著提升。正式结论需多个训练 seed，
并人工核查六维答案是否有来源、是否越界、是否把“不知道”当“假”。

## 5. B 阶段：证据修订

A 模型完成后，使用它在 train/dev 上生成实际 draft，再由教师产生 critique/refinement。
保留改动前后和证据 ID，正确的 draft 也应有 retain 示例，不能所有样本都强制改答案。
同样经过质检和人工审核，导出 B 训练数据；--init-adapter 指向 A-full/adapter，
--replay 指向 A.train.sft.jsonl，混入约 25% A 数据。按同一冻结测试集对比 base/A/B。
网络暂不通时，可提供人工核验的本地来源，不自动编造 Web 检索结果。
B 的提升要分清无证据 draft 能力和给定证据修订能力；本轮 benchmark 测量前者。

## 参考

- https://github.com/Aiyistan/FakeAgent/blob/master/agents/consistency_analyzer.py
- https://github.com/Aiyistan/FakeAgent/blob/master/agents/integrator.py
- https://github.com/Aiyistan/FakeAgent/blob/master/workflows/detector.py

尚待实际验证：Qwen3-VL GPU forward/backward、LoRA 服务加载、教师真实质量、端到端训练耗时与收益。

## 6. Transformers 本地基座/adapter 配对评测（优先使用）

这条路径不依赖 vLLM 对多模态 LoRA 的服务支持，不会自动下载权重。
先停止占用同一 GPU 的 vLLM 服务，一次只运行一个模型。两次推理采用相同 BF16、
图片预算、greedy decoding、完整六维报告提示词；只有 adapter 权重不同。

```bash
python -m sixdim.benchmark run --backend local --execute-local --input outputs/benchmark.test.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/base.local.test.jsonl --model /data/home/202422011133/MMFND/models/Qwen3-VL-8B-Instruct --run-id base-bf16
python -m sixdim.benchmark run --backend local --execute-local --input outputs/benchmark.test.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/A.local.test.jsonl --model /data/home/202422011133/MMFND/models/Qwen3-VL-8B-Instruct --adapter checkpoints/A-full/adapter --run-id A-full
python -m sixdim.benchmark compare --base outputs/base.local.test.jsonl --tuned outputs/A.local.test.jsonl --output outputs/A.local.comparison.json
```

禁止把 API 基线与本地 adapter 输出直接配对：协议 hash 不同会拒绝。
实际 GPU 推理还未运行。若出现格式错误，保留 failed 记录，不修改测试提示词来修饰结果。
结构有效率只说明输出格式，不能替代人工六维质量评估。
