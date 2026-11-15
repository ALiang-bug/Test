# 教师提示词与语义质检重跑

本模块只用于 train/dev 候选的接口试验，provisional 不会伪装成 reviewed。
不产出可直接训练的审批记录。不调用 Web，不根据 fake_cls 强制修正解释。

变更：公共 POLICY 区分直接观察、文本主张、内部一致性、独立事实验证；
限制身份/历史时间/声音/意图推断；媒体无异常不代表来源真实。
fact_conflict_checking=insufficient 时最终事实状态也必须 insufficient。
保留强制 benchmark_pred，与证据充分性分离。

语义 audit 是保守规则：结论超过核验、意图推测标为 supported、事实问题重复一致性、
unmanipulated 缺少来源复核。英文词规则不能覆盖所有表达，语义通过只表示无规则命中，
不等于正确；命中 warning 也不自动证明内容错误。需模型独立复核和人工裁决。
正式教师生成的 verifier 同时看到中间报告和规则标记。
带标记的记录审批和 SFT 导出均要求显式 adjudication，不能靠机器 pass 自动批准。

保持同一个 32B 服务和生成配置，旧文件保留不覆盖。从 Test 根目录执行：

```bash
python -m sixdim.probe run --candidates outputs/pilot-candidates-16-v2.jsonl --before outputs/teacher-interface-probe-2.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/teacher-interface-probe-2-v2.jsonl --allow-network
python -m sixdim.probe compare --before outputs/teacher-interface-probe-2.jsonl --after outputs/teacher-interface-probe-2-v2.jsonl --output outputs/teacher-interface-probe-comparison-v2.jsonl
```

若使用其他教师，传 --base-url / --model 并从 MAR_TEACHER_API_KEY 读取密钥。
默认无 --allow-network 不发请求。run 从旧文件 ID 选择同两条候选，不以标签选答案。
保存新 input、image_sha256、prompt_hash。旧脚本没有这些哈希，所以无法证明完整输入
和原模型权重完全一致；compare 只能检查相同 sample_id、teacher_model，输出会注明限制。
输出拒绝覆盖，失败样本保留，不静默缩小对比集。

对比人工关注：可见细节是否仍编造；身份/过去时/声音是否从文字注入图像；
意图是否保留假设；事实核验是否真正分离；媒体来源是否谨慎；六维修正是否影响标签。
两条开发样本属于提示词调试材料，不是泛化性能评估，规则命中减少不能单独证明提升。

## V3: 第二次模型调用逐主张、逐维度复核

probe run 现在每条调用两次：先生成报告，再把原图、原文、编号 C1..Cn 的
原报告主张、六维报告和规则标记发送给复核模型。同一个模型的第二次调用并非
独立模型共识。复核不接收真假标签，不自动改写原报告。
要求每个主张精确覆盖、原文引用、六维精确覆盖、证据编号合法和总 verdict 一致。
复核中的 insufficient 表示无法确认草稿是否有据，不是要求所有事实都确定。
草稿正确保留未知时，复核可以 pass；存在具体错误时 revise。

保存 evidence_review.review、review_model、review_prompt_hash、report_hash。
machine_review_pass 仅在规则和复核都通过时为 true，仍然 pending，不能直接导出。
status=ok 仅表示两次调用及结构验证完成，不代表复核通过。
复核失败/解析错误保留报告并记录 failed；不自动重试、修复或覆盖。
compare 在 V2/V3 都有 image_sha256 与 input 时严格核对；权重版本仍需用户保持一致。

```bash
python -m sixdim.probe run --candidates outputs/pilot-candidates-16-v2.jsonl --before outputs/teacher-interface-probe-2-v2.jsonl --image-root /data/home/202422011133/MMFND/MAR_KDD/dataset/MMFakeBench --output outputs/teacher-interface-probe-2-v3.jsonl --allow-network
python -m sixdim.probe compare --before outputs/teacher-interface-probe-2-v2.jsonl --after outputs/teacher-interface-probe-2-v3.jsonl --output outputs/teacher-interface-probe-comparison-v3.jsonl
```

人工重点：图像是否真能支持身份和过去时；事实核验是否逐条针对原文主张，
而不是将翅膀/角的媒体异常当作原文中的假主张；制作方式是否无依据写成 AI。
复核仍可能与生成模型共享盲点，不能仅以 overall_pass 判断数据正确。
