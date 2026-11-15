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
