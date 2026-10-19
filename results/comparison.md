本次范围：两个新训练 checkpoint × 四项完整 benchmark，共 5,272 条预测。两项 MME 未评测。Judge 使用当前助手加匹配规则，与 Vision-OPD 使用的 Qwen2.5-72B-Instruct judge 不同，结果不作严格等价比较。原始回答、逐题判分来源和语义复核理由均已保存。

# 评测对比

两组题目、gold 和推理设置核对通过；checkpoint 来自本次 ULVR bbox_crop 正式训练。

| benchmark | n | step_30 | step_65 | delta_pp |
| --- | --- | --- | --- | --- |
| vstar | 191 | 76.44% | 73.30% | -3.14 |
| zoombench | 845 | 46.15% | 43.43% | -2.72 |
| hrbench-4k | 800 | 79.38% | 76.00% | -3.38 |
| hrbench-8k | 800 | 74.75% | 70.75% | -4.00 |
| macro（等权） | — | 69.18% | 65.87% | -3.31 |
| micro（题数加权） | 2636 | 67.11% | 63.77% | -3.34 |

## Category

| benchmark | category | n | step_30 | step_65 |
| --- | --- | --- | --- | --- |
| vstar | direct_attributes | 115 | 74.78% | 71.30% |
| vstar | relative_position | 76 | 78.95% | 76.32% |
| zoombench | unknown | 845 | 46.15% | 43.43% |
| hrbench-4k | cross | 400 | 70.50% | 65.75% |
| hrbench-4k | single | 400 | 88.25% | 86.25% |
| hrbench-8k | cross | 400 | 66.75% | 65.00% |
| hrbench-8k | single | 400 | 82.75% | 76.50% |

本实验准备工具从原始 Parquet 补回 MME category；若导入的旧结果为 unknown，需另补原始分类元数据。

案例按 benchmark/category/变化方向分层抽取，固定随机种子；cases.jsonl 保留原始回答、题目、图片路径和 judge 来源。
结果解释：step_65 四项均低于 step_30，四项 macro 下降 3.31 个百分点。题级比较有 275 条退步、187 条进步。当前没有完整 base 对照，不能据此判断训练相对 base 是否有效，也不能用四项平均与其他六项平均比较。1 条语义歧义题的相反判法只影响四项平均约 0.03 个百分点，不改变方向。
