# 复现并解读对比

[English](reproduction.md) · [工具书](index.zh-CN.md)

执行前查看源码与参考：

```sh
tilelang-weft-bench show q4_k_projection_decode --target k1 --config .local/bench.local.json
tilelang-weft-bench list --kind baselines --target k1 --config .local/bench.local.json
tilelang-weft-bench list --kind suites --target k3-a100 --config .local/bench.local.json
```

日常选择受影响单项；完整套件使用 `full`。K3 `full` 包含全部 136 项，
`ime2` 子集为其中三项矩阵用例。

```sh
tilelang-weft-bench op q4_k_dot --target k1 --config .local/bench.local.json \
  --check-only --output output/current/k1/q4_k_dot
tilelang-weft-bench op --suite full --target k3-a100 --config .local/bench.local.json \
  --baseline output/baselines/k3-a100 --output output/current/k3-a100/full
```

固定参考包含明确的库版本和匹配测量合同，baseline 索引保留原始观察。
确认环境漂移后，只通过同一引擎配对补测受影响参考，历史记录继续保留。
计时参数与选中参考记录一致；既有 K3 参考分组包含不同 warmup 次数。
示例配置允许每个构建或执行进程最多运行两小时。这个等待上限包含源码 JIT 和
输入准备，不改变样本计时窗口。生成的只读输入保存一次，由缓存验证和原生 worker
复用。

使用现有比较命令复用成功记录：

```sh
tilelang-weft-bench compare output/baselines/k1 output/current/k1/full \
  --baseline-impl ggml --output output/reports/k1
tilelang-weft-bench explain output/current/k1/full --case q4_k_dot
tilelang-weft-bench replay output/current/k1/full --case q4_k_dot \
  --output output/current/k1/q4_k_dot
```

`results.json` 保存样本、检查、编译／缓存观察及产物位置；`results.csv` 是引擎的
精简视图。公开的全量算子 CSV 增加源码变体、原生／参考耗时、独立 Triton 对照与
JIT 指标。数值检查记录说明完整检查与抽样检查的实际范围。

倍率大于 1 表示 TileLang 原生 workload 更快。缺少匹配外部参考的 10 项保留空倍率，
仍记录自己的正确性和耗时。对比 Triton 时，先辨明作者算法和准备边界的差异。

从显式选定的结果和固定参考生成展示表：

```sh
python -m tilelang_weft_bench.report \
  --run sg2044=output/current/sg2044/selected --reference sg2044=output/baselines/sg2044 \
  --run k1=output/current/k1/selected --reference k1=output/baselines/k1 \
  --run k3-a100=output/current/k3-a100/selected --reference k3-a100=output/baselines/k3-a100 \
  --run x86-1t=output/current/x86-1t/selected --reference x86-1t=output/baselines/x86-1t \
  --triton x86-1t=output/comparisons/x86-1t --plot --output output/reports/latest
```

正式汇总拒绝缺少 JIT／缓存／热调用证据、native 样本或必要参考的记录。
开发时 `--partial` 明确保留不完整状态。projection prefill 沿用 64 个输出抽样，
并检查所有输出有限；decode 检查全部列。量化 activation 字节按目标声明的
storage 规则进行检查。

结果和产物保存在本仓库忽略跟踪的 `output/` 中。Release 表格与图片是清理私有信息
后的副本；SSH 主机、机器路径、模型和凭据留在本地。重复运行通过引擎的输出事务更新
同一 target/case 当前状态，失败按失败保留。
