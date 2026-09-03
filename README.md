# LLM 预训练集群仿真器（显存建模模块）

单文件零依赖前端：`index.html`（HTML + CSS + 原生 JS，无任何外部库）。

## 运行方式

任意一种：

```bash
# 方式一：本地静态服务（推荐；外部预设文件 presets.json 通过 fetch 加载）
python3 -m http.server 8017 --directory .
# 浏览器访问 http://127.0.0.1:8017
# 方式二：直接双击打开 index.html（file:// 下无法 fetch presets.json，
#         自动回退到页面内置的同一份默认预设）
```

预设配置抽离在 `presets.json`（含 `default` 默认值 与 `presets` 预设列表），
页面加载时覆盖内置 fallback 并重建预设下拉；修改预设无需改动页面代码。

桌面端可拖动右侧参数面板左边缘（蓝色竖条）调整面板宽度（300–620px）；窄窗口（<1080px）下
面板自动回到顶部流式布局，拖拽不可用。显存明细（集群/单卡）与通信耗时对比均在大卡片内并排展示。

回归测试（可选，需要 Node.js）：

```bash
node tests/verify_mem.js      # 主仿真器（显存 + 通信）
node tests/verify_dsv4.js     # DeepSeekV4 参数量/动态激活显存页
node tests/verify_sync.js     # 同步桥接链路（写入/跳转/消费/fetch 不覆盖）
```

## 配套页面：DeepSeekV4 参数量与动态激活显存

`deepseek4.html`（主页右上角"DeepSeekV4 参数量 / 激活显存 ›"可进入）：参照
`DeepSeek4_参数量及MFU计算.xlsx`（v4pro / v4flash 两页）与 `dsk4_显存分析.xlsx` 的口径实现：

- **模型总参数量**：v·h 词表 + 每层 DSA Attn(含 mhc) + Indexer×L_i + Compressor=4×L_c4 + Compressor=128×L_c128
  + MoE（routed n_router + shared n_share）。
- **模型激活参数量**：v·h + L_moE×(DSA Attn + TopK×单专家 + n_shared×单专家)，随 TopK 变化。
- 结构预设（v4pro）加载自 `dsv4_presets.json`，**校准自 DeepSeek4_参数量及MFU计算.xlsx 的 v4pro 页**：
  Lmoe=96、TopK=16、n_router=1024、n_share=1、f_e=4096（×r_e=2）、Li=30 / Lc4=30 / Lc128=31。
  输出与 Excel 逐项一致：模型总参数量 **8698.184466432 B（≈8.70T）**、模型激活参数量 **173.59568896 B**、
  单层 DSA Attn（含 mhc）0.30127 B、单专家 0.08808 B（单层 MoE 合计 90.282 B）。
  页面内置同值 fallback（fetch 失败时回退），修改预设只需改 json。
- **动态激活显存（GiB）**：校准自 dsk4_显存分析.xlsx 的 pro 页（与 v4pro 同构：96 层、TopK=16、
  f_e=4096、nr=1024、TP=1）。逐项与表一致：layer input 0.21875、logits 0.98633、cross entropy 1.97266、
  loss 峰值 4.93164；单层 Attention 临时（表 G40，含 mhc）= **2.40625**，单层 MoE 临时（表 G49，含 mhc）=
  **4.96094**，单层激活（表 J43）= **7.3672**，总激活（表 J44）= **707.25**（×96 层）；重计算保存层输入 = 21。
  关键口径：router 临时 = s·b·n_router·TopK 字节（核对五页 router 值全部吻合）；mhc 在 attention 与 MoE 两图
  峰值各保有一份。

"同步到仿真器"：页面右上角"⇢ 同步到仿真器"按钮将单层 DSA Attn 参数量、单个专家参数量、
单层动态激活（GiB）以及结构字段（MoE 层数 / 路由专家数 / TopK / f_e / hidden / seq_len）
写入 localStorage 后**自动跳转**至主仿真器 index.html；主界面一次性应用全部字段（应用后清除），
并在顶部配置消息区完整列出修改了哪些输入；外部预设 fetch 完成后不会覆盖已同步的输入。

注：动态激活按公式字面实现；参考显存表 v4pro 页使用 topk=16 且 router 维度取 8,192，
本页默认 topk=30（与参数量表 v4pro 一致），router 按 n_router 计，故与显存表不完全相同的输入
下数值略有差异（输入对齐后一致）。

## 当前建模内容（v0.1 · 显存）

### 输入
- 模型参数：单层 Attention 参数量（B）、单个 MoE 专家参数量（B）、专家数量、层数、
  序列长度、Hidden Size、MoE Hidden Size、TopK（EP Alltoall 用）、单层动态激活（GiB）
- 并行方式：FSDP、EP、EFSDP、CP、单卡 HBM 容量（仅用于占用率展示）、
  Recompute（勾选）、SwapOptimizer（勾选，开启后单卡优化器 = 0）、
  SwapActivation（勾选，开启后 Layer Input = 0）
- 单个完整 DP 域卡数 = MAX(FSDP × CP, EP × EFSDP × CP)（FSDP 为 Attention 权重切分度；
  取 Attention 域（FSDP×CP）与 MoE 域（EP×EFSDP×CP）两者较大者，卡数不再额外乘入）
- 通信网络（三层拓扑）：机内带宽（GiB/s）/ 机内大小、超节点带宽（GiB/s）/ 超节点大小、
  超节点间带宽（GiB/s）、FSDP 通信 dtype（下拉：FP32=4 / BF16=2（默认） / FP8=1 字节，Allgather/ReduceScatter 载荷精度）、带宽利用率（默认 0.8，算法带宽 = 理论带宽 × 利用率）

### 输出（全部实时计算）
| 项 | 公式 |
| --- | --- |
| 单个完整 DP 域卡数 | MAX(FSDP × CP, EP × EFSDP × CP) |
| 总模型权重 FP32 (GiB) | (单层 Attention 参数量 + 单个专家参数量 × 专家数量) × 层数 × (4×10⁹/1024³) |
| 总模型梯度 FP32 (GiB) | = 总模型权重 |
| 总优化器 2×FP32 (GiB) | = 总模型权重 × 2（集群口径不随 SwapOptimizer 变化） |
| 总静态显存 (GiB) | 权重 + 梯度 + 优化器 |
| 总动态激活 (GiB) | 单层动态激活 / CP × 层数（输入即 GiB，无需换算） |
| 单卡模型权重 FP32 (GiB) | (单层 Attention 参数量 / FSDP + (单个专家参数量 × 专家数量) / (EP × EFSDP)) × 层数 × (4×10⁹/1024³) |
| 单卡模型梯度 / 优化器 | = 单卡权重；= 单卡权重 × 2；**SwapOptimizer=True 时优化器 = 0 GiB（交换出卡）** |
| 单卡动态激活 (GiB) | Recompute=True → 总动态激活 / 层数；否则 = 总动态激活 |
| 单层预取权重 BF16 (GiB) | (单层 Attention 参数量 + 单个专家参数量 × (专家数量/EP)) × (2×10⁹/1024³) |
| 单层临时梯度 FP32 (GiB) | = 单层预取权重 BF16 × 2 |
| Layer Input (GiB) | SwapActivation=True → 0；否则 序列长度 × Hidden Size × 层数 × 2（BF16） / 1024³ |
| 单卡总显存 (GiB) | 权重 + 梯度 + 优化器 + 动态激活 + Layer Input + 预取权重 + 临时梯度 |

单位换算约定：参数量 B → FP32 显存 GiB：**B × 10⁹ × 4 / 1024³ ≈ B × 3.7253 GiB**（系数在代码中为
`GB_PER_B`，`BYTES_PER_PARAM` 可扩展 BF16(2)/FP8(1)）。

## 通信耗时（EP Alltoall · 均衡，v0.2）

> 中央"📡 通信耗时对比"卡片内将各通信模式**并排实时显示**，目前包含三列：
> **EP Alltoall（均衡）**、**FSDP Allgather/ReduceScatter（分层）**、**EFSDP Allgather/ReduceScatter（分层）**。
> 各列独立计算 KPI / 三层流量 / 堆叠条 / 瓶颈层 / 告警，共用三层网络参数
> （EFSDP 的有效层级大小为物理层级 ÷ EP，见下方公式表）。

三层拓扑流量分解（通信域 = EP，机内/超节点内/超节点间逐层聚合，占比之和 = 100%）：

| 项 | 公式 |
| --- | --- |
| 总接受数据 (MB) | 序列长度 × MoE Hidden Size × 2(BF16) × TopK / 1024² / 通信域 × (通信域 − 1)；页面以 GiB 显示（÷ 1024） |
| 每份数据 (MB) | 总接受数据 / (通信域 − 1) |
| 总流量 | 通信域 − 1 |
| 机内流量 | MIN(总流量, 机内大小 − 1) |
| 超节点内流量 | MAX(MIN(通信域, 超节点大小) − 机内大小, 0) |
| 超节点间流量 | MAX(通信域 − 超节点大小, 0) |
| 算法带宽 (GiB/s) | 总流量 / MAX(机内流量/机内带宽 + 超节点内流量/超节点带宽, 超节点间流量/超节点间带宽) × 带宽利用率（默认 0.8）|
| 通信耗时估算 | 总接受数据(GiB) / 算法带宽（页面固定保留 3 位小数）|
| 瓶颈层 | 机内流量/机内带宽、超节点内流量/超节点带宽、超节点间流量/超节点间带宽 中耗时最大者 |

说明：需求原文"机内流量 = MAX(总流量, 机内大小−1)"在 EP>机内大小时会令三档流量之和
超过总流量，经确认按 **MIN** 实现（同层对端最多 机内大小−1）。
通信参数校验：机内大小不得大于超节点大小、存在超节点间流量时超节点间带宽需 > 0。

## 通信耗时（FSDP Allgather / ReduceScatter · 分层，v0.2）

通信域 = FSDP。载荷为单层 Attention 权重（参数量 B × 10⁹ × dtype 字节），与显存建模的
Attention 切分口径一致，可反映 Allgather（正向取参）/ ReduceScatter（反向梯度归约）的通信量：

| 项 | 公式 |
| --- | --- |
| 总接受数据 (MB) | 单层 Attention 参数量 × 10⁹ × dtype 字节 / 1024² / 通信域 × (通信域 − 1) |
| 每份数据 (MB) | 总接受数据 / (通信域 − 1) |
| 总流量 | 通信域 − 1 |
| 机内流量 | 总流量 − 超节点内流量 − 超节点间流量 |
| 超节点内流量 | MAX(MIN(通信域, 超节点大小) / 机内大小 − 1, 0) × (超节点间流量 + 1) |
| 超节点间流量 | MAX(通信域 / 超节点大小 − 1, 0) |
| 算法带宽 (GiB/s) | 总流量 / MAX(机内流量/机内带宽 + 超节点内流量/超节点带宽, 超节点间流量/超节点间带宽) × 带宽利用率（默认 0.8）|
| 通信耗时估算 | 总接受数据(GiB) / 算法带宽（页面固定保留 3 位小数）|
| 瓶颈层 | 机内流量/机内带宽、超节点内流量/超节点带宽、超节点间流量/超节点间带宽 中耗时最大者 |

说明：机内流量采用**余量法**（总流量 − 超节点内 − 超节点间，下限 0），三层占比之和恒 = 100%。
DeepSeekV4 默认（FSDP=2048，dtype=2 BF16，机内 100 GiB/s）：总接受数据 574 MB（0.6 GiB，每份 287 KB）、
三层流量 1,920 / 126 / 1 份（占比 93.8% / 6.2% / 0%）、算法带宽 82.58 GiB/s（×0.8 利用率）、通信耗时 6.786 ms、
瓶颈层=机内（1,920 份 · 100 GiB/s）。

## 通信耗时（EFSDP Allgather / ReduceScatter · 分层，v0.2）

通信域 = EFSDP，载荷为单层全部 MoE 专家权重（= 单个专家参数量 × 专家数量）。
EFSDP 按 EP 子域组织通信，故有效层级大小 = 物理层级大小 ÷ EP（下限 1）：

| 项 | 公式 |
| --- | --- |
| 总接受数据 (MB) | 单层 MoE 参数量 / EP × 10⁹ × dtype 字节 / 1024² / 通信域 × (通信域 − 1) |
| 每份数据 (MB) | 总接受数据 / (通信域 − 1) |
| 总流量 | 通信域 − 1 |
| 有效超节点大小 | MAX(超节点大小 / EP, 1)；有效机内大小 = MAX(机内大小 / EP, 1) |
| 机内流量 | 总流量 − 超节点内流量 − 超节点间流量 |
| 超节点内流量 | MAX(MIN(通信域, 有效超节点大小) / 有效机内大小 − 1, 0) × (超节点间流量 + 1) |
| 超节点间流量 | MAX(通信域 / 有效超节点大小 − 1, 0) |
| 算法带宽 (GiB/s) | 总流量 / MAX(机内流量/机内带宽 + 超节点内流量/超节点带宽, 超节点间流量/超节点间带宽) × 带宽利用率（默认 0.8）|
| 通信耗时估算 | 总接受数据(GiB) / 算法带宽（页面固定保留 3 位小数）|
| 瓶颈层 | 机内流量/机内带宽、超节点内流量/超节点带宽、超节点间流量/超节点间带宽 中耗时最大者 |

说明：机内流量与 FSDP 一致用**余量法**，三层占比之和恒 = 100%。总接受数据页面以 GiB 显示（÷ 1024）。
DeepSeekV4 默认（EFSDP=16，EP=128，dtype=2 BF16）：有效超节点 1024/128=8、有效机内 16/128→1、
单层 MoE 参数量 ÷EP 后 ≈ 1,888.3 MB（1.8 GiB，每份 126 MB）、
三层流量 0 / 14 / 1 份（占比 0% / 93.3% / 6.7%）、算法带宽 171.43 GiB/s（×0.8 利用率）、通信耗时 10.757 ms、
瓶颈层=超节点内（14 份 · 200 GiB/s）。

## 配置管理

右侧面板顶部"💾 配置管理"支持：

- **预设选择**：下拉选择预设（当前含 `DeepSeekV4 10T(预估)`），选中即应用参数；
  手动修改任意输入后自动切回"自定义配置"。
- **导出配置**：将当前表单值导出为 JSON 文件下载。
- **文件导入**：选择本地 JSON 文件导入（支持嵌套 `{model, parallel, network, options}` 与扁平两种格式）。
- **粘贴导入**：点击"📋 粘贴导入"在弹窗中直接粘贴 JSON 文本后导入（与文件导入同一套校验/归一化逻辑）。

导出 JSON 格式示例：

```json
{
  "name": "DeepSeekV4 10T(预估)",
  "version": 1,
  "model":  { "attnB": 0.301, "expertB": 0.066, "numExperts": 2048, "layers": 61, "seqLen": 2048, "hiddenSize": 7168, "moeHidden": 3072, "topk": 30, "activationGB": 5.21 },
  "parallel": { "fsdp": 2048, "ep": 128, "efsdp": 16, "cp": 1, "hbmGB": 96 },
  "network": { "intraBW": 100, "intraSize": 16, "superBW": 200, "superSize": 1024, "interBW": 50, "fsdpDtype": 2, "util": 0.8 },
  "options": { "recompute": true, "swapOptimizer": false, "swapActivation": false }
}
```

内置预设 DeepSeekV4 10T(预估)：单层 Attention 0.301B、单专家 0.066B、专家 2048、层数 61、
序列长度 2048、Hidden Size 7168、MoE Hidden Size 3072、TopK 30、单层动态激活 5.21GB、
FSDP 2048 / EP 128 / EFSDP 16 / CP 1、Recompute=True、SwapOptimizer=False、SwapActivation=False、单卡 HBM 96GB
（总参数 ≈ 8.26T，卡数 2048）；网络：机内 100GB/s×16、超节点 200GB/s×1024、
超节点间 50GB/s、FSDP dtype=BF16（默认，下拉可按 FP32/FP8 切换）。页面初始示例即此预设参数。
预设配置抽离在 `presets.json`（主仿真器）与 `dsv4_presets.json`（DeepSeekV4 结构页），
页面加载时 fetch 覆盖内置 fallback 并重建预设下拉；修改预设只需改 JSON。
Layer Input 按 BF16（2 字节/元素）折算，EP Alltoall / FSDP Allgather dtype 亦为 BF16
（`ACTIVATION_BYTES` / `MOE_DTYPE_BYTES`，FSDP 载荷字节数由输入 `fsdpDtype` 调节）。

## 后续规划
- 更多通信模式（allreduce 全归约 / 分层 ReduceScatter 细化）
- SwapOptimizer 显存交换策略建模
- TopK 对动态激活的影响
- 精度扩展（BF16/FP8）与更多预设