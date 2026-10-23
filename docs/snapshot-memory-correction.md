# FSDP 内存口径与快照校准（2026-10-23）

本次修正以 `fbfa6a3984a724a4b9e748f78ccfa312044a4df2` 为基线。实测数据见 [JSON](../data/memory-snapshot-20261023.json)。数值只适用于下述冻结配置，不是通用模型常数。

## 实验与证据范围

Ascend 950DT 八卡，FSDP8 / EP2 / EFSDP4 / CP1，31 个主层加 1 个 MTP 层、6 个路由专家、序列长度 4096、micro batch 1、gradient accumulation 1，UAM、Muon 和 MegaMoE redispatch 开启，常规 profiler 和 redispatch VERIFY 关闭。BF16 和 Full MXFP8 分别运行独立的 15 步进程；所有 16 个 rank 收据均 PASS，loss/grad norm 有限，完成后两次资源检查均空闲。这不是先前 40 步 ABBA 性能实验，采集耗时不用于性能结论。

冻结源码：

| 仓库 | commit |
|---|---|
| MindSpeed-LLM | `4cf1c7ddb5e7ed2aece7cf9bbd93fe4210af3f34` |
| FSDPTurbo | `630f391efdbf7f85fe0fc4c9add63b484722d562` |
| ascend_deepep_hkq | `97919718d2151dd972956c80b3b114ae25126fba` |
| SHMEM | `8f4b4791b06e8f55fb6d2ce00e1d3c2f100c344f` |

初始化前开启 allocator state 与 Python 栈，完成第 11 步并同步后启用完整事件记录，保存起点；完成第 12 步前向、反向、优化器更新和设备同步后保存终点，再关闭记录。`context='alloc'`、`stacks='python'`、`max_entries=200000`。原始快照采用 `pickle(memory._snapshot(), protocol=4)`，与 `_dump_snapshot` 的快照结构一致，避免该环境 `_dump_snapshot` 的额外 profiler fallback。

四份原始文件合计 **21,746,634 bytes**。BF16 和 MXFP8 终点分别有 82,832 和 150,992 个 rank0 事件；起点前缀 18 个事件完整保留，均小于容量，没有环形历史覆盖。其他 rank 的稳态 allocated 峰值未超过 rank0 + max(512 MiB, 5%)，因此不补采。原始 pickle 包含路径和存储地址，仅保留在私有交付目录，不放入本仓库。仅加载自己信任的 pickle。

## 口径及重建

| rank0 第 12 步 | BF16 GiB | Full MXFP8 GiB |
|---|---:|---:|
| allocated 峰值 | 48.148 | 48.454 |
| active 峰值 | 51.117 | 51.809 |
| allocator reserved 峰值 | 69.957 | 71.670 |
| 整段运行物理 HBM 采样最大值 | 78.505 | 79.487 |

精确字节数以 JSON 为准。物理 HBM 约每 10 秒通过 npu-smi 采样，包含整卡开销；它既不是连续峰值，也不与上表其他峰值同一时刻。不能相减后把差值全部叫作碎片或 SHMEM。

重放以起点仍存活的 storage 为初态，按地址和分配世代跟踪 alloc / free_requested / free_completed。`free_requested` 降低 allocated，只有 `free_completed` 才降低 active；地址释放后再使用属于新分配。参数别名以 `(device, storage address)` 去重，不按 tensor 对象数量累加。本环境分配按 512 字节对齐，重放终态逐块地址和大小以及 allocated/active 峰值全部与 allocator 统计吻合，误差 **0 bytes**。

快照还附加了 6 个 workspace_snapshot 段：BF16 750,780,416 bytes，MXFP8 549,453,824 bytes。直接把全部 segments.total_size 加起来会高估 PTA reserved；扣除这些显式工作区后，与 allocator reserved 统计误差为零。这些段不能再次算入 allocator allocated，也不应视为所有 SHMEM/HCCL 内存的完备清单。

源码与八 rank 启动日志共同确认 SHMEM window_pool=1：BF16 每窗口 706,740,224 bytes，配置对称堆 1,413,480,448 bytes；MXFP8 每窗口 383,778,816 bytes，配置对称堆 767,557,632 bytes。对应 `dispatcher_mega_moe.py::_ensure_mega_moe_runtime` 和 `dispatcher_mega_moe_mxfp8.py` 的 `(pool+1)*window`、2 MiB 对齐逻辑，native `ShmemInstance` / `TrainW8A8Runtime` 持有并释放堆。这是已核对的配置量，不是额外测出的物理 resident 字节。不能把对端映射乘 EP 作为新增本地物理分配。HCCL、runtime 和驱动占用尚无独立分项计数，保留未知，不从不同时间的峰值差额反推。

两份终点 pickle 均通过本机安装的 MindStudio Insight 官方 dump2db 解析，并在 MemScope 界面实际打开，时间轴分别覆盖 0…82831 和 0…150991 事件。界面事件图与 allocator 分段统计覆盖口径不同，报告中的定量峰值使用起点状态加完整事件重建。

调用栈分类互斥但不等于精确的张量语义。模型激活、算子输出和工作区有些共用栈；Muon 状态和临时计算也可能同栈。因此保留 `*_or_workspace` 名称和 `unattributed`，未归因量在 allocated 峰值为 BF16 3,120,110,592 bytes、MXFP8 2,547,588,096 bytes。UAM 类别只表示由对应恢复栈分配的缓冲，不代表 UAM 的所有存活激活。梯度相关栈也未全部等同于持久梯度分片。

## 公式修正

1. **存储、计算、通信精度分开。** 实测参数均为 FP32，唯一存储 13,647,800,796 bytes，allocator 对齐后 13,647,906,816 bytes。两组均 BF16 AllGather、FP32 ReduceScatter；Full MXFP8 改变支持算子的计算及量化副本，不自动把参数存储或 AllGather 降成一字节。页面分别提供 paramBytes、gradBytes、AG、RS 和计算精度。FP8 通信估算尚不含 scale 元数据。
2. **交换后不是自动零。** 优化器相关独立峰值达到 14,211,668,992 bytes（约 13.24 GiB），尽管 CPU swap 已开启。UAM 恢复和预取也存在驻留窗口。页面新增交换后驻留/工作区输入；零表示未建模，不能作为“已证明零显存”的结论。
3. **峰值必须按生命周期。** 同一 allocated 峰值的互斥分类可以相加，分类各自独立峰值不能相加冒充实测峰值。例如优化器独立峰值约 13.24 GiB，但在整步 allocated 最高时该栈类别只有 512 bytes。主计算器因此称为“分项容量估算”，实测表另外展示。容量项有别名或生命周期重叠时，先去重并按时间重建；不要加入无来源常数去拟合总量。
4. **Layer Input 包含 MBS 和 CP。** 使用 `seq × microBatch × hidden × layers / CP × activationBytes`；DeepSeek4 同步桥接携带 MBS。用户输入的动态激活本身已针对当前 MBS，不再重复乘 MBS。

`snapshot-bf16` / `snapshot-mxfp8` 预设固定并行和精度；参数量来自全部本地参数 inventory，非路由参数平均分摊到 32 层（含 embedding、shared experts 和 MTP），只用于总容量核对，不表示每层实际一样大。预设中未知激活/驻留项为 0 并明确标注待校准；旁边的快照表才是实测值。算子工作区、量化 scale、HCCL、SHMEM、设备基线以及碎片都不允许用一个推测常数合并。

## 配置兼容与验证

导出版本 2，新增 `network.allGatherDtype` / `reduceScatterDtype` 和 `memory` 字段。旧版仅有 `fsdpDtype` 时仍映射到两个通信精度；新字段优先。保留旧字段供旧读者使用。主页面内置 fallback 与 presets.json 同步维护。

```sh
node tests/verify_mem.js
node tests/verify_dsv4.js
node tests/verify_sync.js
python tests/verify_snapshot.py
```

前三组覆盖页面公式、精度独立、交换驻留、v1/v2 配置往返、两组预设和 DeepSeek4 桥接。快照测试覆盖地址复用、异步释放、参数别名、独立分类峰值不相加、以及两组公开校准数据的自洽性。完整原始快照重放脚本在 `tools/replay_snapshot.py`，其 allocator 对齐假设仅针对本次冻结环境。

参考：[MindStudio Insight 内存调优指南](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/memory_tuning.md)。本 PR 只更新源码；线上站点须维护者合并部署后更新。

任务本地采集入口为 `tools/capture_rank.py`（通过 TRAIN_ENTRYPOINT 指定），依赖对应冻结的 train_fsdp2.py。环境变量 REPO_ROOT、BENCH_ROOT、RUN_DIR、BENCH_ARM、RANK 由启动器提供；SNAPSHOT_RANK 和 SNAPSHOT_MAX_ENTRIES 可配置。该脚本不负责登录、抢占设备或重建环境。
