# 第 8 周：Graphs、协作组与动态并行

进入 `cpp/3_CUDA_Features/`。本周三条主线：CUDA Graphs、协作组、从设备启动 kernel（CDP）。Tensor Core 放到第 9 周。

## 周一：协作组进阶（精读）

**路径**：`binaryPartitionCG`、`warpAggregatedAtomicsCG`

**做**

- 先用 10 分钟回顾第 4 周 `simpleCooperativeGroups`（块内组）
- `binaryPartitionCG`：按谓词把线程分成两组再归约
- `warpAggregatedAtomicsCG`：一个 warp 里先选出代表，再对全局计数器做一次原子加，而不是 32 次

**想清楚**

1. 聚合原子把原子次数从「每线程一次」降到「每组一次」，正确性靠什么把组内部分和先算完？
2. 二元划分和 `__ballot_sync` 一类投票是什么关系？
3. 这一天的协作范围仍然是块内还是网格级？

**完成标准**：能解释「warp 聚合原子」相对 `simpleAtomicIntrinsics` 快在哪里，以及它什么时候没有收益（冲突本来就不严重时）。

## 周二：CUDA Graphs 创建与启动（精读）

**路径**：`simpleCudaGraphs`

**做**

- 构建运行。样例同时演示 Graph API 手搭节点，以及流捕获（stream capture）
- 把一次捕获的起止 API 标出来：开始捕获、这段里的 memcpy/kernel、结束捕获、实例化、启动
- 对比「每次都单独 launch」和「把已实例化的图再启动」在主机侧开销上的含义，先理解，不测数字

**想清楚**

1. 捕获期间为什么不能有未完成的、会把工作漏出捕获流的操作？
2. 图实例（executable graph）和图模板（graph）是什么关系？
3. 空图或只有一次 kernel 的图，值得捕获吗？

**完成标准**：能默写捕获路径的五步，并指出手搭节点路径少了哪一步（没有「在流上录制」）。

## 周三：图里的内存节点（精读一份）

**路径**：`graphMemoryNodes`、`graphMemoryFootprint`

**做**

- `graphMemoryNodes`：在图内分配和释放，分别看显式 Graph API 和捕获两条路
- `graphMemoryFootprint`：内存节点如何复用虚拟地址和物理页。读 README 的结论，再在源码里找到复用被打印或被断言的位置

**想清楚**

1. 图内 `cudaMalloc` 捕获和周一学的 `cudaMallocAsync` 谁的生命周期绑在图的一次启动上？
2. 地址复用为什么能降低足迹，又在什么情况下会让你看到「上一轮的脏数据」？
3. 释放节点如果漏了，实例化会在什么阶段失败？

**完成标准**：能说出图内存节点相对普通 `cudaMalloc` 的两个约束：分配在图里配对、地址可能被复用。

## 周四：条件节点与 Jacobi 更新（精读 jacobi）

**路径**：`graphConditionalNodes`、`jacobiCudaGraphs`

**做**

- `graphConditionalNodes`：CUDA 12.4 起的条件节点。看条件如何决定子图是否执行，主机如何设置条件值
- `jacobiCudaGraphs`：迭代法里更新已实例化的图。对照 `cudaGraphExecKernelNodeSetParams` 和 `cudaGraphExecUpdate` 两条更新路径，样例 README 点了这两个 API

**想清楚**

1. 条件节点解决的是「主机每次 if 再决定 launch」，还是设备端分支？
2. 改 kernel 参数时，为什么优先小更新而不是整图销毁重建？
3. Jacobi 的收敛判断放在图内还是图外？

**完成标准**：能选择一种说法讲清「捕获一次，迭代很多次，偶尔改参数」。

## 周五：动态并行（精读 cdpSimplePrint 和 cdpSimpleQuicksort）

**路径**：`cdpSimplePrint`、`cdpSimpleQuicksort`、`cdpAdvancedQuicksort`、`cdpQuadtree`、`cdpBezierTessellation`

**做**

- 精读前两份。设备端 `<<<>>>` 或等价启动，子网格何时相对父线程可见
- 后三份对照：快排的分治、四叉树、贝塞尔细分，各自在什么地方「父线程决定再开一块工作」
- 确认设备计算能力 ≥ 3.5。看样例如何把设备运行时链接进来（CMake 里的 `cudadevrt` 或等价设置）

**想清楚**

1. 动态并行的收益是负载不均时少做主机往返，代价是设备侧启动开销和调试难度。
2. 子 kernel 的错误如何回到主机？
3. 为什么简单 printf 样例仍然有价值：它把「设备上启动」和算法剥离开。

**完成标准**：能画出快速排序样例里一次父启动到子启动的调用关系。高级快排、四叉树、细分只要求各记一个「工作划分」句子。
