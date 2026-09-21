# 第 9 周：Tensor Core 与高级内存

清完 `cpp/3_CUDA_Features/`。Tensor Core 样例很多，但模式重复：只精读一份 WMMA GEMM，其余做数据类型对照。

## 周一：WMMA GEMM（精读）

**路径**：`cudaTensorCoreGemm`

**做**

- 读 README：Volta 引入的 Tensor Core、`wmma` API、以及 `cudaFuncAttributeMaxDynamicSharedMemorySize` 用来提高动态共享内存上限
- 在源码里找到 `fragment`、`load_matrix_sync`、`mma_sync`、`store_matrix_sync`
- 看清一个 warp 共同完成一块小矩阵乘，不是一个线程做完整 GEMM
- 计算能力低于 7.0 时走读，不要和编译失败死磕

**想清楚**

1. WMMA fragment 分布在 warp 的寄存器里，单个线程看不到完整小矩阵。
2. 动态共享内存上限这个属性，和 Tensor Core 本身是两件独立的事，样例把它们放在一起。
3. 结果为什么仍要和 cuBLAS 或 CPU 做容差比较？

**完成标准**：能说出 load / mma / store 三步，以及参与者是 warp。

## 周二：其余精度的 WMMA（对照，不必全编译）

**路径**：`immaTensorCoreGemm`、`tf32TensorCoreGemm`、`bf16TensorCoreGemm`、`dmmaTensorCoreGemm`

**做**

- 各读 README 第一段，填表：数据类型、架构代际（Volta / Ampere）、是否使用异步拷贝把全局内存送进共享内存
- 精读其中一份带 pipeline 异步拷贝的（建议 `tf32TensorCoreGemm` 或 `bf16TensorCoreGemm`）里 `cuda::pipeline` 或等价接口出现的位置
- `dmmaTensorCoreGemm` 额外标出：协作组上的异步拷贝接口

**想清楚**

1. TF32、BF16、FP16、INT8、FP64 在样例里谁是输入、谁是累加器？
2. 异步拷贝相对第 2 周矩阵乘的同步加载，重叠的是什么？
3. 为什么同一套 WMMA 调用换数据类型就要换一套架构门槛？

**完成标准**：五份 Tensor Core 样例的对照表（含周一）完成。只要求一份能在本机跑通；其余允许走读。

## 周三：全局到共享的异步拷贝（精读）

**路径**：`globalToShmemAsyncCopy`

**做**

- 这是矩阵乘，但是用 `cp.async` 或 CUDA pipeline 把全局内存拷进共享内存
- 找到 arrive/wait barrier 在哪里替代或补充 `__syncthreads()`
- 和周一的「属性只是加大共享内存」区分开：这一份的重点是拷贝异步化

**想清楚**

1. 异步拷贝完成前就去读共享内存，会读到什么？
2. 计算能力 8.0 以下为什么样例要退回同步路径或直接放弃？
3. 它和第 6 周卷积的 halo 加载能否用同一套异步拷贝改写（只思考，不要求改代码）？

**完成标准**：能指出等待点，并说明等待的是拷贝完成而不是所有线程到达。

## 周四：图之外的内存新接口（对照）

**路径**：`memMapIPCDrv`、`cudaCompressibleMemory`、`localityDomains`、`localityDomainsDrv`、`dmabufInterop`

**做**

- `memMapIPCDrv`：用 `cuMemMap` 做跨进程共享。对照第 2 周 `vectorAddMMAP` 和第 4 周 `simpleIPC`
- `cudaCompressibleMemory`：可压缩内存的分配属性。看它如何通过驱动查询支持情况
- `localityDomains` 与 `localityDomainsDrv`：Runtime / Driver 两套「局部化 green context + 局部内存池」。两份对看，只记 API 前缀差异
- `dmabufInterop`：Linux dma-buf 文件描述符的导出和导入，含 `fork` 和跨进程传递 fd。Windows 上走读 README 的三个演示（同进程、跨进程、跨 GPU）

**想清楚**

1. `cuMemMap` 家族反复出现，是因为「虚拟地址连续、物理属性可指定」。
2. Green context 和默认主上下文比，缩小的是什么可见范围？
3. dma-buf 和 CUDA IPC 都在跨进程共享显存，句柄形态为什么不同（fd vs CUDA IPC handle）？

**完成标准**：一张「句柄类型」表：设备指针、IPC handle、cuMemMap handle、dma-buf fd。

## 周五：特性册里剩下的短样例

**路径**：`bindlessTexture`、`newdelete`、`ptxjit`、`StreamPriorities`

**做**

- `bindlessTexture`：纹理对象 / 表面对象 / mipmap，对照第 4 周的经典纹理绑定
- `newdelete`：设备侧 `new`/`delete` 和虚函数。看分配发生在设备堆，而不是共享内存
- `ptxjit`：Driver API 从 PTX 做 JIT，并用 `cuLink*` 做运行时链接。和第 2 周 `matrixMulDynlinkJIT`、`vectorAdd_nvrtc` 放在同一张「编译路径」笔记里
- `StreamPriorities`：创建不同优先级的流。优先级只影响同设备上准备执行的工作时谁先得到 SM，不改变单条流内的顺序

**想清楚**

1. 设备 `new` 失败时样例如何表现？
2. 流优先级是暗示还是硬实时保证？
3. 特性册到这里，哪些 API 你只在「有第二块 GPU / Linux / SM 8+」时才真的跑过？把缺口写进周六总结。

**完成标准**：`3_CUDA_Features` 每个子目录都在笔记里有一行（跑过 / 走读 / 跳过及原因）。
