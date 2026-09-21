# 第 2 周：入门 Runtime，最小内核

对应 `cpp/0_Introduction/`。本周把「分配、拷贝、启动、同步、校验」做成肌肉记忆。Programming Guide 里 vectorAdd 对应第 3 章，matrixMul 对应第 6 章，样例 README 写了这一点。

## 周一：vectorAdd（精读）

**路径**：`cpp/0_Introduction/vectorAdd`

**读**：样例 README，然后整份 `.cu`。

**做**

- 单样例构建并运行，确认有正确性对比输出
- 标出 grid 和 block 怎么从 `N` 算出来，最后一个 block 如何避免越界
- 对照 `Common/helper_cuda.h` 的错误检查包在哪些调用上

**想清楚**

1. `cudaMalloc` / `cudaMemcpy` / kernel / `cudaFree` 哪几步是同步的？
2. 元素个数不是 block 大小整数倍时，谁负责边界？
3. 这个样例故意没做哪些优化（共享内存、向量化加载）？

**完成标准**：能在纸上写出 launch 配置，并说明 host 侧如何判定结果正确。

## 周二：同一加法的另外三种启动（对照）

**路径**

- `vectorAddDrv`：Driver API 启动
- `vectorAdd_nvrtc`：运行时把源码编译成 PTX/CUBIN 再启动
- `vectorAddMMAP`：用 `cuMemMap` 换掉普通设备分配，访问方式仍然连续

**做**：三份都打开。精读 `vectorAddDrv` 的模块加载和 `cuLaunchKernel`；另外两份只追「和周一相比多出来的 API」。能编则运行 `vectorAddDrv`。

**想清楚**

1. Runtime API 的 `<<<>>>` 最终落到哪些 Driver 调用？
2. NVRTC 适合什么场景，它为什么不替代 nvcc 离线编译？
3. `cuMemMap` 改变的是物理属性还是 kernel 里的指针算法？

**完成标准**：一张三列对照表：谁编译 kernel、谁分配内存、kernel 函数体是否同一份。

## 周三：设备端调试与计时（精读 clock，对照其余）

**路径**：`simplePrintf`、`simpleAssert`、`clock`

**做**

- 运行 `simplePrintf`，确认设备端 `printf` 在同步之后才刷出来
- 读 `simpleAssert`：断言失败如何把设备停住，计算能力要求是什么
- 精读 `clock`：用 `clock()` 或 `clock64()` 量的是一个 block 内的周期，不是整次 launch 的墙钟时间

**想清楚**

1. 为什么设备 `printf` 不能用来判断 kernel 已经结束？
2. `clock` 样例为什么强调「一个 block」，而不是 GPU 总耗时？
3. 断言样例和主机侧 `checkCudaErrors` 各抓哪一类错误？

**完成标准**：能说出三种观察手段：主机错误码、设备 printf、块内时钟。

## 周四：matrixMul（精读）

**路径**：`cpp/0_Introduction/matrixMul`

**读**：README 对「为讲清楚，不是最快 GEMM」的声明；host 文件和 `matrixMul_kernel.cu`。

**做**

- 构建运行。记下 tile 边长、共享内存用量、每个线程算几个输出元素
- 找到样例后半段如何调用 cuBLAS 做对照，分清「教学 kernel」和「库 kernel」

**想清楚**

1. 为什么要先把子块搬进共享内存，而不是每个乘累加都读全局内存？
2. `__syncthreads()` 在这个 kernel 里保护的是哪一次数据复用？
3. 结果校验的容差来自哪里，为什么浮点 GEMM 不用逐位相等？

**完成标准**：能画出一个 tile 从全局内存到共享内存再到寄存器的路径。

## 周五：矩阵乘的 Driver 与 JIT 版本（对照）

**路径**：`matrixMulDrv`、`matrixMulDynlinkJIT`

**做**

- `matrixMulDrv` 对照周四，只看启动方式
- `matrixMulDynlinkJIT` 看运行时加载 CUDA Driver、从 PTX 做 JIT 的步骤
- 不要在这个样例上调性能。README 已说明高性能路径是 cuBLAS

**想清楚**

1. 动态链接 Driver 解决的是部署问题还是算法问题？
2. PTX JIT 和周二的 NVRTC 输入各是什么（PTX 文本 vs CUDA C 源码）？
3. 三个矩阵乘样例的 kernel 数学是否相同？

**完成标准**：用五句话讲完「同一矩阵乘，三种把代码送上 GPU 的办法」。周六总结时把这五句话写进笔记。
