# 第 15 周：Python 样例

README「CUDA Python samples」：代码在 `python/`，顶层 CMake 不编译它们。运行方式是样例目录里 `pip install -r requirements.txt`，再用 Python 3.10+ 执行脚本。依赖钉在各目录 `requirements.txt`，目标是 CUDA 13.x 的 `cuda.core`。

公共帮助在 `python/Utilities/`。先看它被谁导入，不要单独当成应用。

如果本机 Python 环境与 CUDA 13 不匹配，该天改为走读脚本，把 `import` 失败原文记下来，不要改样例去迁就旧版本。

## 周一：和 C++ 入门对齐

**路径**

- `python/1_GettingStarted/vectorAdd`
- `deviceQuery`
- `systemInfo`
- `simplePrint`

**做**

- 各读 README 和脚本。跑通 `vectorAdd` 与 `deviceQuery`
- 对照 `cpp/0_Introduction/vectorAdd` 和 `cpp/1_Utilities/deviceQuery`：设备对象、内存分配、launch、同步在 `cuda.core` 里叫什么

**想清楚**

1. Python 样例的 kernel 是字符串/源码交给运行时编译，还是调用已经写好的库？
2. `simplePrint` 对应哪一个 C++ 样例？
3. `systemInfo` 比 `deviceQuery` 多打印了哪一类信息？

**完成标准**：一张 API 对照，左右两列是 C++ Runtime 和 `cuda.core`。

## 周二：数组、图像、分析工具

**路径**：`numpyVsCupy`、`blurImageUnifiedMemory`、`copyImageArraytoGPU`、`kernelNsysProfile`

**做**

- `numpyVsCupy`：同一计算在 NumPy 和 CuPy 的写法差。确认样例没有偷偷把结果留在 GPU 上还不同步
- 两个图像样例：统一内存模糊，以及把图像数组拷上设备。对照第 3 周统一内存和第 4 周纹理，看 Python 版还有没有纹理对象
- `kernelNsysProfile`：README 里的 nsys 命令原样抄到笔记。本周不要求把 nsys 用熟，但要知道它包在进程外面，不是 Python API

**想清楚**

1. CuPy 和 `cuda.core` 在仓库里是什么关系（README 说以 cuda.core 为主，CuPy 是互操作或对照）？
2. 图像样例的像素布局是 HWC 还是 CHW？
3. 性能数字若没有 nsys/ncu，你至少还能用什么（CUDA event，第 3 周）？

**完成标准**：四个脚本的运行命令写在笔记里，包括你没跑成的那个。

## 周三：把第 5 周的原语用 Python 再写一遍

**路径**：`python/2_CoreConcepts/` 下的 `reduction`、`parallelReduction`、`reductionMultiBlockCG`、`parallelHistogram`、`prefixSum`、`blockwiseSum`、`binarySearch`、`matrixMulSharedMem`

**做**

- 不要八份都精读。精读 `reduction` 和 `prefixSum`
- `matrixMulSharedMem` 对照 `cpp/0_Introduction/matrixMul` 的共享内存 tile
- 其余对照 C++ 同名或同算法样例，各写一句「Python 版少了什么样板代码」

**想清楚**

1. 归约的优化阶梯在 Python 样例里是否还在，还是只保留最终版本？
2. 前缀和的块间进位在脚本的哪一段？
3. 二分搜索是第 10 周 CUB `LowerBound` 的教学版。数据规模小的时候，启动开销会不会盖过收益？

**完成标准**：点出三对 C++/Python 对应目录。

## 周四：图、流、内存资源、编译缓存

**路径**：`cudaGraphs`、`streamingCopyComputeOverlap`、`simpleZeroCopy`、`memoryResources`、`greenContext`、`jitLtoLinking`、`persistentProgramCache`、`tmaTensorMap`、`launchConfigTuning`、`processCheckpoint`、`cudaComputeLambdas`

**做**

- 精读两份：`streamingCopyComputeOverlap`（第 3 周的流）和 `cudaGraphs`（第 8 周的图）
- `jitLtoLinking` 对照第 10 周 `jitLto`
- `greenContext` 对照第 9 周 `localityDomains`
- `tmaTensorMap` 只要求知道它在配置张量内存加速器的映射，走读即可
- 其余脚本读 README 的第一段，归类到「内存 / 编译 / 占用」之一

**想清楚**

1. Python 里的流重叠还有没有锁页内存这一前提？
2. 程序缓存解决的是每次进程启动都 NVRTC 一次的成本。
3. 这些脚本哪些必须多 GPU 或新架构？沿用第 1 周的能力表。

**完成标准**：每个打开过的目录在笔记里有「对应 C++ 样例或无对应」一栏。

## 周五：框架互操作与多 GPU

**路径**

- `python/3_FrameworkInterop/customPyTorchKernel`
- `customTensorFlowKernel`
- `python/4_DistributedComputing/simpleP2P`
- `multiGPUGradientAverage`
- `ipcMemoryPool`

**做**

- 框架样例：自定义 kernel 如何拿到框架已经分配的张量存储。没装 PyTorch/TensorFlow 就走读指针获取处
- `simpleP2P` 对照 `cpp/0_Introduction/simpleP2P`
- `ipcMemoryPool` 对照第 7 周 `streamOrderedAllocationIPC`
- `multiGPUGradientAverage`：看梯度在哪次同步被平均，这是数据并行训练的最小骨架，不是完整训练循环

**想清楚**

1. 互操作的危险点是生命周期：框架释放张量之后，CUDA 指针是否还有效？
2. IPC 内存池跨的是进程。Python 的 `multiprocessing` 和 C++ `fork` 在 Windows 上行为不同，样例 README 若限定了系统，以 README 为准。
3. 整个 Python 册是在重复 C++ 册的主题。你应能用 C++ 目录名索引它们，而不是当成第二套知识。

**完成标准**：一张「Python 目录 → 已学 C++ 目录」映射表，覆盖本周精读过的脚本。
