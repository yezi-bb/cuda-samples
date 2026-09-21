# 第 10 周：CUDA 库

`cpp/4_CUDA_Libraries/`。分类 README 只列了共轭梯度、NvSci、视线、海洋 FFT、随机雾。目录里还有 CUB、cu++、nvJitLink。以目录为准，把 README 没写进索引的样例补进本周。

库样例的读法：先找「库调用」，再找「样例自己写的 kernel」。自己写的那部分才和前几周对比。

## 周一：CUB 设备级算法（精读 cubDeviceFind）

**路径**：`cubDeviceFind`、`cubDeviceSegmentedScan`、`cubDeviceTransform`

**做**

- 精读 `cubDeviceFind`：`FindIf`、`LowerBound`、`UpperBound`，主机用 `std::find_if` / `std::lower_bound` / `std::upper_bound` 对答案
- 看临时存储（temp storage）两段式：先问大小，再分配，再真正执行
- 另外两份只确认：分段扫描的段边界如何传入，transform 是否不再手写 kernel

**想清楚**

1. CUB 为什么要先查询临时空间，而不是在 API 里隐式 `cudaMalloc`？
2. 设备端二分和你在 CPU 上写的二分，并行度来自哪（多个查找同时进行，而不是把一次二分拆成 32 路）？
3. 第 5 周手写的 `scan` 和 `cubDeviceSegmentedScan` 谁负责处理段间不合并？

**完成标准**：能写出 CUB 调用的三步：问大小、分配、执行。

## 周二：cu++ 与 nvJitLink（精读 jitLto）

**路径**：`libcuxxRandom`、`libcuxxMdspan`、`jitLto`

**做**

- `jitLto`：NVRTC 产出可重定位代码，`nvJitLink` 做链接和 LTO，再启动 SAXPY。和第 9 周 `ptxjit` 对比，多出来的是「多个设备代码对象链在一起」
- `libcuxxRandom`、`libcuxxMdspan`：看 C++ 封装如何包住随机数和多维视图。各记公开类型名，不逐行读实现

**想清楚**

1. NVRTC 编译和 nvJitLink 链接是前后两步，不是同一个库。
2. LTO 发生在运行时，还是 nvcc 离线编译时？
3. `mdspan` 解决的是下标计算，不是数据搬运。

**完成标准**：编译路径笔记补上第四列：nvcc、NVRTC、Driver PTX JIT、NVRTC+nvJitLink。

## 周三：cuRAND 可视化与 Thrust 视线（能显示则运行）

**路径**：`randomFog`、`lineOfSight`

**做**

- `randomFog`：伪随机和拟随机的雾。无 OpenGL 就走读生成器初始化和内核消费随机数的位置，对照第 6 周估 π
- `lineOfSight`：高度图上一条射线的可见性，用 Thrust。把「射线步进」和「Thrust 算法名」分开记

**想清楚**

1. 随机雾样例的教学目标是分布质量，不是渲染技巧。
2. 视线算法里，一旦某个点被挡住，后面的点还要不要继续算？样例如何并行这件事？
3. 这两个样例如果去掉图形或 Thrust，核心数值还在不在？

**完成标准**：各用五句话写下数据从哪来、库做了哪一步、结果如何判定。

## 周四：cuFFT 海洋（精读计算部分）

**路径**：`oceanFFT`

**做**

- 频谱生成 → cuFFT 逆变换 → 高度场。把 `cufftPlan` 和 `cufftExec` 的调用圈出来
- OpenGL 渲染部分走读即可，除非本机显示正常且你希望看动画
- 对照第 6 周可分离卷积：那里是空间域，这里是频域。第 11 周的 `convolutionFFT2D` 会把两者接上

**想清楚**

1. 计划对象（plan）为什么可以复用？
2. 实数到复数的布局（厄米共轭）在样例里有没有被照顾到？
3. 渲染和计算之间的互操作资源，和第 4 周 `simpleCUDA2GL` 是否同一类注册？

**完成标准**：能指出哪一次是库变换、哪一次是样例自己的 kernel（生成频谱或算斜率）。

## 周五：共轭梯度三件套（精读单 GPU 图版本）

**路径**：`conjugateGradientCudaGraphs`、`conjugateGradientMultiBlockCG`、`conjugateGradientMultiDeviceCG`

**做**

- 精读第一份：cuBLAS + cuSPARSE 的稀疏迭代，被 CUDA Graph 捕获。这是第 8 周图的「库调用也能进图」
- 第二份：多块协作组 + 统一内存。对照第 5 周 `reductionMultiBlockCG`
- 第三份：多 GPU 协作组，统一内存的预取和访问建议（prefetch / advise）。没有多块 GPU 或计算能力不够则走读

**想清楚**

1. 稀疏矩阵在样例里是 CSR 还是别的格式，向量存在统一内存还是设备内存？
2. 图捕获的是一次迭代还是整个求解？
3. `cudaNvSci` 在本目录，但依赖 NvSci，留到第 16 周和 Tegra 一起走读，本周不要在它上面耗一天。

**完成标准**：能讲清三份共轭梯度分别在「图、单机多块协作、多 GPU」上多做了什么，数学迭代式不要求推导。
