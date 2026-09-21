# CUDA Samples 学习计划

依据仓库根目录 [README.md](../README.md)（CUDA Toolkit 13.4）制定。目标不是把每个可执行文件都点一遍，而是顺着官方分类，把构建系统、公共代码、Runtime/Driver API、算法、特性、库、领域样例、性能、NVVM、Tile 和 Python 绑定串成一条能复述的路径。

当前机器是 Windows。计划里的构建命令按 README 的 Windows 小节来写。Tegra、QNX、EGL、dma-buf 等样例在本机通常编不过或跑不起来，这些天改成源码走读，不要卡在环境上。

## 周期

| 项 | 安排 |
| --- | --- |
| 总长度 | 16 周 |
| 每个学习日 | 约 2–3 小时 |
| 节奏 | 周一到周五做当天任务，周六用 1 小时补漏并写本周三句话总结，周日休息 |
| 深度 | 精读 / 对照 / 走读，见下文 |

不要一上来 `cmake` 全量编译。README 写明默认会为该版本支持的全部 GPU 架构编译，耗时很长。先用单样例构建，架构号在第 1 周第 2 天确定。

## 每日固定动作

1. 打开该样例目录的 `README.md`，记下依赖、最低计算能力、支持的操作系统。
2. 精读日：单样例构建并运行，从 `main` 追到 kernel launch，标出分配、拷贝、启动、同步、校验五步。
3. 对照日：不从头读，只标出和已精读样例的 API 差异。
4. 走读日：依赖或平台不满足时，读 `README.md`、`CMakeLists.txt` 和关键 `.cu`/`.cpp`，在笔记里写明阻塞原因。
5. 用 [笔记模板.md](./笔记模板.md) 记一页。当天三个问题答不上来，这一天不算完成。

单样例构建（在样例目录内，把 `<arch>` 换成第 1 周记下的 SM 版本，例如 `89`）：

```bat
mkdir build
cd build
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=<arch> ..
cmake --build . --config Release
```

生成器名称必须和本机 Visual Studio 一致。README 的示例是 `Visual Studio 16 2019`；装的是 VS 2022 就用上面的 `Visual Studio 17 2022`。也可以用 VS 2019 16.5+ 直接打开样例目录，不必先生成解决方案。

## 深度标记

- **精读**：构建、运行、把 host/device 数据流画出来。
- **对照**：和本周已经精读的样例比差异，只读增量代码。
- **走读**：本机缺少显示、MPI、Vulkan、Tegra 或第二块 GPU 时采用。读懂调用关系即可。

样例 README 里写了 `EXIT_WAIVED`（退出码 2）时，表示样例主动放弃运行（缺依赖或设备不满足），不要当成自己写错了。

## 周次索引

| 周 | 主题 | 对应 README |
| --- | --- | --- |
| [第 1 周](./第01周-工程骨架与构建.md) | 仓库地图、CMake、公共头文件、测试脚本 | Getting Started、Building、Running All Samples |
| [第 2 周](./第02周-入门Runtime最小内核.md) | vectorAdd / matrixMul 家族 | [0. Introduction](../cpp/0_Introduction/README.md) |
| [第 3 周](./第03周-流事件与内存模型.md) | 流、事件、锁页、统一内存 | 0. Introduction |
| [第 4 周](./第04周-多设备纹理与DriverAPI.md) | 多 GPU、纹理、Driver API | 0. Introduction |
| [第 5 周](./第05周-设备查询与并行原语.md) | 设备查询、归约、扫描、直方图 | [1. Utilities](../cpp/1_Utilities/README.md)、[2. Concepts](../cpp/2_Concepts_and_Techniques/README.md) |
| [第 6 周](./第06周-卷积排序与蒙特卡洛.md) | 卷积、排序网络、cuRAND | 2. Concepts |
| [第 7 周](./第07周-内存池与概念收尾.md) | 流序分配、Thrust、其余概念样例 | 2. Concepts |
| [第 8 周](./第08周-Graphs与动态并行.md) | CUDA Graphs、Cooperative Groups、CDP | [3. CUDA Features](../cpp/3_CUDA_Features/README.md) |
| [第 9 周](./第09周-TensorCore与高级内存.md) | WMMA、异步拷贝、cuMemMap、Green Context | 3. CUDA Features |
| [第 10 周](./第10周-CUDA库.md) | CUB、cuFFT、cuRAND、cuSPARSE、nvJitLink | [4. CUDA Libraries](../cpp/4_CUDA_Libraries/README.md) |
| [第 11 周](./第11周-领域图像与金融.md) | 图像、变换、期权定价 | [5. Domain Specific](../cpp/5_Domain_Specific/README.md) |
| [第 12 周](./第12周-仿真与图形互操作.md) | n-body、流体、OpenGL / D3D / Vulkan | 5. Domain Specific |
| [第 13 周](./第13周-性能样例.md) | 对齐、转置、统一内存、Graph 伸缩 | [6. Performance](../cpp/6_Performance/README.md) |
| [第 14 周](./第14周-libNVVM与CUDA-Tile.md) | NVVM IR 与 Tile C++ | [7. libNVVM](../cpp/7_libNVVM/README.md)、[9. CUDA Tile](../cpp/9_CUDA_Tile/README.md) |
| [第 15 周](./第15周-Python样例.md) | cuda.core，不走顶层 CMake | [CUDA Python samples](../README.md) |
| [第 16 周](./第16周-平台样例与总复习.md) | Tegra 走读、依赖总表、自测 | [8. Platform Specific](../cpp/8_Platform_Specific/Tegra/README.md) |

## 学完时应能独立讲清

1. 一个样例从 `CMakeLists.txt` 到 `<<<grid, block, smem, stream>>>` 的完整路径。
2. Runtime API 与 Driver API 的分工，以及 NVRTC / nvJitLink / libNVVM 三条编译路径的差别。
3. 全局内存、共享内存、纹理、零拷贝、统一内存、流序内存池各自解决什么问题。
4. 归约、扫描、直方图、矩阵乘在样例里各自做了哪一层优化，库版本（cuBLAS、CUB、Tile）又把哪一层交给了库。
5. 哪些样例依赖显示、第二块 GPU、MPI、Vulkan、Tegra，以及 `run_tests.py` 为什么把它们标成 skip。
