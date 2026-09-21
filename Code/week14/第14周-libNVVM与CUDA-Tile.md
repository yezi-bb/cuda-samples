# 第 14 周：libNVVM 与 CUDA Tile

两册都短，但和 nvcc 写 kernel 不是同一条工具链。前三天 `cpp/7_libNVVM/`，后两天 `cpp/9_CUDA_Tile/`。

## 周一：NVVM 最小闭环（精读 simple）

**路径**：`cpp/7_libNVVM/README.md`、`simple`、`ptxgen`

**做**

- 读分类 README 的构建说明：`CUDA_HOME`、`LIBNVVM_HOME`，Windows 可用 `utils/build.bat` 或并进顶层 CMake（README 写 Windows 往往要单独编）
- `simple`：从文件读 NVVM IR，编成 PTX，用 Driver API 启动
- `ptxgen`：链上 libdevice、按 NVVM 规范做验证、再出 PTX。`test_args.json` 里的参数示例是 `test.ll` 和 `-arch=compute_75`，以当前文件为准核对架构是否适合你的 GPU
- 跑通一份即可

**想清楚**

1. NVVM IR 不是 PTX。中间还有 libNVVM 这一跳。
2. 和第 2 周 NVRTC 相比，输入从 CUDA C 换成了 LLVM IR。
3. libdevice 提供的是设备端数学库，不是驱动程序。

**完成标准**：能按顺序说出 simple 样例的三步：读 IR、编译出 PTX、Driver 启动。

## 周二：IR 示例目录（走读加能跑则跑）

**路径**：`cuda-shared-memory`、`syscalls`、`uvmlite`、`device-side-launch`

**做**

- `cuda-shared-memory`：IR 里共享内存怎么声明。对照第 2 周矩阵乘的 `__shared__`
- `syscalls`：设备端 malloc / free / vprintf 在 IR 层长什么样。对照第 9 周 `newdelete` 和 `simplePrintf`
- `uvmlite`：统一虚拟内存的最小 IR 程序。对照 `cudaMallocManaged`
- `device-side-launch`：在 IR 层做动态并行。对照第 8 周 `cdpSimplePrint`

**想清楚**

1. 这些样例不是新功能，而是同一功能的更低一层写法。
2. 设备端 printf 最终会进到哪类系统调用，主机何时能看到输出？
3. 你是否需要会写 IR？本仓库的目标是读懂样例如何驱动 libNVVM，不是成为 IR 作者。

**完成标准**：四个目录各注明「对应前面哪一个 CUDA C 样例」。

## 周三：cuda-c-linking（有合适 LLVM 才编译）

**路径**：`cuda-c-linking`

**做**

- 读 README 关于 LLVM 版本的限制：LLVM 15+ 的 opaque pointer 只有 Blackwell 及更新架构的 libNVVM 接受；要在更老的 GPU 上跑，用 LLVM 7 到 14
- 样例做的事：用 LLVM API 生成 IR，和 nvcc 产出的 PTX 链接，再启动
- 本机没有匹配的 LLVM 开发包时，走读 `CMakeLists.txt` 里的 `ENABLE_CUDA_C_LINKING_SAMPLE`，不要为了这一天单独编一套 LLVM

**想清楚**

1. 链接的两端语言不同：一边是 LLVM 生成的 NVVM，一边是 nvcc 的 PTX。
2. 这和第 10 周 `jitLto` 的差别：一个偏 LLVM/libNVVM，一个偏 NVRTC/nvJitLink。
3. 版本不匹配时样例选择退出码 2（waive）而不是生成坏代码。这和 `EXIT_WAIVED` 的设计一致。

**完成标准**：在编译路径总表上把 libNVVM 标成一条独立的列，并写上你本机是否真的跑过。

## 周四：Tile 入门（精读 tileVectorAdd）

**路径**：`cpp/9_CUDA_Tile/README.md`、`helloTile`、`tileVectorAdd`、`tileTranspose`

**做**

- `helloTile`：tile kernel 如何启动，SIMT kernel 和 tile kernel 如何经全局内存交接
- 精读 `tileVectorAdd`：`partition_view` 把数据切成 1024 一片，掩码加载处理最后一片越界
- `tileTranspose`：每块拿一个 n×m 小块，块内转置再写回。和第 13 周 SIMT 转置对比「谁在写共享内存 bank」

**想清楚**

1. Tile 编程不是再写一遍 `__global__` 加下标。数据单位是块视图。
2. 掩码加载对应第 2 周 vectorAdd 里的边界判断，但写法不同。
3. SIMT 与 Tile 混用时，交接面为什么是全局内存而不是共享内存？

**完成标准**：能说明 `partition_view` 在向量加和转置里各切哪一维。

## 周五：Tile 上的矩阵与一层网络算子

**路径**：`tileMatmul`、`tileMatmulAutotuner`、`tileBmm`、`tileLayerNorm`、`tileRope`、`tileSpMV`

**做**

- 精读 `tileMatmul`：FP16 乘、FP32 累加，`cuda::tiles::mma`。样例对比朴素实现和带编译提示的实现，并用事件计时。这是第 9 周 WMMA 的 Tile 对应物
- `tileMatmulAutotuner`：在 tile 尺寸和提示上做自动搜索。看搜索空间，不必等它搜完所有组合
- 其余四份各读 README 的形状约定：
  - `tileBmm`：静态持久化的批量 GEMM，块数约等于 SM 数
  - `tileLayerNorm`：持久化块沿行做归一化
  - `tileRope`：旋转位置编码，按 GPT-NeoX 的对半拆分约定
  - `tileSpMV`：稀疏矩阵乘向量

**想清楚**

1. 「持久化块 + grid-stride」和第 5 周多块协作归约都在减少启动次数，但同步方式不同。
2. Autotuner 搜的是编译期 tile 参数，不是运行时的 grid。
3. 到本周末，同一矩阵乘你已经见过：教学 SIMT、cuBLAS、WMMA、Tile。各用一个目录名记住。

**完成标准**：四条矩阵乘路径写在同一页笔记上。Tile 样例至少跑通 `helloTile` 或 `tileVectorAdd` 之一；其余允许走读。
