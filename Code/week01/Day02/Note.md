**读**

- README「Windows」和「Building a Single Sample」
- 根 `CMakeLists.txt`：`cmake_minimum_required(3.20)`、语言 `C CXX CUDA`、`CMAKE_CUDA_ARCHITECTURES` 的默认逻辑
- `cmake/CudaSampleArchs.cmake` 里的主架构列表，知道默认会编哪些 SM

**做**

- 确认本机 `nvcc`、`cmake`、Visual Studio 生成器名称
- 只构建 `cpp/1_Utilities/deviceQuery`（Utilities 最适合当探针）。架构先用 `native` 试一次；若生成器不接受 `native`，改成设备的计算能力数字（`deviceQuery` 输出里的 major.minor，例如 8.9 写成 `89`）
- 把成功的生成器名和 `<arch>` 写进笔记，后面 15 周都用这一对

**想清楚**

1. 为什么单独进样例目录配置时，必须自己传 `CMAKE_CUDA_ARCHITECTURES`？
2. 全量 `cmake ..` 不传架构时，时间会花在哪里？
3. `ENABLE_CUDA_DEBUG` 打开后，编译器少做了什么，为什么默认关闭？

**完成标准**：`deviceQuery` 的 Release 可执行文件能打印本机 GPU 名称和计算能力。



1. 打开 **x64 Native Tools Command Prompt for VS 2022**（独立命令行，不是 VS 内置终端）
2. 手动创建 build 目录，命令行执行 cmake 生成 sln：

```
# 进入你的项目源码目录
cd E:\CUDA\Learning\cuda-samples\code\week01\Day02
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
```