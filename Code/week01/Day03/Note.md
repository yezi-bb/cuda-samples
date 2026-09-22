**读**

- `Common/helper_cuda.h`：`checkCudaErrors`、`getLastCudaError`、`findCudaDevice`、`gpuGetMaxGflopsDeviceId`、`_ConvertSMVer2Cores`
- `Common/helper_cuda_drvapi.h`、`Common/helper_string.h`、`Common/helper_timer.h` 的头注释和主要入口
- 任意一个已能打开的样例 `CMakeLists.txt`，看它怎样把 `Common` 加进 include 路径

**做**

- 在 `deviceQuery` 源文件里搜 `findCudaDevice` 或 `checkCudaErrors`，确认样例没有自己再写一套错误处理
- 列出后面会反复见到的 6 个头文件：`helper_cuda.h`、`helper_string.h`、`helper_timer.h`、`helper_math.h`、`helper_image.h`、`helper_multiprocess.h`

**想清楚**

1. `checkCudaErrors` 失败时进程怎么退出，和 kernel 启动错误为什么常常要再调 `getLastCudaError`？

   - exit(EXIT_FAILURE); 直接内部处理退出了，记录日志 ；
   - CUDA kernel launch（`<<<>>>`）是**异步调用**：
     - 核函数提交到 GPU 队列，**kernel 启动本身几乎永远返回成功**，不会立刻捕获核内部非法访问、越界、shared 内存溢出这类运行时报错；
     - `<<<>>>` 语法**不能直接捕获 GPU 运行异常**，主机端函数返回值只代表 “提交任务成功”；
     - GPU 上真正执行报错会保存在**全局上下文的 last error 状态**里，必须调用 `cudaGetLastError()` 抓取这个 pending 错误

2. `EXIT_WAIVED`（值为 2）和普通失败有什么区别？这和后面的测试脚本有什么关系？

   - `EXIT_FAILURE`(1)：**程序本身出错**，代码异常、CUDA 调用失败、断言失败，属于程序运行故障；
   - `EXIT_WAIVED`(2)：**测试跳过 / 不执行**，不是 bug。CUDA Samples 测试框架专用返回码： 含义：当前环境**不满足样例运行条件**，不是代码崩溃。 举例：GPU 算力版本太低、缺少对应硬件、不支持该特性，样例主动放弃执行，不是报错。

   > 和测试脚本关系： 上层 shell/Python 测试脚本读取进程退出码：
   >
   > - 返回 0：PASS 测试通过
   > - 返回 1：FAIL，代码故障
   > - 返回 2：WAIVED，**跳过本次测试，不标记为失败**，CI / 自动化测试不会判定用例崩掉，仅标记为 “不适用”。

3. 选设备的默认策略是「第 0 块」还是「理论峰值更高的那块」？

   - **默认：优先选 0 号 GPU（device 0）**，**不是按理论峰值性能自动选卡**。

**完成标准**：能指出一个样例从命令行 `-device=` 到真正 `cudaSetDevice` 的函数链。

- 命令行传入参数，例如 `./sampleName -device=1`
- 全局参数解析工具 `cmdLineParse()` / `_cmdLineDevice` 捕获 `-device` 参数，解析得到整数 deviceID
- `printf` 打印选中设备 ID 日志
- 调用 `cudaSetDevice(deviceID)`，把当前主机线程绑定到指定 GPU 上下文
- 紧接着一般会搭配 `checkCudaErrors(cudaGetDeviceProperties(&prop, deviceID))` 读取设备属性做校验
- 校验失败直接 exit；校验通过，后续所有 CUDA API/kernel 都作用在这个 device 上

精简链路： `命令行参数` → `cmdLineParse 解析 -device` → 提取 device 编号 → `cudaSetDevice(id)` 绑定 GPU 上下文