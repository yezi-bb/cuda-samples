# 第 4 周：多设备、纹理与 Driver API

收尾 `cpp/0_Introduction/`。多 GPU 和图形互操作按第 1 周的能力表决定精读还是走读。

## 周一：多 GPU 与 P2P（有第二块 GPU 则精读，否则走读）

**路径**：`simpleMultiGPU`、`simpleP2P`、`cudaOpenMP`

**做**

- `simpleMultiGPU`：每个 GPU 一个主机线程，上下文如何绑定
- `simpleP2P`：`cudaDeviceCanAccessPeer`、`cudaDeviceEnablePeerAccess`、统一虚拟地址下的跨设备指针
- `cudaOpenMP`：用 OpenMP 把工作分到多块 GPU。没有第二块 GPU 或没有 OpenMP 时走读，记下样例如何检测设备数量并退出

**想清楚**

1. P2P 为什么通常要求两块同类 GPU？
2. 打开 peer access 之后，还需要 `cudaMemcpyPeer` 吗，什么时候直接解引用？
3. 多线程访问 GPU 时，运行时上下文是每线程一份还是共享的？

**完成标准**：能说出「一台机器上多块 GPU」和「一块 GPU 上多条流」不是同一层次的并行。

## 周二：进程间通信与 MPI（按环境走读或精读）

**路径**：`simpleIPC`、`simpleMPI`

**做**

- `simpleIPC`：一个进程一块 GPU，用 CUDA IPC 传递设备指针。Linux 或 Windows TCC 才有意义，普通 Windows WDDM 多半要走读
- `simpleMPI`：主机用 MPI 分发，设备做计算。没装 MPI 就读 `CMakeLists.txt` 看它如何被跳过，再读源码里的通信边界

**想清楚**

1. IPC 共享的是设备指针，还是把数据拷回主机再走套接字？
2. MPI 进程和 CUDA 设备的映射在样例里是谁决定的？
3. 和周一的多线程多 GPU 相比，多进程多了哪一步句柄导出/导入？

**完成标准**：能区分三种多设备形态：多线程、多进程 IPC、MPI。

## 周三：纹理入门（精读 simpleTexture，对照其余）

**路径**：`simpleTexture`、`simpleTexture3D`、`simpleTextureDrv`、`simplePitchLinearTexture`

**做**

- 精读 `simpleTexture`：`cudaArray`、纹理对象、`tex2D` 的归一化坐标和滤波
- 对照 3D、pitch 线性、Driver API 版本，各记一个「存储布局差异」
- 能编就跑 `simpleTexture`，看主机侧对采样结果的校验

**想清楚**

1. 纹理适合哪种访问模式（二维局部性、硬件插值），不适合哪种（被多个线程散写）？
2. `cudaArray` 和普通 `cudaMalloc` 线性内存为什么不能混用同一套拷贝参数？
3. Driver 版纹理样例改的是绑定方式还是采样数学？

**完成标准**：能解释纹理对象、数组、采样器三者谁管布局、谁管寻址。

## 周四：立方体、分层、表面写入（对照）

**路径**：`simpleCubemapTexture`、`simpleLayeredTexture`、`simpleSurfaceWrite`、`simpleCUDA2GL`

**做**

- 立方体纹理和分层纹理各用哪一个采样函数
- `simpleSurfaceWrite`：表面写入和只读纹理的差别
- `simpleCUDA2GL`：CUDA 图像如何交回 OpenGL。没有 OpenGL 显示就走读，看互操作资源的注册和映射

**想清楚**

1. 表面写入为什么不能用普通纹理取数函数代替？
2. 分层纹理和三维纹理在坐标上差在哪一维？
3. 图形互操作样例里，资源的所有者是 CUDA 还是图形 API？

**完成标准**：列出本周遇到的四种纹理形态：2D、3D、cubemap、layered，各用一句话。

## 周五：原子、投票、屏障、Runtime/Driver 混用（精读两个）

**路径**：`simpleAtomicIntrinsics`、`systemWideAtomics`、`simpleVoteIntrinsics`、`simpleCooperativeGroups`、`simpleAWBarrier`、`simpleDrvRuntime`

**做**

- 精读 `simpleAtomicIntrinsics`：全局原子加，理解冲突时的正确性与串行化
- 精读 `simpleVoteIntrinsics`：`__any_sync` / `__all_sync` 的 mask 必须覆盖哪些线程
- 对照 `simpleCooperativeGroups`：组内同步和 `__syncthreads` 的范围
- `simpleAWBarrier`、`systemWideAtomics`、`simpleDrvRuntime` 各写三行：API 名称、同步范围、本机能否运行
- `simpleDrvRuntime` 重点看：Runtime 加载的 fatbinary 如何交给 Driver 再启动

**想清楚**

1. 投票函数为什么要带 `_sync` 和 mask？
2. 协作组在 Introduction 这份样例里限制在一个 block 内，跨 block 的版本在哪一册（预告第 5、8 周）？
3. 系统范围原子和设备全局原子差在参与者是否包括 CPU 或其他 GPU？

**完成标准**：Introduction 一册在笔记里有一张「已读样例 → 一个 API」索引。没跑的样例注明原因。
