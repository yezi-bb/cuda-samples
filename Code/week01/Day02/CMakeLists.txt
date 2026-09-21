# ==============================================================
# deviceQuery CMakeLists.txt 【带完整详细注释，可直接覆盖粘贴】
# 用途：CUDA Samples 探针程序 deviceQuery 构建脚本
# ==============================================================

# 指定本CMake脚本要求的最低CMake版本
# CMAKE_CUDA_ARCHITECTURES、CUDA导入目标、生成器表达式等特性依赖3.20+
cmake_minimum_required(VERSION 3.20)

# 引入CUDA样例架构探测脚本（相对路径向上跳3层到samples根目录cmake文件夹）
# 作用：探测GPU计算能力、辅助设置CMAKE_CUDA_ARCHITECTURES
# 注意：单独在deviceQuery目录cmake时，缺少顶层上下文，不会自动加载全套SM列表
# include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/DetectCudaArch.cmake")

# 全局构建开关：如果顶层项目设置 SAMPLE_SKIP_BUILD=ON，则直接退出，不编译当前样例
# 批量编译全部CUDA Samples时用来跳过指定样例；单独编译此目录默认OFF，不会触发return
if(SAMPLE_SKIP_BUILD)
    return()
endif()

# 定义项目名称 deviceQuery
# LANGUAGES C CXX CUDA：声明项目使用C、C++、CUDA三种语言
# 声明CUDA语言后，CMake自动寻找nvcc，接管CUDA代码编译流程
project(deviceQuery LANGUAGES C CXX CUDA)

# 加载CUDA样例公共配置脚本，所有CUDA样例共用
# 内部包含统一编译警告、ENABLE_CUDA_DEBUG（CUDA调试开关）、全局编译属性等
include("${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSampleCommon.cmake")

# 查找本机CUDA Toolkit工具包，REQUIRED表示找不到则直接终止CMake配置并报错
# 成功后自动定义导入目标 CUDA::cudart，以及变量 CUDAToolkit_INCLUDE_DIRS
find_package(CUDAToolkit REQUIRED)

# 【旧式CMake全局头文件】添加samples公共Common目录
# 作用：给当前目录所有目标暴露Common下的头文件
# 现代CMake更推荐使用 target_include_directories 仅针对单个目标设置
include_directories(../../../Common)

# 创建可执行程序目标 deviceQuery，源码文件为 deviceQuery.cpp
# CMake识别源码并根据project声明的CUDA语言，将CUDA相关代码交由nvcc编译
add_executable(deviceQuery deviceQuery.cpp)

# 目标私有编译选项（生成器表达式）
# $<$<COMPILE_LANGUAGE:CUDA>:xxx>：只有编译CUDA语言文件时，才把选项传给nvcc；C/C++编译忽略
# --extended-lambda：nvcc参数，开启CUDA扩展lambda，支持在GPU核函数内使用C++ lambda表达式
# PRIVATE：该编译选项仅作用于deviceQuery本身，不会传递给其他依赖项目
target_compile_options(deviceQuery PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# 指定目标需要的C++/CUDA C++标准特性
# cxx_std_17：主机端C++代码使用C++17
# cuda_std_17：设备端CUDA代码使用CUDA C++17
# CMake会自动向MSVC/nvcc追加对应标准参数，不用手写 /std:c++17
target_compile_features(deviceQuery PRIVATE cxx_std_17 cuda_std_17)

# 给deviceQuery目标添加私有头文件搜索路径
# 换成这一行，PRIVATE，同时给编译器 + VS IntelliSense 提供头文件路径
target_include_directories(deviceQuery PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common
)

# 链接库：CUDA运行时库 cudart（cudaMalloc、cudaGetDeviceProperties等API依赖此库）
# CUDA::cudart 是CMake导入目标，自带头文件、库路径、依赖，比手动写-lcudart更稳定跨平台
# PUBLIC：本目标链接cudart；如果其他目标链接deviceQuery，会继承cudart依赖；可执行文件场景PUBLIC/PRIVATE差别很小
target_link_libraries(deviceQuery PUBLIC
    CUDA::cudart
)

# 引入安装脚本模块，注册安装规则
include(InstallSamples)
# 设置样例安装逻辑，cmake --install 时将exe复制到指定安装目录
# 普通编译（cmake --build）不会触发安装，不影响日常调试构建
setup_samples_install()
