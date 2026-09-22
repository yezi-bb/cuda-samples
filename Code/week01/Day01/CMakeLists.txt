# 启用 compile_commands.json 文件
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# 指定版本
cmake_minimum_required(VERSION 3.20)
# 声明语言类型
project(Demo C CXX CUDA)
# 设置CUDA架构 89
set(CMAKE_CUDA_ARCHITECTURES 89)
# 旧版兼容写法，Windows下不容易踩坑
find_package(CUDAToolkit REQUIRED)
# 可执行文件
add_executable(Demo addKernel.cu)
#cuda选项
target_compile_options(Demo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
# 标准特性
target_compile_features(Demo PRIVATE cxx_std_17 cuda_std_17)
# 添加头文件
target_include_directories(Demo PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Common
)
# 链接 CUDA
target_link_libraries(Demo PRIVATE
    CUDA::cudart
)

