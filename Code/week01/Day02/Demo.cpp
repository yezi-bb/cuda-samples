#include <iostream>
#include <cuda_runtime.h>

int main()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "deviceCount: " << deviceCount << std::endl;
    char c = getchar();
    return 0;
}