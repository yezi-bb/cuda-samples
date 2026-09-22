#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel: vector add
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host memory allocation
    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    // Init array
    for(int i=0; i<N; i++)
    {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // Device memory allocation
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host -> device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1) / threadsPerBlock;
    std::cout << "blocks: " << blocksPerGrid << ", threads per block: " << threadsPerBlock << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check kernel launch error
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
         char c= getchar();
        return -1;
    }

    // Copy device -> host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    bool ok = true;
    for(int i=0; i<N; i++)
    {
        if(h_C[i] != h_A[i] + h_B[i])
        {
            std::cerr << "Error at index " << i << std::endl;
            ok = false;
            break;
        }
    }
    if(ok) std::cout << "All results are correct!" << std::endl;
    char c= getchar();
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();
    return 0;
}
