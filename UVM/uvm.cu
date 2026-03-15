#include <cstdio>
#include <cuda_runtime.h>

__global__ void square(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = data[idx] * data[idx];
}

int main() {
    const int N = 1024;
    int *data;
    cudaMallocManaged(&data, N * sizeof(int)); // 分配 UVM 内存

    // 在 CPU 初始化数据
    for (int i = 0; i < N; i++) data[i] = i;

    // 启动 GPU 内核
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    square<<<grid, block>>>(data, N);

    // 显式同步，确保 GPU 完成计算
    cudaDeviceSynchronize();

    // CPU 直接访问结果
    for (int i = 0; i < 10; i++) printf("%d ", data[i]);
    cudaFree(data);
    return 0;
}