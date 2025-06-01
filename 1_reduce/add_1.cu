#include <stdio.h>

__global__ void add_one(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 10;
    }
}

int main() {
    const int N = 10;
    int host_data[N];

    // 初始化 host 数据
    for (int i = 0; i < N; ++i) {
        host_data[i] = i;
    }

    int *dev_data;
    size_t size = N * sizeof(int);

    // 分配设备内存
    cudaMalloc((void**)&dev_data, size);

    // 将数据从 host 拷贝到 device
    cudaMemcpy(dev_data, host_data, size, cudaMemcpyHostToDevice);

    // 启动核函数
    int threads_per_block = 8;
    int blocks = (N + threads_per_block - 1) / threads_per_block;
    add_one<<<blocks, threads_per_block>>>(dev_data, N);

    // 等待核函数执行完成
    cudaDeviceSynchronize();

    // 将数据从 device 拷贝回 host
    cudaMemcpy(host_data, dev_data, size, cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(dev_data);
    cudaDeviceSynchronize();
    // 打印结果，检查是否每个元素加 1
    printf("Result after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("host_data[%d] = %d\n", i, host_data[i]);
    }

    return 0;
}