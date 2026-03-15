#include <cstdio>
#include <cuda_runtime.h>

// GPU 内核：计算数组元素的平方
__global__ void add_one(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] +1;
    }
}

int main() {
    const int N = 1 << 20; // 1M 元素（约 4MB 数据）
    int *data;

    // 1. 分配 UVM 托管内存
    cudaMallocManaged(&data, N * sizeof(int));

    // 2. 在 CPU 上初始化数据
    for (int i = 0; i < N; i++) {
        data[i] = static_cast<int>(i);
    }

    // 3. 显式预取数据到 GPU 显存（设备号 0）
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaMemPrefetchAsync(data, N * sizeof(int), 0, stream); // 目标设备：GPU 0
    cudaStreamSynchronize(stream);

    // 4. 启动 GPU 内核
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    add_one<<<grid, block>>>(data, N);

    // 5. 预取数据回 CPU 内存（可选）
    cudaMemPrefetchAsync(data, N * sizeof(int), cudaCpuDeviceId, stream); // 目标设备：CPU
    cudaStreamSynchronize(stream);

    // 6. 同步等待 GPU 完成（确保数据一致性）
    cudaDeviceSynchronize();

    // 7. CPU 访问结果（此时数据已回迁）
    printf("data[0] = %d\n", data[0]);
    printf("data[%d] = %d\n", N-1, data[N-1]);

    // 清理资源
    cudaFree(data);
    cudaStreamDestroy(stream);
    return 0;
}