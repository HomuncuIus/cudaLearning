#include <stdio.h>


__global__ void add(float* a, float* b, float* res){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    res[index] = a[index] + b[index];
}

int main() {

    // 初始化 host 数据
    int N = 1 << 16;
    int nBytes = N * sizeof(float);

    // 分配设备内存
    float *a, *b, *res;
    cudaMallocManaged((void**)&a, nBytes);
    cudaMallocManaged((void**)&b, nBytes);
    cudaMallocManaged((void**)&res, nBytes);

    for (size_t i = 0; i < N; i++){
        a[i] = 1.0;    
        b[i] = 2.0;
        res[i] = 0.0;
    }

    // 调用核函数
    dim3 block(1024);
    dim3 grid(N / block.x);
    add <<<grid, block>>>(a, b, res);

    // 等待核函数执行完成
    cudaDeviceSynchronize();

    // 检验结果
    float error = 0.0;
    for (size_t i = 0; i < N; i++){
        error += fabs(a[i] + b[i] - res[i]);
        printf("res for %zu is %f\n", i, res[i]);
    }
    printf("error is %f\n", error);

    // 释放设备内存
    cudaFree(a);
    cudaFree(b);
    cudaFree(res);

    return 0;
}