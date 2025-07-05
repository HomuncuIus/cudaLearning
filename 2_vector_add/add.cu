#include <stdio.h>


__global__ void add(float* a, float* b, float* res){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    res[index] = a[index] + b[index];
}

int main() {

    // 初始化 host 数据
    int N = 1 << 16;
    int nBytes = N * sizeof(float);
    float a[N], b[N], res[N];
    for (size_t i = 0; i < N; i++){
        a[i] = 1.0;    
        b[i] = 2.0;
        res[i] = 0.0;
    }

    // 分配设备内存
    float *d_a, *d_b, *d_res;
    cudaMalloc((void**)&d_a, nBytes);
    cudaMalloc((void**)&d_b, nBytes);
    cudaMalloc((void**)&d_res, nBytes);

    // 拷贝host侧入参进入device
    cudaMemcpy(d_a, (float*)a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, (float*)b, nBytes, cudaMemcpyHostToDevice);

    // 调用核函数
    dim3 block(1024);
    dim3 grid(N / block.x);
    add <<<grid, block>>>(d_a, d_b, d_res);

    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 拷贝device侧结果回host
    cudaMemcpy((float*)res, d_res, nBytes, cudaMemcpyDeviceToHost);

    // 检验结果
    float error = 0.0;
    for (size_t i = 0; i < N; i++){
        error += fabs(a[i] + b[i] - res[i]);
        printf("res for %zu is %f\n", i, res[i]);
    }
    printf("error is %f\n", error);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_res);

    return 0;
}