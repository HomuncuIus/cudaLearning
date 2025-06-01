#include <stdio.h>

__global__ void hello_from_GPU_one_dim(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world from GPU from block %d thread %d threadIdx %d\n", blockIdx.x, threadIdx.x, idx);
}

__global__ void hello_from_GPU_two_dim(){
    int block_global_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int thread_global_idx = block_global_idx * blockDim.x * blockDim.y + blockDim.x * threadIdx.y + threadIdx.x;
    // int block_idx = 
    printf("Hello world from GPU from block x:%d y:%d thread x:%d y:%d block_global_idx %d thread_global_idx %d\n", 
        blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, block_global_idx, thread_global_idx);
}

__global__ void hello_from_GPU_complete(){
    int block_global_idx = (blockIdx.z * gridDim.x * gridDim.y) + (blockIdx.y * gridDim.x) + (blockIdx.x);
    int thread_global_idx = (block_global_idx * blockDim.x * blockDim.y * blockDim.z) + 
                            (threadIdx.z * blockDim.x * blockDim.y) + 
                            (threadIdx.y * blockDim.x) + (threadIdx.x);
    printf("Hello world from GPU from block x:%d y:%d z:%d thread x:%d y:%d z:%d block_global_idx %d thread_global_idx %d\n", 
        blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, block_global_idx, thread_global_idx);
}


int main(void){
    // 检查可用cuda设备
    int iDeviceCount = 0;
    cudaGetDeviceCount(&iDeviceCount);
    printf("there are %d devices available\n", iDeviceCount);

    // 输出helloWorld -- dim 1 * dim 1 
    printf("hello_from_GPU_one_dim\n");
    hello_from_GPU_one_dim<<<2, 4>>>();
    cudaDeviceSynchronize();

    // 输出helloWorld -- dim 2 * dim 2
    printf("hello_from_GPU_two_dim\n");
    dim3 grid_size(3, 2); // gridDim.x = 3, gridDim.y = 2
    dim3 block_size(5, 4); // blockDim.x = 5, blockDim.y = 4
    hello_from_GPU_two_dim<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();

    // 输出helloWorld -- dim 3 * dim 3
    printf("hello_from_GPU_complete\n");
    dim3 grid_size_complete(3, 2, 3); // gridDim.x = 3, gridDim.y = 2, gridDim.z = 3
    dim3 block_size_complete(2, 2, 3); // blockDim.x = 2, blockDim.y = 2, blockDim.z = 3
    hello_from_GPU_complete<<<grid_size_complete, block_size_complete>>>();
    cudaDeviceSynchronize();
    return 0;
}