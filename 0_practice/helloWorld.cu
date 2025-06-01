#include <stdio.h>

__global__ void hello_from_GPU(){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world from GPU from block %d thread %d threadIdx %d\n", blockIdx.x, threadIdx.x, idx);
}


int main(void){
    hello_from_GPU<<<2, 4>>>();
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}