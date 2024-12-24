#include <stdio.h>

int main() {
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    printf("使用GPU device: %d - %s\n", dev, devProp.name);
    printf("SM的数量: %d\n", devProp.multiProcessorCount);
    printf("每个线程块的共享内存大小: %zu Byte\n", devProp.sharedMemPerBlock);
    printf("每个线程块的最大线程数: %d\n", devProp.maxThreadsPerBlock);
    printf("每个SM的最大线程数: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("每个SM的最大线程束数: %d\n", devProp.maxThreadsPerMultiProcessor / 32);
    return 0;
}