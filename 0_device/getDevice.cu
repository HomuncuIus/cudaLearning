#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int dev = 0;
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);

    // 1. 设备基本信息
    printf("===== GPU设备基本信息 =====\n");
    printf("使用GPU device %d: %s\n", dev, devProp.name);
    printf("计算框架: %d.%d\n", devProp.major, devProp.minor);
    printf("全局内存总量: %.2f GB\n", devProp.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("常量内存总量: %zu KB\n", devProp.totalConstMem / 1024);
    printf("时钟频率: %.2f MHz\n", devProp.clockRate / 1000.0);

    // 2. 线程与Block配置
    printf("\n===== 线程与Block配置 =====\n");
    printf("每个Block的共享内存大小: %zu Byte\n", devProp.sharedMemPerBlock);
    printf("每个Block的寄存器数量: %d\n", devProp.regsPerBlock);
    printf("每个Block的最大线程数: %d\n", devProp.maxThreadsPerBlock);
    printf("Block维度最大值: [%d, %d, %d]\n", 
           devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Grid维度最大值: [%d, %d, %d]\n",
           devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);

    // 3. SM架构信息
    printf("\n===== SM架构信息 =====\n");
    printf("SM数量: %d\n", devProp.multiProcessorCount);
    printf("每个SM的最大线程数: %d\n", devProp.maxThreadsPerMultiProcessor);
    printf("每个SM的最大线程束数: %d\n", devProp.maxThreadsPerMultiProcessor / devProp.warpSize);
    printf("每个SM的最大Block数: %d\n", devProp.maxBlocksPerMultiProcessor);
    printf("每个SM的共享内存总量: %zu KB\n", devProp.sharedMemPerMultiprocessor / 1024);

    // 4. 内存与缓存
    printf("\n===== 内存与缓存信息 =====\n");
    printf("显存总线位宽: %d-bit\n", devProp.memoryBusWidth);
    printf("显存频率: %.2f GHz\n", devProp.memoryClockRate / 1e6);
    printf("L2缓存大小: %zu KB\n", devProp.l2CacheSize / 1024);
    printf("是否支持全局内存L1缓存: %s\n", devProp.globalL1CacheSupported ? "是" : "否");

    // 5. 功能支持
    printf("\n===== 功能支持 =====\n");
    printf("是否支持并发Kernel执行: %s\n", devProp.concurrentKernels ? "是" : "否");
    printf("是否支持统一寻址: %s\n", devProp.unifiedAddressing ? "是" : "否");
    printf("是否启用ECC: %s\n", devProp.ECCEnabled ? "是" : "否");
    printf("是否支持协作组Kernel: %s\n", devProp.cooperativeLaunch ? "是" : "否");
    printf("单/双精度性能比: %d:1\n", devProp.singleToDoublePrecisionPerfRatio);

    // 6. 其他高级特性
    printf("\n===== 高级特性 =====\n");
    printf("异步引擎数量: %d\n", devProp.asyncEngineCount);
    printf("是否支持主机直接访问Managed Memory: %s\n", 
           devProp.directManagedMemAccessFromHost ? "是" : "否");
    printf("持久化L2缓存最大大小: %zu MB\n", devProp.persistingL2CacheMaxSize / (1024 * 1024));

    return 0;
}