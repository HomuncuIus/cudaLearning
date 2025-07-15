// baseline中的kernel函数使用global memory，会导致核函数中input被直接修改，使用shared memory可以避免该问题 -- 非性能优化

#include <stdio.h>
#include <random>

#define DATA_PER_BLOCK 256
#define THREAD_PER_BLOCK 128
#define WARP_SIZE 32

__global__ void reduce(float* input, float* res){
    
    __shared__ volatile float shared[DATA_PER_BLOCK]; // 静态shared memory，效率更高但此处的参数不能用blockDim.x，只能用宏或者模板，不然不是静态的，会报错
    shared[threadIdx.x] = input[blockIdx.x * DATA_PER_BLOCK + threadIdx.x]; // 因为只有128个线程了，每个线程要搬两份数据
    shared[THREAD_PER_BLOCK + threadIdx.x] = input[blockIdx.x * DATA_PER_BLOCK + THREAD_PER_BLOCK + threadIdx.x]; 
    __syncthreads(); // 所有线程都要取完数据才能下一步
    
    for (int offset = DATA_PER_BLOCK / 2; offset > WARP_SIZE; offset /= 2){
        if (threadIdx.x < offset){
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    for (int offset = WARP_SIZE; offset >= 1; offset /= 2){
        if (threadIdx.x < offset){
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
    }

    if (threadIdx.x == 0){
        res[blockIdx.x] = shared[0];
    }
}

int main() {
    const int N = 1 << 20;
    const int nBytes = N * sizeof(float);
    
    // 初始化 host 数据
    std::random_device rd;  // 随机设备，用于种子
    std::mt19937 gen(rd()); // Mersenne Twister 引擎
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // 均匀分布 [0.0, 1.0)

    
    float input[N], result[N / DATA_PER_BLOCK];
    float resultHost[N / DATA_PER_BLOCK];
    float blockRes = 0;
    int assignCount = 0;
    int resIndex = 0;
    for (size_t i = 0; i < N; i++){
        input[i] = dis(gen); // 生成随机 float
        // input[i] = 1.0; // 1.0
        blockRes += input[i];
        assignCount += 1;
        // printf("data at %zu is %f \n", i, input[i]);
        if (assignCount % DATA_PER_BLOCK == 0){
            resultHost[resIndex] = blockRes;
            blockRes = 0;
            resIndex += 1;
        }
    }
    // for (size_t i = 0; i < N / DATA_PER_BLOCK; i++){
    //     printf("result from host at %zu is %f \n", i, resultHost[i]);
    // }
    // 分配设备内存
    float *d_input, *d_res;
    cudaMalloc((void**)&d_input, nBytes);
    cudaMalloc((void**)&d_res, nBytes / DATA_PER_BLOCK);

    // 拷贝host侧入参进入device
    cudaMemcpy(d_input, (float*)input, nBytes, cudaMemcpyHostToDevice);
    
    // 调用核函数
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid(N / DATA_PER_BLOCK);
    reduce <<<grid, block>>>(d_input, d_res);

    // 等待核函数执行完成
    cudaDeviceSynchronize();
    
    // 拷贝device侧结果回host
    cudaMemcpy((float*)result, d_res, nBytes / DATA_PER_BLOCK, cudaMemcpyDeviceToHost);
    // cudaMemcpy((float*)input, d_input, nBytes, cudaMemcpyDeviceToHost);

    // for (size_t i = 0; i < N; i++){
    //     // printf("input[%zu] is %f\n", i, input[i]);
    //     if (input[i] > 1) {
    //         printf("input[%zu] is %f\n", i, input[i]);
    //     }
    // }
    // 检验结果
    float accError = 0.0;
    bool errorCheck = false;
    for (size_t i = 0; i < N / DATA_PER_BLOCK; i++){
        accError += fabs(result[i] - resultHost[i]);
        errorCheck = fabs(result[i] - resultHost[i]) > 1e-3f || errorCheck;
        if (fabs(result[i] - resultHost[i]) > 1e-3f){
            printf("-------result from device at %zu is %f host %f--------\n", i, result[i], resultHost[i]);
            continue; 
        }
        // printf("result from device at %zu is %f host %f\n", i, result[i], resultHost[i]);
    }
    printf("error check pass? %d accError is %f\n", !errorCheck, accError);

    // 释放设备内存
    cudaFree(d_input);
    cudaFree(d_res);

    return 0;
}