#include <cstdio>
#include <random>

__global__ void add(float* a, float*b, float* res){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    res[tid] = a[tid] + b[tid];
    // if (tid < 10){
    //     printf("DEV: a is %f, b is %f, res is %f\n", a[tid], b[tid], res[tid]);
    // }
}

int main(){
    constexpr int N = 1 << 25;
    constexpr int NByte = N * sizeof(float);
    printf("N is %d, NByte is %d\n", N, NByte);

    std::random_device rd;  // 随机设备，用于种子
    std::mt19937 gen(rd()); // Mersenne Twister 引擎
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // 均匀分布 [0.0, 1.0)

    float *host_a = new float[N];
    float *host_b = new float[N];
    float *host_res = new float[N];
    for(int i = 0; i < N; i++){
        host_a[i] = 1.0;
        host_b[i] = dis(gen);
        host_res[i] = host_a[i] + host_b[i];
        // if (i < 5){
        //     printf("host_a is %f, host_b is %f, host_res is %f\n", host_a[i], host_b[i], host_res[i]);
        // }
    }

    float *dev_a, *dev_b, *dev_res;
    float *res_from_dev = new float[N];
    cudaMalloc((void**)&dev_a, NByte);
    cudaMalloc((void**)&dev_b, NByte);
    cudaMalloc((void**)&dev_res, NByte);

    cudaMemcpy(dev_a, host_a, NByte, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, NByte, cudaMemcpyHostToDevice);

    dim3 grid(N / (1 << 8));
    dim3 block(64);
    add<<<grid, block>>>(dev_a, dev_b, dev_res);

    cudaDeviceSynchronize();
    cudaMemcpy(res_from_dev, dev_res, NByte, cudaMemcpyDeviceToHost);
    float err = 0;
    for (int i = 0; i < (1 << 10); i++){
        err = res_from_dev[i] - host_res[i];
    }
    printf("The error is %f\n", err);

    return 0;
}

