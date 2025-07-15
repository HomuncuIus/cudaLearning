#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define THREAD_PER_BLOCK 256


__global__ void reduce(float *d_in,float *d_out){
    float *input_begin = d_in + blockDim.x * blockIdx.x; // 获取当前block的起始索引的数组

    // VERSION 1: intuitive 
    // if (threadIdx.x == 0 or 2 or 4 or 6){
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 1] 
    // }
    // __syncthreads();
    // if (threadIdx.x == 0 or 4){
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 2] 
    // }
    // __syncthreads();
    // if (threadIdx.x == 0){
    //     input_begin[threadIdx.x] += input_begin[threadIdx.x + 4] 
    // }
    // __syncthreads();

    // VERSION 2: using loop for more universal situation 
    for (int t = 1; t < blockDim.x; t *= 2){
        if (threadIdx.x % (t * 2) == 0) input_begin[threadIdx.x] += input_begin[threadIdx.x + t];
        __syncthreads();
    }

    if (threadIdx.x == 0){
        d_out[blockIdx.x] = input_begin[0];
    }
}

bool check(float *out, float *res, int n){
    for (int i = 0; i < n; i++){
        if (abs(out[i] - res[i]) > 0.005){
            return false;
        }
    }
    return true;
}

int main(){
    const int N = 1 << 25; // num of threads 
    int block_num = N / THREAD_PER_BLOCK; // 32 * 1024 * 4 

    // allocate memory in CPU, for final result check
    float *input = (float *)malloc(N * sizeof(float)); 
    float *out=(float *)malloc((N/THREAD_PER_BLOCK) * sizeof(float)); 
    float *res=(float *)malloc((N/THREAD_PER_BLOCK) * sizeof(float));

    // generate random data for reduce op
    for (int i = 0; i < N; i++){
        // input[i] = 2.0 * (float)drand48() - 1.0;
        input[i] = 1;
    }

    // calculate result in CPU
    for (int i = 0; i < block_num; i++){
        float cur = 0;
        for (int j = 0; j < THREAD_PER_BLOCK; j++){
            cur += input[i * THREAD_PER_BLOCK + j];
        }
        res[i] = cur;
    }

    // allocate memory in GPU
    float *d_input;
    cudaMalloc((void **)&d_input, N * sizeof(float));
    float *d_output;
    cudaMalloc((void **)&d_output,(N / THREAD_PER_BLOCK) * sizeof(float));

    // make sure the inputs of CPU and GPU are the same 
    cudaMemcpy(d_input, input, N*sizeof(float), cudaMemcpyHostToDevice);

    // init Grid and Block 
    dim3 Grid(N / THREAD_PER_BLOCK);
    dim3 Block(THREAD_PER_BLOCK);

    reduce<<<Grid,Block>>>(d_input,d_output);

    // move the result of GPU to out 
    cudaMemcpy(out, d_output, block_num * sizeof(float), cudaMemcpyDeviceToHost);

    // compare GPU out and CPU res
    if (check(out, res, block_num)) printf("the ans is right\n");
    else{
        printf("the ans is wrong\n");
        for (int i = 0; i < block_num; i++){
            printf("%lf ",out[i]);
        }
        printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_output);
}