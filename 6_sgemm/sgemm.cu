#include <cstdio>
#include <random>
#include "util.h"

using namespace util;
class Perf {
public: 
    Perf(){
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
        cudaEventRecord(m_start);
        cudaEventSynchronize(m_start);
    }

    ~Perf(){
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
        float elapsed_time = 0;
        cudaEventElapsedTime(&elapsed_time, m_start, m_end);
        printf("Time is %f\n", elapsed_time);
    }
private:
    cudaEvent_t m_start, m_end;
};


// 主机侧GEMM函数
void host_sgemm(float* a, float* b, float* c, const size_t m_size, const size_t n_size, const size_t k_size){
    for (size_t m = 0; m < m_size; m++){
        for (size_t n = 0; n < n_size; n++){
            float temp = 0;
            for (size_t k = 0; k < k_size; k++){
                temp  += a[m * k_size + k] * b[k * n_size + n];
            }
            c[m * n_size + n] = temp; 
        }
    }
}

// 设备侧GEMM函数v1,功能实现
__global__ void dev_sgemm(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float temp = 0.f;
    for (int k = 0; k < k_size; k++){
        // printf("tid x %d y %d k %zu aid %zu bid %zu a %f b %f\n", x, y, k, y * k_size + k, k * n_size + x, a[y * k_size + k], b[k * n_size + x]);
        temp += a[y * k_size + k] * b[k * n_size + x];
    }
    // printf("tid x %d y %d res %f\n", x, y, temp);
    c[y * n_size + x] = temp;
}

// 设备侧GEMM函数v2,引入共享内存，减少全局内存搬运次数
template <int bn, int bm, int bk>
__global__ void dev_sgemm_sharedmem(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_shared[bm][bk]; // 注意不能用k_size,需要用编译器常量
    __shared__ float b_shared[bk][bn];
    float temp = 0.f;
    for (int k = 0; k < k_size; k+=bk){
        // 注意,下面a_shared和b_shared是因为正好形状一致且都为矩形且和block的x，y都能对应才能这么取，不然肯定有问题，得做分支判断
        a_shared[threadIdx.y][threadIdx.x] = a[y * k_size + k + threadIdx.x]; // 对于a来说，y * k_size已经求出了纵轴的偏移量，之后只要求横轴的偏移量，即k + threadIdx.x
        b_shared[threadIdx.y][threadIdx.x] = b[x + (k + threadIdx.y) * n_size]; // 对于b来说，k每加16，就下来了16行，即16 * n_size个数据
         __syncthreads(); // 所有线程都要取完数据才能下一步
        for (int i = 0; i < bk; i++){
            temp += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x];
        }
         __syncthreads();
    }
    c[y * n_size + x] = temp; 
}

// 设备侧GEMM函数v2_1,试一下double buffer
// 好吧 几乎没有效果 ...
template <int bn, int bm, int bk>
__global__ void dev_sgemm_sharedmem_db(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_shared[2][bm][bk]; // 注意不能用k_size,需要用编译器常量
    __shared__ float b_shared[2][bk][bn];
    float temp = 0.f;
    bool db = true;
    a_shared[0][threadIdx.y][threadIdx.x] = a[y * k_size + threadIdx.x]; 
    b_shared[0][threadIdx.y][threadIdx.x] = b[x + threadIdx.y * n_size]; 
    __syncthreads();
    for (int k = bk; k < k_size; k+=bk){
        a_shared[db][threadIdx.y][threadIdx.x] = a[y * k_size + k + threadIdx.x]; 
        b_shared[db][threadIdx.y][threadIdx.x] = b[x + (k + threadIdx.y) * n_size]; 
        for (int i = 0; i < bk; i++){
            temp += a_shared[!db][threadIdx.y][i] * b_shared[!db][i][threadIdx.x];
        }
        db = !db;
         __syncthreads();
    }
    for (int i = 0; i < bk; i++){
        temp += a_shared[!db][threadIdx.y][i] * b_shared[!db][i][threadIdx.x];
    }
    c[y * n_size + x] = temp; 
}

// 设备侧GEMM函数v3,引入float4，减少全局内存向共享内存搬运时的指令数
template <int bn, int bm, int bk>
__global__ void dev_sgemm_float4_improved(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_shared[32][32]; // 注意不能用k_size,需要用编译器常量
    __shared__ float b_shared[32][32];
    float temp[4] = {0.0};
    for (int k = 0; k < k_size; k+=bk){
        // 这里必然会出现bank conflict
        reinterpret_cast<float4*>(&(a_shared[threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(a)[y * k_size /4 + k/4 + threadIdx.x];
        reinterpret_cast<float4*>(&(b_shared[threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(b)[(k + threadIdx.y) * n_size/4 + x];
        __syncthreads(); // 所有线程都要取完数据才能下一步
        for (int j = 0; j < 4; j++){
            for (int i = 0; i < bk; i++){
                temp[j] += a_shared[threadIdx.y][i] * b_shared[i][threadIdx.x * 4 + j];
            }
        }
        __syncthreads();
    }
    reinterpret_cast<float4*>(&c[y * n_size + x * 4])[0] = reinterpret_cast<float4*>(&temp[0])[0];
}

// 设备侧GEMM函数v4(基于v3优化而来),引入外积和寄存器，减少向共享内存获取数据的次数
template <int bn, int bm, int bk>
__global__ void dev_sgemm_float4_outter_product(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_shared[64][64]; // 注意不能用k_size,需要用编译器常量
    __shared__ float b_shared[64][64];
    float temp[16] = {0.0};
    float a_reg[4] = {0.0};
    float b_reg[4] = {0.0};
    for (int k = 0; k < k_size; k+=bk){
        // 每个线程循环4次，每次用float4取4个数，共计16个数
        for (int move_index = 0; move_index < 4; move_index++){
            reinterpret_cast<float4*>(&(a_shared[threadIdx.y * 4 + move_index][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(&a[(y * 4 + move_index) * k_size + k + threadIdx.x * 4])[0];
            reinterpret_cast<float4*>(&(b_shared[threadIdx.y * 4 + move_index][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(&b[(k + threadIdx.y * 4 + move_index) * n_size + x * 4])[0];
        }
        __syncthreads(); // 所有线程都要取完数据才能下一步

        for (int i = 0; i < bk; i++){
            a_reg[0] = a_shared[threadIdx.y * 4][i];
            a_reg[1] = a_shared[threadIdx.y * 4 + 1][i];
            a_reg[2] = a_shared[threadIdx.y * 4 + 2][i];
            a_reg[3] = a_shared[threadIdx.y * 4 + 3][i];
            reinterpret_cast<float4*>(&b_reg[0])[0] = reinterpret_cast<float4*>(&(b_shared[i][threadIdx.x * 4]))[0];
            for (int j = 0; j < 4; j++){
                for (int i = 0; i < 4; i++){
                    temp[j * 4 + i] += a_reg[j] * b_reg[i];
                }
            }
        }
        __syncthreads();
    }

    for (int j = 0; j < 4; j++){
        reinterpret_cast<float4*>(&c[(y * 4 + j) * n_size + x * 4 ])[0] = reinterpret_cast<float4*>(&temp[j * 4])[0];
    }
}

// 设备侧GEMM函数v5,引入double buffer，减少同步(此版本基于v3优化，因为v4用的共享内存太多了，可能不能开双倍了)
// 实测效果变差了
template <int bn, int bm, int bk>
__global__ void dev_sgemm_double_buffer(float* a, float* b, float* c, const int m_size, const int n_size, const int k_size){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float a_shared[2][32][32]; // 注意不能用k_size,需要用编译器常量
    __shared__ float b_shared[2][32][32];
    float temp[4] = {0.0};

    bool pingpong = false;
    reinterpret_cast<float4*>(&(a_shared[pingpong][threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(a)[y * k_size /4 + threadIdx.x];
    reinterpret_cast<float4*>(&(b_shared[pingpong][threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(b)[threadIdx.y * n_size/4 + x];
    __syncthreads();
    for (int k = bk; k < k_size; k+=bk){
        reinterpret_cast<float4*>(&(a_shared[!pingpong][threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(a)[y * k_size /4 + k/4 + threadIdx.x];
        reinterpret_cast<float4*>(&(b_shared[!pingpong][threadIdx.y][threadIdx.x * 4]))[0] = reinterpret_cast<float4*>(b)[(k + threadIdx.y) * n_size/4 + x];
        for (int j = 0; j < 4; j++){
            for (int i = 0; i < bk; i++){
                temp[j] += a_shared[pingpong][threadIdx.y][i] * b_shared[pingpong][i][threadIdx.x * 4 + j];
            }
        }
        pingpong = !pingpong;
        __syncthreads();
    }
    for (int j = 0; j < 4; j++){
        for (int i = 0; i < bk; i++){
            temp[j] += a_shared[pingpong][threadIdx.y][i] * b_shared[pingpong][i][threadIdx.x * 4 + j];
        }
    }
    reinterpret_cast<float4*>(&c[y * n_size + x * 4])[0] = reinterpret_cast<float4*>(&temp[0])[0];
}

int main(){
    size_t m = 512;
    size_t n = 1024;
    size_t k = 2048;
    const size_t A_matrix_size = m * k;
    const size_t B_matrix_size = k * n;
    const size_t C_matrix_size = m * n;

    float* A_host = new float[A_matrix_size];
    float* B_host = new float[B_matrix_size];
    float* C_host_result = new float[C_matrix_size];
    float* A_dev = new float;
    float* B_dev = new float;
    float* C_dev = new float;
    float* C_dev_result = new float[C_matrix_size];

    util::init_data(A_host, A_matrix_size);
    init_data(B_host, B_matrix_size);
    host_sgemm(A_host, B_host, C_host_result, m, n, k);

    cudaMalloc((void**)&A_dev, A_matrix_size * sizeof(float));
    cudaMalloc((void**)&B_dev, B_matrix_size * sizeof(float));
    cudaMalloc((void**)&C_dev, C_matrix_size * sizeof(float));
    cudaMemcpy(A_dev, A_host, A_matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_dev, B_host, B_matrix_size * sizeof(float), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < 3; i++){
        constexpr int BLOCK_X_SIZE = 16;
        constexpr int BLOCK_Y_SIZE = 16;

        Perf();
        dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid(n / BLOCK_X_SIZE, m / BLOCK_Y_SIZE);
        dev_sgemm<<<grid, block>>>(A_dev, B_dev, C_dev, m, n, k);
        cudaDeviceSynchronize(); 

        cudaMemcpy(C_dev_result, C_dev, C_matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        compare_result(C_host_result, C_dev_result, C_matrix_size);
        cudaMemset(C_dev, 0, C_matrix_size * sizeof(float));
    }

    for (size_t i = 0; i < 3; i++){
        constexpr int BLOCK_X_SIZE = 16;
        constexpr int BLOCK_Y_SIZE = 16;
        constexpr int BLOCK_K_SIZE = 16;

        Perf();
        dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid(n / BLOCK_X_SIZE, m / BLOCK_Y_SIZE);
        dev_sgemm_sharedmem<BLOCK_X_SIZE, BLOCK_Y_SIZE, BLOCK_K_SIZE><<<grid, block>>>(A_dev, B_dev, C_dev, m, n, k);
        cudaDeviceSynchronize(); 

        cudaMemcpy(C_dev_result, C_dev, C_matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        compare_result(C_host_result, C_dev_result, C_matrix_size);
        cudaMemset(C_dev, 0, C_matrix_size * sizeof(float));
    }

    for (size_t i = 0; i < 3; i++){
        constexpr int BLOCK_X_SIZE = 8;
        constexpr int BLOCK_Y_SIZE = 32;
        constexpr int BLOCK_K_SIZE = 16;

        Perf();
        dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid(n / BLOCK_X_SIZE / 4, m / BLOCK_Y_SIZE);
        dev_sgemm_float4_improved<0, 0, BLOCK_K_SIZE*2><<<grid, block>>>(A_dev, B_dev, C_dev, m, n, k);
        cudaDeviceSynchronize(); 

        cudaMemcpy(C_dev_result, C_dev, C_matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        compare_result(C_host_result, C_dev_result, C_matrix_size);
        cudaMemset(C_dev, 0, C_matrix_size * sizeof(float));
    }

    for (size_t i = 0; i < 3; i++){
        constexpr int BLOCK_X_SIZE = 8;
        constexpr int BLOCK_Y_SIZE = 32;
        constexpr int BLOCK_K_SIZE = 16;

        Perf();
        dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid(n / BLOCK_X_SIZE / 4, m / BLOCK_Y_SIZE);
        dev_sgemm_double_buffer<0, 0, BLOCK_K_SIZE*2><<<grid, block>>>(A_dev, B_dev, C_dev, m, n, k);
        cudaDeviceSynchronize(); 

        cudaMemcpy(C_dev_result, C_dev, C_matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        compare_result(C_host_result, C_dev_result, C_matrix_size);
        cudaMemset(C_dev, 0, C_matrix_size * sizeof(float));
    }

    for (size_t i = 0; i < 3; i++){
        constexpr int BLOCK_X_SIZE = 16;
        constexpr int BLOCK_Y_SIZE = 16;
        constexpr int BLOCK_K_SIZE = 16;

        Perf();
        dim3 block(BLOCK_X_SIZE, BLOCK_Y_SIZE);
        dim3 grid(n / BLOCK_X_SIZE / 4, m / BLOCK_Y_SIZE / 4);
        dev_sgemm_float4_outter_product<0, 0, BLOCK_K_SIZE*4><<<grid, block>>>(A_dev, B_dev, C_dev, m, n, k);
        cudaDeviceSynchronize(); 

        cudaMemcpy(C_dev_result, C_dev, C_matrix_size * sizeof(float), cudaMemcpyDeviceToHost);
        compare_result(C_host_result, C_dev_result, C_matrix_size);
        cudaMemset(C_dev, 0, C_matrix_size * sizeof(float));
    }



    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(C_dev);
    return 0;
}
