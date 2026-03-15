#include <random>
#include <iostream>

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
        printf("The time is %f\n", elapsed_time);
    }
private:
    cudaEvent_t m_start, m_end;
};



void init_data(float* input, size_t length){
    // 创建随机设备作为种子
    std::random_device rd;
    // 使用 Mersenne Twister 算法生成器
    std::mt19937 gen(rd());
    // 创建一个 [0.0, 1.0) 的浮点均匀分布
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < length; i++){
        input[i] = dist(gen);
        // input[i] = i;
    }
}

void print_data(float* input, int high, int width){
    for (size_t m = 0; m < high; m++){
        for (size_t n = 0; n < width; n++){
            printf("%f, ", input[m * width + n]);
        }
        printf("\n");
    }
}

void host_transpose(float* input, float* output, int high, int width){
    for (size_t m = 0; m < high; m++){
        for (size_t n = 0; n < width; n++){
            output[n * high + m] = input[m * width + n];
        }
    }
}

void compare_result(float* res1, float* res2, size_t length){
    float total_error = 0;
    for (size_t i = 0; i < length; i++){
        total_error += abs(res1[i] - res2[i]);
        // printf("The error on index %zu is %f, res1 %f, res2 %f \n", i, abs(res1[i] - res2[i]), res1[i], res2[i]);
    }
    printf("The error on is %f\n", total_error);
}

__global__ void device_transpose(float* input, float* output, int high, int width){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // int input_offset = x * width + y;
    // int output_offset = y * high + x;
    int input_offset = y * width + x;
    int output_offset = x * high + y;
    // printf("dev: x %d %d %d %d y %d %d %d %d offset %d %d  \n", x, blockIdx.x, blockDim.x, threadIdx.x, y, blockIdx.y, blockDim.y, threadIdx.y, input_offset, output_offset);
    // printf("dev:  %f %f \n", output[output_offset],input[input_offset]);
    output[output_offset] = input[input_offset];
    // printf("dev:  %d %d \n", output_offset,input_offset);
}



int main(){

    int high = 2048;
    int width = 512;
    // int high = 64;
    // int width = 16;
    int total_size = high * width;
    int nBytes = total_size * sizeof(float);
    float* host_input = new float[total_size];
    float* host_output = new float[total_size];
    float* result_from_dev = new float[total_size];

    init_data(host_input, total_size);
    host_transpose(host_input, host_output, high, width);
    // print_data(host_output, width, high);

    // TODO 注意，dev_input dev_output这种dev内存指针声明时这种写法是错误的，
    // 只要float* dev_input = new float;就行。不需要数组
    float* dev_input = new float[total_size];
    float* dev_output = new float[total_size];
    cudaMalloc((void**)&dev_input, nBytes);
    cudaMalloc((void**)&dev_output, nBytes);
    cudaMemcpy(dev_input, host_input, nBytes, cudaMemcpyHostToDevice);
    
    for (size_t i = 0; i < 5; i++){
        Perf();
        dim3 block(8, 32);
        dim3 grid(width / block.x, high / block.y);
        device_transpose<<<grid, block>>>(dev_input, dev_output, high, width);
        // cudaError_t err = cudaGetLastError();  // 检查 launch 是否成功
        // if (err != cudaSuccess) {
        //     printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        // }
        cudaDeviceSynchronize(); 

        cudaMemcpy(result_from_dev, dev_output, nBytes, cudaMemcpyDeviceToHost);
        // print_data(result_from_dev, width, high);
        compare_result(host_output, result_from_dev, total_size);
    }
    
    // compare_result(host_output, result_from_dev, total_size);
    

    return 0;
}
