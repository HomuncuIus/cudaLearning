#include <cstdio>
#include <random>
#include "util.h"

namespace util {

void init_data(float* input, size_t length){
    // 创建随机设备作为种子
    std::random_device rd;
    // 使用 Mersenne Twister 算法生成器
    std::mt19937 gen(rd());
    // 创建一个 [0.0, 1.0) 的浮点均匀分布
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < length; i++){
        input[i] = dist(gen);
        // input[i] = 1;
        // input[i] = i;
    }
}

void print_data(float* input, size_t high, size_t width){
    for (size_t m = 0; m < high; m++){
        for (size_t n = 0; n < width; n++){
            printf("%f, ", input[m * width + n]);
        }
        printf("\n");
    }
}

void compare_result(float* res1, float* res2, size_t length){
    float total_error = 0;
    for (size_t i = 0; i < length; i++){
        total_error += fabs(res1[i] - res2[i]) > 1e-4? fabs(res1[i] - res2[i]) : 0;
        // if (i < 100){
        //     printf("The error on index %zu is %.9g, res1 %.9g, res2 %.9g \n", i, fabs(res1[i] - res2[i]), res1[i], res2[i]);
        // }
        // printf("The error on index %zu is %.9g, res1 %.9g, res2 %.9g \n", i, abs(res1[i] - res2[i]), res1[i], res2[i]);
    }
    printf("The error is %.9g\n", total_error);
}

}