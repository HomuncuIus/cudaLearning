#include <stdio.h>

int main(void){
    int iDeviceCount = 0;
    cudaGetDeviceCount(&iDeviceCount);
    printf("there are %d devices available\n", iDeviceCount);
    return 0;
}