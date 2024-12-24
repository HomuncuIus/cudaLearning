find_package(CUDAToolkit)
# target_link_libraries()
add_executable(baseline baseline.cu)
target_link_libraries(baseline PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})

# cmake_minimum_required(VERSION 3.20.0)
# project(cuda_learning VERSION 0.1.0 LANGUAGE CUDA CXX C)
# add_subdirectory(1_reduce)

