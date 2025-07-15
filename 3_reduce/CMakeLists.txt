find_package(CUDAToolkit)
add_executable(baseline baseline.cu)
add_executable(baseline_official baseline_official.cu)
add_executable(optimised1_shared_memory optimised1_shared_memory.cu)
add_executable(optimised2_warp_no_divergence optimised2_warp_no_divergence.cu)
add_executable(optimised3_fix_bank_conflict optimised3_fix_bank_conflict.cu)
add_executable(optimised4_1_reduce_thread optimised4_1_reduce_thread.cu)
add_executable(optimised4_2_reduce_block optimised4_2_reduce_block.cu)
add_executable(optimised5_reduce_sync optimised5_reduce_sync.cu)
add_executable(optimised6_warp_shuffle optimised6_warp_shuffle.cu)
 
# target_link_libraries(baseline PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
# target_compile_options(baseline PRIVATE -g -G) # 配置调试信息