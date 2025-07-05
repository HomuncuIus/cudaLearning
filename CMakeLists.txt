cmake_minimum_required(VERSION 3.28.0)
project(cuda_learning VERSION 0.1.0 LANGUAGES CUDA CXX C)

add_subdirectory(0_device)
add_subdirectory(1_grid_block_and_thread)
