# find_package(CUDAToolkit)
add_executable(transpose transpose.cu)
target_compile_options(transpose PRIVATE -lineinfo)