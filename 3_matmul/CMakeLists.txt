add_executable(matmul matmul.cu)
target_compile_options(matmul PRIVATE -g -G) # 配置调试信息
