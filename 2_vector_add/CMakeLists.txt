add_executable(add add.cu)
target_compile_options(add PRIVATE -g -G) # 配置调试信息

add_executable(add2 add_optimised.cu)
add_executable(matmul matmul.cu)