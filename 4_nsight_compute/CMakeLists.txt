# find_package(CUDAToolkit)
add_executable(add add.cu)
target_compile_options(add PRIVATE -lineinfo)

add_executable(add_dev_only add_dev_only.cu)
target_compile_options(add_dev_only PRIVATE -lineinfo)