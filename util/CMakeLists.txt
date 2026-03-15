add_library(util STATIC
    util.cpp
)


# # 设置包含目录 ，感觉可能写错了
target_include_directories(util
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/
)
