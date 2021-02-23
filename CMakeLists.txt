cmake_minimum_required(VERSION 3.5.1)
project(mininn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 -fopenmp")

include_directories("/usr/local/include" "/usr/local/opt/llvm/include")
link_directories("/usr/local/lib" "/usr/local/opt/llvm/lib")

set(APP_INCLUDES
        include/
        include/minidnn
 )

#Include the directories and now your cpp files will recognize your headers
include_directories(${APP_INCLUDES})

add_executable(mininn main.cpp)
target_sources(mininn PUBLIC mnist.cc)
TARGET_LINK_LIBRARIES(mininn
        pthread
        #"${TORCH_LIBRARIES}"
        )


