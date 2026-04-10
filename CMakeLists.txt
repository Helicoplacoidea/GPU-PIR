cmake_minimum_required (VERSION 3.22.1)
project (fss-cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr -diag-suppress=20012 -DUSE_L2_CACHE")
find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/emp-base.cmake)

FIND_PACKAGE(emp-ot REQUIRED)
INCLUDE_DIRECTORIES(${EMP-OT_INCLUDE_DIRS})

FIND_PACKAGE(OpenSSL REQUIRED)

ENABLE_TESTING()
ADD_SUBDIRECTORY(test)

set(CMAKE_CUDA_STANDARD 16) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++17 -arch=sm_89")  

# set(CMAKE_CUDA_FLAGS "-G  -O0 ")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_VERBOSE_MAKEFILE ON)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
endif()


#sudo /usr/local/cuda/bin/ncu --nvtx --call-stack --set full -f --export test.ncu-rep ./bin/test_pir
#
