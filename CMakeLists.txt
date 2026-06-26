cmake_minimum_required(VERSION 3.22.1)

# Auto-detect GPU architecture BEFORE project() so it takes effect.
# CMake's project() with CUDA language initializes CMAKE_CUDA_ARCHITECTURES
# to "native" by default, which can be unreliable in some environments.
# Detection order:
#   1. User-supplied CMAKE_CUDA_ARCHITECTURES (via -D flag)
#   2. nvidia-smi query (asks the driver for the actual hardware compute capability)
#   3. Fallback: native (CMake's default compiler probing)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader,nounits
            OUTPUT_VARIABLE _GPU_COMPUTE_CAPS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(_GPU_COMPUTE_CAPS)
            # Take the first GPU's compute capability (e.g. "9.0" -> "90")
            string(REPLACE "\n" ";" _CAP_LIST "${_GPU_COMPUTE_CAPS}")
            list(GET _CAP_LIST 0 _FIRST_CAP)
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_FIRST_CAP}")
            message(STATUS "Auto-detected GPU architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")
        else()
            set(CMAKE_CUDA_ARCHITECTURES native)
            message(STATUS "nvidia-smi found but returned no GPU info, falling back to native detection")
        endif()
    else()
        set(CMAKE_CUDA_ARCHITECTURES native)
        message(STATUS "nvidia-smi not found, falling back to native detection")
    endif()
endif()

project(fss-cuda LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(emp-tool REQUIRED)
find_package(emp-ot REQUIRED)

add_compile_definitions(USE_L2_CACHE)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

enable_testing()
add_subdirectory(test)