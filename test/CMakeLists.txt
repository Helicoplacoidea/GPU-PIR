set(CMAKE_CUDA_ARCHITECTURES 89) 

macro (add_test_executable _name)
    file(GLOB CUSOURCES "${_name}*.cu")
    file(GLOB SOURCES "${_name}.cpp")
    file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/mpc_cuda/*.cu")

    add_executable(${_name} ${SOURCES} ${CUDA_SOURCES} ${CUSOURCES})
    target_link_libraries(${_name} ${EMP-OT_LIBRARIES})
    target_include_directories(${_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/mpc_keys
        /usr/local/cuda/include
    )
    target_compile_definitions(${_name} PRIVATE
        "entry_size=16"
        "NUM_STREAMS=10"
        "NUM_CHUNKS=32768")
    target_compile_options(${_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    )
endmacro()

macro (add_bench_executable _name)
    file(GLOB BENCH_CUSOURCES "${_name}*.cu")
    file(GLOB BENCH_SOURCES "${_name}.cpp")
    file(GLOB CUDA_SOURCES "${CMAKE_SOURCE_DIR}/mpc_cuda/*.cu")

    add_executable(${_name} ${BENCH_SOURCES} ${CUDA_SOURCES} ${BENCH_CUSOURCES})
    target_link_libraries(${_name} ${EMP-OT_LIBRARIES})
    target_include_directories(${_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/mpc_keys
        /usr/local/cuda/include
    )
    target_compile_definitions(${_name} PRIVATE
        "entry_size=16"
        "NUM_STREAMS=10"
        "NUM_CHUNKS=32768")
endmacro()

macro (add_test_case_with_run _name)
    add_test_executable(${_name})
    add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_bench_case_with_run _name)
    add_bench_executable(${_name})
    add_test(NAME ${_name} COMMAND "./run" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_plain_test_case _name)
    add_test_executable(${_name})
    add_test(NAME ${_name} COMMAND "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${_name}" WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()


# Test cases
add_test_case_with_run(test_pir)
add_plain_test_case(test_pir_utils)

# Bench cases
add_bench_case_with_run(bench_pir)
