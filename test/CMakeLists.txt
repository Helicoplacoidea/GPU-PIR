function(configure_pir_target _name _enable_lineinfo)
    target_link_libraries(${_name} emp-ot::emp-ot emp-tool::emp-tool)
    target_include_directories(${_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/mpc_keys
        /usr/local/cuda/include
    )
    target_compile_definitions(${_name} PRIVATE
        "NUM_STREAMS=10"
        "NUM_CHUNKS=32768"
    )
    if(_enable_lineinfo)
        target_compile_options(${_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
    endif()
endfunction()

macro (add_test_executable _name)
    file(GLOB CUSOURCES CONFIGURE_DEPENDS "${_name}*.cu")
    file(GLOB SOURCES CONFIGURE_DEPENDS "${_name}.cpp")
    file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/mpc_cuda/*.cu")

    add_executable(${_name} ${SOURCES} ${CUDA_SOURCES} ${CUSOURCES})
    configure_pir_target(${_name} TRUE)
endmacro()

macro (add_bench_executable _name)
    file(GLOB BENCH_CUSOURCES CONFIGURE_DEPENDS "${_name}*.cu")
    file(GLOB BENCH_SOURCES CONFIGURE_DEPENDS "${_name}.cpp")
    file(GLOB CUDA_SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/mpc_cuda/*.cu")

    add_executable(${_name} ${BENCH_SOURCES} ${CUDA_SOURCES} ${BENCH_CUSOURCES})
    configure_pir_target(${_name} FALSE)
endmacro()

macro (add_test_case_with_run _name)
    add_test_executable(${_name})
    add_test(NAME ${_name} COMMAND $<TARGET_FILE:${_name}> WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_bench_case_with_run _name)
    add_bench_executable(${_name})
    add_test(NAME ${_name} COMMAND $<TARGET_FILE:${_name}> WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()

macro (add_plain_test_case _name)
    add_test_executable(${_name})
    add_test(NAME ${_name} COMMAND $<TARGET_FILE:${_name}> WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endmacro()


# Test cases
add_test_case_with_run(test_pir)
add_test(NAME test_pir_small COMMAND $<TARGET_FILE:test_pir> 12 33 WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
add_plain_test_case(test_pir_utils)
add_plain_test_case(test_fused_eval)
set_tests_properties(test_fused_eval PROPERTIES SKIP_RETURN_CODE 77)

# Bench cases
add_bench_case_with_run(bench_pir)

# Sector-view latency study (build only; run explicitly)
if(EXISTS "${CMAKE_SOURCE_DIR}/test/bench_sector_view.cu")
    add_bench_executable(bench_sector_view)
endif()

# Low-latency / high-throughput serving study (build only; run explicitly)
if(EXISTS "${CMAKE_SOURCE_DIR}/test/bench_serving.cpp")
    add_bench_executable(bench_serving)
endif()

# LUT-only bench
if(EXISTS "${CMAKE_SOURCE_DIR}/test/bench_lut_only.cpp")
    add_bench_executable(bench_lut_only)
    add_test(NAME bench_lut_only COMMAND $<TARGET_FILE:bench_lut_only> WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/")
endif()
