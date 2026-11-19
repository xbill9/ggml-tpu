cmake_minimum_required(VERSION 3.18)
project(ggml-tpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_CPP_DIR "$ENV{HOME}/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_DIR}/build-tpu" CACHE PATH "llama.cpp build tree (BUILD_SHARED_LIBS=ON)")
set(PYTHON_EXE "python3" CACHE STRING "Python interpreter: embedded by the JAX executor, and used at build time to generate compile options")
option(GGML_TPU_JAX "build the JAX executor (embeds Python)" ON)
set(GGML_TPU_RPATH "${LLAMA_CPP_BUILD}/bin" CACHE STRING "where libggml-tpu.so finds libggml-base ($ORIGIN for a relocatable build)")

find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)

set(TPU_SOURCES src/ggml-tpu.cpp src/lower.cpp src/hlo.cpp src/exec_pjrt.cpp)
if (GGML_TPU_JAX)
    list(APPEND TPU_SOURCES src/exec_jax.cpp)
endif()
add_library(ggml-tpu SHARED ${TPU_SOURCES})

# XLA compile options for the PJRT executor, serialized at build time
add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/compile_options.inc"
    COMMAND ${PYTHON_EXE} "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_compile_options.py" "${CMAKE_CURRENT_BINARY_DIR}/compile_options.inc"
    DEPENDS tools/gen_compile_options.py
    COMMENT "Generating XLA compile options")
add_custom_target(ggml-tpu-compile-options DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/compile_options.inc")
add_dependencies(ggml-tpu ggml-tpu-compile-options)

target_include_directories(ggml-tpu PRIVATE
    include
    third_party
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src")
target_compile_definitions(ggml-tpu PRIVATE GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED GGML_SHARED)
target_link_libraries(ggml-tpu PRIVATE "${GGML_BASE_LIB}" ${CMAKE_DL_LIBS})
set_target_properties(ggml-tpu PROPERTIES BUILD_RPATH "${GGML_TPU_RPATH}")

if (GGML_TPU_JAX)
    execute_process(COMMAND ${PYTHON_EXE} -c "import sysconfig,sys;print(sysconfig.get_paths()['include']);print(sysconfig.get_config_var('LIBDIR'));print(sysconfig.get_config_var('LDLIBRARY'));print(sys.base_prefix)"
                    OUTPUT_VARIABLE PY_INFO OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    string(REPLACE "\n" ";" PY_INFO "${PY_INFO}")
    list(GET PY_INFO 0 PY_INCLUDE)
    list(GET PY_INFO 1 PY_LIBDIR)
    list(GET PY_INFO 2 PY_LDLIBRARY)
    list(GET PY_INFO 3 PY_HOME)
    message(STATUS "ggml-tpu: JAX executor embeds Python from ${PY_HOME} (${PY_LIBDIR}/${PY_LDLIBRARY})")
    target_include_directories(ggml-tpu PRIVATE "${PY_INCLUDE}")
    target_compile_definitions(ggml-tpu PRIVATE GGML_TPU_WITH_JAX
        GGML_TPU_PYTHON_HOME="${PY_HOME}"
        GGML_TPU_PYTHON_LIB="${PY_LIBDIR}/${PY_LDLIBRARY}")
    target_link_libraries(ggml-tpu PRIVATE "${PY_LIBDIR}/${PY_LDLIBRARY}")
    set_target_properties(ggml-tpu PROPERTIES BUILD_RPATH "${PY_LIBDIR};${GGML_TPU_RPATH}")
endif()

# tpu-compare: CPU vs TPU logits on one model
find_library(LLAMA_LIB llama PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB ggml PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
add_executable(tpu-compare tools/compare.cpp)
target_include_directories(tpu-compare PRIVATE "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(tpu-compare PRIVATE "${LLAMA_LIB}" "${GGML_LIB}" "${GGML_BASE_LIB}")
set_target_properties(tpu-compare PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}/bin")

# tpu-regress: graph patterns that broke the backend, CPU vs TPU through ggml_backend_sched
add_executable(tpu-regress tools/regress.cpp)
target_include_directories(tpu-regress PRIVATE "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(tpu-regress PRIVATE "${GGML_LIB}" "${GGML_BASE_LIB}")
set_target_properties(tpu-regress PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}/bin")
