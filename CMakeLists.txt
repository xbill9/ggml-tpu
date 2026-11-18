cmake_minimum_required(VERSION 3.18)
project(ggml-tpu CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_CPP_DIR "$ENV{HOME}/llama.cpp" CACHE PATH "llama.cpp source tree")
set(LLAMA_CPP_BUILD "${LLAMA_CPP_DIR}/build-tpu" CACHE PATH "llama.cpp build tree (BUILD_SHARED_LIBS=ON)")
set(PYTHON_EXE "python3" CACHE STRING "Python interpreter whose JAX the backend embeds")

execute_process(COMMAND ${PYTHON_EXE} -c "import sysconfig,sys;print(sysconfig.get_paths()['include']);print(sysconfig.get_config_var('LIBDIR'));print(sysconfig.get_config_var('LDLIBRARY'));print(sys.base_prefix)"
                OUTPUT_VARIABLE PY_INFO OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" PY_INFO "${PY_INFO}")
list(GET PY_INFO 0 PY_INCLUDE)
list(GET PY_INFO 1 PY_LIBDIR)
list(GET PY_INFO 2 PY_LDLIBRARY)
list(GET PY_INFO 3 PY_HOME)
message(STATUS "ggml-tpu: embedding Python from ${PY_HOME} (${PY_LIBDIR}/${PY_LDLIBRARY})")

find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)

add_library(ggml-tpu SHARED
    src/ggml-tpu.cpp
    src/lower.cpp
    src/hlo.cpp
    src/exec_jax.cpp
    src/exec_pjrt.cpp)

target_include_directories(ggml-tpu PRIVATE
    include
    "${LLAMA_CPP_DIR}/ggml/include"
    "${LLAMA_CPP_DIR}/ggml/src"
    "${PY_INCLUDE}")

target_compile_definitions(ggml-tpu PRIVATE
    GGML_BACKEND_DL GGML_BACKEND_BUILD GGML_BACKEND_SHARED GGML_SHARED
    GGML_TPU_PYTHON_HOME="${PY_HOME}"
    GGML_TPU_PYTHON_LIB="${PY_LIBDIR}/${PY_LDLIBRARY}")

target_link_libraries(ggml-tpu PRIVATE "${GGML_BASE_LIB}" "${PY_LIBDIR}/${PY_LDLIBRARY}" ${CMAKE_DL_LIBS})
set_target_properties(ggml-tpu PROPERTIES BUILD_RPATH "${PY_LIBDIR};${LLAMA_CPP_BUILD}/bin")

# tpu-compare: CPU vs TPU logits on one model
find_library(LLAMA_LIB llama PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB ggml PATHS "${LLAMA_CPP_BUILD}/bin" NO_DEFAULT_PATH REQUIRED)
add_executable(tpu-compare tools/compare.cpp)
target_include_directories(tpu-compare PRIVATE "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(tpu-compare PRIVATE "${LLAMA_LIB}" "${GGML_LIB}" "${GGML_BASE_LIB}")
set_target_properties(tpu-compare PROPERTIES BUILD_RPATH "${LLAMA_CPP_BUILD}/bin")
