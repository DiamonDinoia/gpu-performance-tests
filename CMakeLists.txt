cmake_minimum_required(VERSION 3.16)
project(MemoryTest  VERSION 1.0 LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()


include(FindCUDA/select_compute_arch)
CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
string(REPLACE "." "" INSTALLED_GPU_CCS_4 "${INSTALLED_GPU_CCS_3}")
string(REPLACE "+PTX" "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_4}")

set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

include_directories(include)

add_executable(latency-test latency.cu)
target_compile_features(latency-test PRIVATE cxx_std_17 cuda_std_17) 

add_executable(shared-test shared.cu)
target_compile_features(shared-test PRIVATE cxx_std_17 cuda_std_17)

add_executable(branch-test branch.cu)
target_compile_features(branch-test PRIVATE cxx_std_17 cuda_std_17)
