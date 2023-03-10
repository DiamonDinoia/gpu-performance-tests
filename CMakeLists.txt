cmake_minimum_required(VERSION 3.16)
project(MemoryTest  VERSION 1.0 LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED) 

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


set(CPM_DOWNLOAD_VERSION 0.36.0)
if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()

if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
  NAME samples
  GITHUB_REPOSITORY nvidia/cuda-samples
  VERSION 12.1
  DOWNLOAD_ONLY TRUE
  GIT_SHALLOW TRUE
)

add_executable(bandwidth-test bandwidth.cu)
target_compile_features(bandwidth-test PRIVATE cxx_std_17 cuda_std_17)
target_include_directories(bandwidth-test PRIVATE ${samples_SOURCE_DIR}/Common)

message(STATUS {${CUDAToolkit_INCLUDE_DIRS}})