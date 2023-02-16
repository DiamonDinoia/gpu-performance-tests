#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "utils.h"

constexpr auto ITERATIONS = 5;

// This function generates an array of random numbers on the host
// copies it to the device and prints it
// and frees the meemory on the device
Element* generateRandomNumbersOnDevice(RandomNumberGenerator& generator,
                                       const unsigned long    N) {
    // create a random number generator
    // generate an array of random numbers
    std::vector<Element> vector = generator.getShuffledVector(N);

    // copy the array to the device
    Element* deviceArray;
    CUDA_CALL(cudaMalloc(&deviceArray, vector.size() * sizeof(Element)));
    CUDA_CALL(cudaMemcpy(deviceArray, vector.data(),
                         vector.size() * sizeof(Element),
                         cudaMemcpyHostToDevice));
    return deviceArray;
}

// this function frees the memory on the device
template <typename T>
void freeMemoryOnDevice(T* deviceArray) {
    CUDA_CALL(cudaFree(deviceArray));
}

// this kernel reads an array of random numbers from the device
// it is designed to do random memory accesses
__global__ void latencyKernel(const Element* array, const unsigned long N,
                              const unsigned long reads,
                              unsigned long*      result) {
    // get the thread id
    unsigned long index = (blockIdx.x * blockDim.x + threadIdx.x) % N;

    // iterate over the reads
    for (unsigned long i = 0; i < reads; ++i) { index = array[index].index; }
    result[0] = index;
}

// this kernel invalidates the cache of the GPU
__global__ void invalidateCacheKernel(unsigned long*      array,
                                      const unsigned long N) {
    // get the thread id
    unsigned long index = (blockIdx.x * blockDim.x + threadIdx.x) % N;
    // read the array
    array[index] = 0;
}

// this function creates an array and uses it to invalidate the GPU cache and
// the frees it
void invalidateCache() {
    const unsigned long blockCount  = 256;
    const unsigned long threadCount = 256;
    const unsigned long N           = blockCount * threadCount;
    // create an array
    unsigned long* deviceArray;
    CUDA_CALL(cudaMalloc(&deviceArray, 100 * sizeof(unsigned long)));
    // call the kernel
    invalidateCacheKernel<<<blockCount, threadCount>>>(deviceArray, N);
    // wait for the kernel to finish
    CUDA_CALL(cudaDeviceSynchronize());
    // free the memory
    CUDA_CALL(cudaFree(deviceArray));
}

// this function call latencyKernel kernel
// it also measures the time it takes to execute the kernel
// and returns the time in milliseconds
double measureLatency(const Element* deviceArray, const unsigned long N,
                      const unsigned long reads,
                      const unsigned long threadCount,
                      const unsigned long blockCount,
                      unsigned long*      resultArray) {
    invalidateCache();
    CUDA_CALL(cudaDeviceSynchronize());
    // create an event to measure the time
    auto start = std::chrono::high_resolution_clock::now();
    // call the kernel
    latencyKernel<<<blockCount, threadCount>>>(deviceArray, N, reads,
                                               resultArray);
    // wait for the kernel to finish
    CUDA_CALL(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    // compute the time in nanoseconds
    double duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
            .count();
    return duration;
}

// this function call measureLatency ITERATIONS times and computes the average
// time then it prints it and returns it
double measureLatencyAndPrint(const Element* deviceArray, const unsigned long N,
                              const unsigned long reads,
                              const unsigned long threadCount,
                              const unsigned long blockCount,
                              unsigned long*      resultArray) {
    // measure the time
    double sum = 0;
    for (unsigned int i = 0; i < ITERATIONS; ++i) {
        sum += measureLatency(deviceArray, N, reads, threadCount, blockCount,
                              resultArray);
    }
    // compute the average time
    double averageTime = sum / ITERATIONS;
    // print the average time
    std::cout << "Average time: " << averageTime << " ms" << std::endl;
    return averageTime;
}

// this function creates a csv file with the header
void createCSVFile(const std::string& OUTPUT_CSV) {
    // open the file
    std::ofstream file;
    file.open(OUTPUT_CSV);
    // write the header
    file << "N,reads,threadCount,blockCount,latency" << std::endl;
    // close the file
    file.close();
}

// this function appends size, threeadCount, blockCount, reads and latency to
// the a csv file
void appendToCSV(const double N, const unsigned long reads,
                 const unsigned long threadCount,
                 const unsigned long blockCount, const double latency,
                 const std::string& OUTPUT_CSV) {
    // open the file
    std::ofstream file;
    file.open(OUTPUT_CSV, std::ios_base::app);
    // write the data
    file << N << "," << reads << "," << threadCount << "," << blockCount << ","
         << latency << std::endl;
    // close the file
    file.close();
}

// this function calls generateRandomNumbersOnDevice, measureLatencyAndPrint and
// freeMemoryOnDevice
void runTest(const unsigned long N, const unsigned long reads,
             const unsigned long threadCount, const unsigned long blockCount,
             const std::string& OUTPUT_CSV) {
    const auto mem_block_size = operator""_KB(N);
    // Each memory access fetches a cache line
    const auto num_nodes = mem_block_size / kCachelineSize;
    // create a random number generator
    RandomNumberGenerator generator;
    // generate random numbers on the device
    std::cout
        << "Generating random numbers on the CPU and copying them to the GPU"
        << std::endl;
    Element* deviceArray = generateRandomNumbersOnDevice(generator, num_nodes);

    // allocate memory for the result
    unsigned long* resultArray;
    CUDA_CALL(cudaMalloc(&resultArray, sizeof(unsigned long)));
    // measure the latency
    std::cout << "Measuring latency" << std::endl;
    auto time     = measureLatencyAndPrint(deviceArray, num_nodes, reads,
                                           threadCount, blockCount, resultArray);

    auto readTime = time / reads;
    std::cout << "Average memory read time: " << readTime << " ns" << std::endl;

    // copy the result from the device to the host
    unsigned long result;
    CUDA_CALL(cudaMemcpy(&result, resultArray, sizeof(unsigned long),
                         cudaMemcpyDeviceToHost));
    std::cout << "Result: " << result << std::endl;

    // free the memory on the device
    std::cout << "Freeing memory on the device" << std::endl;
    freeMemoryOnDevice(deviceArray);
    // frees the result array
    freeMemoryOnDevice(resultArray);
    // if the output csv does not exist it calls createCSVFile
    if (!std::filesystem::exists(OUTPUT_CSV)) { createCSVFile(OUTPUT_CSV); }
    // appends the data to the csv file
    appendToCSV(mem_block_size * 64. / (1024 * 1024), reads, threadCount,
                blockCount, readTime, OUTPUT_CSV);
}
