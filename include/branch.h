#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>

#include "utils.h"

constexpr auto  RUNS       = 5;
constexpr auto  ITERATIONS = 16384 << 2;

__global__ void branch_kernel(int* result, const int y, const int z,
                              const int iterations1, const int iterations2,
                              const bool* random) {
    int a   = 0;
    int b   = 0;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) % 32;

    if (random[idx]) {
        for (int i = 0; i < iterations1; i++) {
            a = a & y;
            a = a | z;
        }
    } else {
        for (int i = 0; i < iterations2; i++) {
            b = b | y;
            b = b & z;
        }
    }
    result[idx] = a + b;
}

// this function uses getRandomBoolVector to generate a random boolean vector
// then copies it to the device
// the parameters are the RandomNumberGenerator and returns the device pointer
// to the random vector
bool* getRandomBoolVectorDevice(RandomNumberGenerator& rng, const double p) {
    const auto random = rng.getRandomBoolVector(p);
    // print the random vector
    std::cout << "Random[] = ";
    for (int i = 0; i < 32; i++) { std::cout << random[i] << ", "; }
    std::cout << std::endl;
    bool* d_random;
    CUDA_CALL(cudaMalloc(&d_random, sizeof(bool) * random.size()));
    CUDA_CALL(cudaMemcpy(d_random, random.data(), sizeof(bool) * random.size(),
                         cudaMemcpyHostToDevice));
    return d_random;
}

// this function call branch_kernel ITERATIONS times with the random vector and
// measures the average execution time
double runBranchKernel(const bool* d_random) {
    const int  y         = 0xAAAAAAAA;
    const int  z         = 0x55555555;

    const auto blocksize = 64;
    const auto size      = blocksize * 1048576;
    const auto nBytes    = sizeof(int) * size;

    dim3       block(blocksize, 1);
    dim3       grid(size / block.x, 1);

    int*       d_result;
    CUDA_CALL(cudaMalloc(&d_result, nBytes));

    // warm up
    branch_kernel<<<grid, block>>>(d_result, y, z, ITERATIONS, ITERATIONS,
                                   d_random);
    CUDA_CALL(cudaDeviceSynchronize());

    // measure
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < RUNS; i++) {
        branch_kernel<<<grid, block>>>(d_result, y, z, ITERATIONS, ITERATIONS,
                                       d_random);
    }
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    // execution time
    const auto execution_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count() /
        RUNS;
    // free memory
    CUDA_CALL(cudaFree(d_result));
    return execution_time;
}

// this function creates a csv file if noit exist and populates it with a header
// containing the p values and the execution time if the file exists it appends
// the execution time to the end of the file
void writeToFile(const double p, const double execution_time) {
    std::ofstream file;
    file.open("branch.csv", std::ios::app);
    if (file.tellp() == 0) { file << "p,execution_time" << std::endl; }
    file << p << "," << execution_time << std::endl;
    file.close();
}

// this function has p and seed as parameters and calls
// getRandomBoolVectorDevice then runBranchKernel to measure the execution time
// of branch_kernel
void runBranchKernel(const double p, RandomNumberGenerator& generator) {
    const auto d_random       = getRandomBoolVectorDevice(generator, p);
    const auto execution_time = runBranchKernel(d_random);
    CUDA_CALL(cudaFree(d_random));
    std::cout << "Execution time for p = " << p << " is " << execution_time
              << " ms" << std::endl;
    writeToFile(p, execution_time);
}
