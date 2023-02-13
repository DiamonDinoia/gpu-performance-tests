#include <XoshiroCpp.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

// this macro checks if a cuda function call was successful
#define CUDA_CALL(x)                                        \
    do {                                                    \
        if ((x) != cudaSuccess) {                           \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(x);                                        \
        }                                                   \
    } while (0)

constexpr auto ITERATIONS = 5;

// this class contains a random number generator
// it is used to generate random numbers on the host
class RandomNumberGenerator {
   public:
    RandomNumberGenerator(const unsigned long seed = std::random_device()())
        : m_generator(seed) {
        std::cout << "Seed: " << seed << std::endl;
    }

    // returns an array of integers with a given size, range is equal the size
    // that is shuffled using merseene twister
    std::vector<unsigned long> getRandomVector(const unsigned N) {
        std::vector<unsigned long> array(N);
// fill the array with numbers from 0 to N
#pragma omp simd
        for (unsigned long i = 0; i < N; ++i) { array[i] = i; }
        // shuffle the array
        std::shuffle(array.begin(), array.end(), m_generator);
        return array;
    }

   private:
    XoshiroCpp::Xoshiro256PlusPlus m_generator;
};

// This function generates an array of random numbers on the host
// copies it to the device and prints it
// and frees the meemory on the device
unsigned long* generateRandomNumbersOnDevice(RandomNumberGenerator& generator,
                                             const unsigned long    N) {
    // create a random number generator
    // generate an array of random numbers
    std::vector<unsigned long> vector = generator.getRandomVector(N);

    // copy the array to the device
    unsigned long* deviceArray;
    CUDA_CALL(cudaMalloc(&deviceArray, vector.size() * sizeof(unsigned long)));
    CUDA_CALL(cudaMemcpy(deviceArray, vector.data(),
                         vector.size() * sizeof(unsigned long),
                         cudaMemcpyHostToDevice));
    // print GPU memory utiuization
    size_t free, total;
    CUDA_CALL(cudaMemGetInfo(&free, &total));
    std::cout << "GPU memory usage: " << (total - free) / 1024 / 1024 << " MB"
              << std::endl;
    return deviceArray;
}

// this function frees the memory on the device
void freeMemoryOnDevice(unsigned long* deviceArray) {
    CUDA_CALL(cudaFree(deviceArray));
}

// this kernel reads an array of random numbers from the device
// it is designed to do random memory accesses
__global__ void latencyKernel(const unsigned long* array, const unsigned long N,
                              const unsigned long reads,
                              unsigned long*      result) {
    // get the thread id
    unsigned long index = (blockIdx.x * blockDim.x + threadIdx.x) % N;

    // iterate over the reads
    for (unsigned long i = 0; i < reads; ++i) { index = array[index]; }
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
double measureLatency(const unsigned long* deviceArray, const unsigned long N,
                      const unsigned long reads,
                      const unsigned long threadCount,
                      const unsigned long blockCount,
                      unsigned long*      resultArray) {
    invalidateCache();
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
double measureLatencyAndPrint(const unsigned long* deviceArray,
                              const unsigned long N, const unsigned long reads,
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
void appendToCSV(const unsigned long N, const unsigned long reads,
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
    // create a random number generator
    RandomNumberGenerator generator;
    // generate random numbers on the device
    std::cout
        << "Generating random numbers on the CPU and copying them to the GPU"
        << std::endl;
    unsigned long* deviceArray = generateRandomNumbersOnDevice(generator, N);

    // allocate memory for the result
    unsigned long* resultArray;
    CUDA_CALL(cudaMalloc(&resultArray, sizeof(unsigned long)));
    // measure the latency
    std::cout << "Measuring latency" << std::endl;
    auto time     = measureLatencyAndPrint(deviceArray, N, reads, threadCount,
                                           blockCount, resultArray);

    auto readTime = time / (reads * threadCount * blockCount);
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
    appendToCSV(N, reads, threadCount, blockCount, readTime, OUTPUT_CSV);
}

// // this function parses the command line arguments using getopt it accepts
// and
// // and returns size of the array, number of reads, number of threads and
// number
// // of blocks
// void parseCommandLineArguments(const int argc, const char* argv[],
//                                unsigned long& N, unsigned long& reads,
//                                unsigned long& threadCount,
//                                unsigned long& blockCount) {
//     // check if the number of arguments is correct
//     if (argc != 5) {
//         std::cout << "Usage: " << argv[0] << " N reads threadCount
//         blockCount"
//                   << std::endl;
//         exit(1);
//     }
//     // parse the arguments
//     N           = atoll(argv[1]);
//     reads       = atoll(argv[2]);
//     threadCount = atoll(argv[3]);
//     blockCount  = atoll(argv[4]);
// }

// int main(const int argc, const char* argv[]) {
//     // parse the command line arguments
//     unsigned long N, reads, threadCount, blockCount;
//     parseCommandLineArguments(argc, argv, N, reads, threadCount, blockCount);
//     // print the arguments
//     std::cout << "N: " << N << std::endl;
//     std::cout << "reads: " << reads << std::endl;
//     std::cout << "threadCount: " << threadCount << std::endl;
//     std::cout << "blockCount: " << blockCount << std::endl;
//     // run the test
//     runTest(N, reads, threadCount, blockCount);
//     return 0;
// }