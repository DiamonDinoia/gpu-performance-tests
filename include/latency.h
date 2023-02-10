#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <random>

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
    RandomNumberGenerator(const unsigned long seed = 42) : m_generator(seed) {}

    // returns an array of integers with a given size, range is equal the size
    // that is shuffled using merseene twister
    std::vector<unsigned long> getRandomVector(const unsigned N) {
        std::vector<unsigned long> array(N);
        // fill the array with numbers from 0 to N
        for (unsigned long i = 0; i < N; ++i) { array[i] = i; }
        // shuffle the array
        std::shuffle(array.begin(), array.end(), m_generator);
        return array;
    }

   private:
    std::mt19937_64 m_generator;
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
    for (unsigned long i = 0; i < reads; ++i) {
        index = array[index];
        result[0] += index;
    }
}

// this function call latencyKernel kernel
// it also measures the time it takes to execute the kernel
// and returns the time in milliseconds
double measureLatency(const unsigned long* deviceArray, const unsigned long N,
                      const unsigned long reads,
                      const unsigned long threadCount,
                      const unsigned long blockCount,
                      unsigned long*      resultArray) {
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

// this function calls generateRandomNumbersOnDevice, measureLatencyAndPrint and
// freeMemoryOnDevice
void runTest(const unsigned long N, const unsigned long reads,
             const unsigned long threadCount, const unsigned long blockCount) {
    // create a random number generator
    RandomNumberGenerator generator;
    // generate random numbers on the device
    std::cout << "Generating random numbers on the device" << std::endl;
    unsigned long* deviceArray = generateRandomNumbersOnDevice(generator, N);

    // allocate memory for the result
    unsigned long* resultArray;
    CUDA_CALL(cudaMalloc(&resultArray, sizeof(unsigned long)));
    // measure the latency
    std::cout << "Measuring latency" << std::endl;
    auto time     = measureLatencyAndPrint(deviceArray, N, reads, threadCount,
                                           blockCount, resultArray);

    auto readTime = time / reads;
    std::cout << "Avewrage memory read time: " << readTime << " ns"
              << std::endl;

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