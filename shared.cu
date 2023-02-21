#include "latency.h"

constexpr auto OUTPUT_CSV = "shared-latency.csv";

// this function calls runTest() with N from 1024 up to 2^32 in powers of 2
// reads = 2^32 , threadsCount = 1 , blockSize = 1
void runTestPowersOf2() {
    auto iteration = 0;
    for (unsigned long i = 1ULL << 3ULL; i < 1ULL << 6ULL; i <<= 1) {
        const auto size = operator""_KB(i);
        std::cout << "N = " << size * 64. / (1024 * 1024) << " MB" << std::endl;
        runTest<true>(i, 1ULL << 18, 128, 1, OUTPUT_CSV);
    }
}

int main() {
    runTestPowersOf2();
    return 0;
}