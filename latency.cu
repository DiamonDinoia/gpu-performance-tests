#include "latency.h"

constexpr auto OUTPUT_CSV = "latency.csv";

// this function calls runTest() with N from 1024 up to 2^32 in powers of 2
// reads = 2^32 , threadsCount = 1 , blockSize = 1
void runTestPowersOf2() {
    for (unsigned long i = 1ULL << 13ULL; i <= 1ULL << 31ULL; i <<= 1) {
        const auto size = (i * 8.) / (1ULL << 20ULL);
        std::cout << "N = " << size << " MB" << std::endl;
        runTest(i, 1ULL << 15, 1, 1, OUTPUT_CSV);
        runTest(i, 1ULL << 15, 128, 128, OUTPUT_CSV);
    }
}

// the main call runtestPowersOf2()
int main() {
    runTestPowersOf2();
    return 0;
}