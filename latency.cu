#include "latency.h"

// this function calls runTest() with N from 1024 up to 2^32 in powers of 2
// reads = 2^32 , threadsCount = 1 , blockSize = 1
void runTestPowersOf2() {
    for (unsigned long i = 10ULL; i <= 38ULL; i++) {
        unsigned long size = (1ULL << i) / (1ULL << 17ULL);
        std::cout << "N = " << size << " MB" << std::endl;
        runTest(1ULL << i, 1ULL << 32ULL, 1ULL, 1ULL);
    }
}

// the main call runtestPowersOf2()
int main() {
    runTestPowersOf2();
    return 0;
}