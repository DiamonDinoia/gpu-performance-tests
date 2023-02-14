#include "branch.h"

// this function calls runBranchKernel with probability 1 the first time and
// decreases it by 0.1 each time by .10
void runTest() {
    RandomNumberGenerator generator{};
    // for loop to call runBranchKernel with probability 1, 0.9, 0.8, 0.7, 0.6,
    // 0.5, 0.4, 0.3, 0.2, 0.1
    for (int i = 10; i > 0; i--) { runBranchKernel(i * 0.1, generator); }
}

// the main call runtestPowersOf2()
int main() {
    runTest();
    return 0;
}