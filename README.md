# gpu-performance-test
This repository contains the code to test GPU memory and branch penalty

### Branch penalty

The pseudocode for the benchmark is:
```cpp
// Generate a random array with a discrete distribution
bool random[32] = [1 with probability p else 0 ]
copy random to GPU
__global__ kernel (bool *random){
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) % 32;
    if (idx) {
        do something
    } else {
        do the same thing with different parameters
    }
}
```
Note: Waps are assumed of size = 32.  

The CPU code generates an array R with 32 elements. Each element has a discrete distribution: it is 1 with probability p, and 0 with probability 1-p. This array is then copied to the GPU.

In the GPU kernel, the variable idx is computed as THREAD_ID % 32. If R[idx] is 1, the kernel performs a certain action, while if it is 0, the kernel executes a function with the same number of PTX instructions. This results in branch divergence every time a 0 is encountered, which can be seen in the graph below.

![](images/branch_penalty.png)  

The results show that if even one warp diverges, the performance of the program is halved.

### Memory latency
The pseudocode for the benchmark is:
```cpp
auto vector = [i for i in N]
vector.shuffle();
copy the vector to the GPU
__global__ kernel(uint64_t* vector) {
    int tid = (threadIdx.x + blockIdx.x * blockDim.x) % N;
    auto index = tid;
    for (i=0 to reads) {
        index = vector[index];
    }
}
```
The code begins by creating an array where array[i] = i and then shuffling it using the xoroshiro-cpp random number generator. This method is used to avoid random number generation on the GPU, which can affect the accuracy of the results. The resulting array is then copied to the GPU.

The kernel uses the thread ID to address one element of the array, and then assigns index = vector[index] in a loop that is executed read times. Because of how the array is constructed, this results in uniform stochastic memory accesses. The kernel is executed read times to dilute the calling overhead, which can affect the accuracy of the results.  

![](images/read_latency.png)  

The results show that when the array fits in cache, the latency is around 30ns. However, when the array no longer fits in cache, the latency increases to ~170ns, and in some cases, up to ~300ns. If memory contention (parallel accesses from multiple threads) is introduced, the latency can increase to around ~360ns. This can be seen in the graph below.

### Note: tests are executed on a NVIDIA RTX 3090ti and CUDA V11.8.89

Github actions from: https://github.com/Ahdhn/CUDATemplate