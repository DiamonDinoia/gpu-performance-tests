#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <vector>

// this macro checks if a cuda function call was successful
#define CUDA_CALL(x)                                        \
    do {                                                    \
        if ((x) != cudaSuccess) {                           \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(x);                                        \
        }                                                   \
    } while (0)

// User-defined literals
auto constexpr operator"" _B(unsigned long long int n) { return n; }
auto constexpr operator"" _KB(unsigned long long int n) { return n * 1024; }
auto constexpr operator"" _M(unsigned long long int n) {
    return n * 1000 * 1000;
}

// Cache line size: 64 bytes for x86-64, 128 bytes for A64 ARMs
const auto kCachelineSize = 64_B;
// Memory page size. Default page size is 4 KB
const auto kPageSize = 4_KB;

// Singly linked list node with padding
struct Element {
    unsigned long index;
    std::byte     padding[kPageSize];
};

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
    std::vector<Element> getShuffledVector(const unsigned N) {
        std::vector<Element> list(N);
        // initialise the array with sequential indices
        for (auto i = 0; i < N; ++i) { list[i].index = i; }
        // shuffle the array
        std::shuffle(list.begin(), list.end(), m_generator);
        return list;
    }

    // this function generates a vector of 32 booleans using a discrete
    // distribution with proability p and 1-p for 0 and 1 respectively
    std::array<bool, 32> getRandomBoolVector(const double p) {
        std::discrete_distribution<unsigned char> distribution({1 - p, p});
        //
        std::array<bool, 32> array{};
        for (auto& elem : array) {
            elem = static_cast<bool>(distribution(m_generator));
        }
        return array;
    }

   private:
    std::mt19937_64 m_generator;
};
