#include <XoshiroCpp.hpp>
#include <algorithm>
#include <array>
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
        for (unsigned long i = 0; i < N; ++i) { array[i] = i; }
        // shuffle the array
        std::shuffle(array.begin(), array.end(), m_generator);
        return array;
    }

    // this function generates a vector of 32 booleans using a discrete
    // distribution with proability p and 1-p for 0 and 1 respectively
    std::array<bool, 32> getRandomBoolVector(const double p) {
        std::discrete_distribution<bool> distribution({1 - p, p});
        //
        std::array<bool, 32> array{};
        for (auto& elem : array) { elem = distribution(m_generator); }
        return array;
    }

   private:
    XoshiroCpp::Xoshiro256PlusPlus m_generator;
};
