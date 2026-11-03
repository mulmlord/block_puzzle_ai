// Random sources for the engine.
//
// PyRandom reproduces CPython's random.Random bit for bit (MT19937, int seeding via init_by_array,
// randrange by rejection on getrandbits), so C++ games can replay game.py games seed for seed.
// Xoshiro256 (xoshiro256**) is the fast source for training and benchmarks.
#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace bpa {

class PyRandom {
public:
    explicit PyRandom(uint64_t seed) {
        // random.seed(int): the absolute value as 32-bit words, least significant first; 0 -> [0]
        uint32_t key[2] = {static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32)};
        init_by_array(key, (seed >> 32) ? 2 : 1);
    }

    uint32_t genrand_uint32() {
        if (index_ >= N) twist();
        uint32_t y = mt_[index_++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= y >> 18;
        return y;
    }

    // 1 <= k <= 32
    uint32_t getrandbits(int k) { return genrand_uint32() >> (32 - k); }

    // n >= 1; Random._randbelow_with_getrandbits
    uint32_t randrange(uint32_t n) {
        const int k = std::bit_width(n);
        uint32_t r = getrandbits(k);
        while (r >= n) r = getrandbits(k);
        return r;
    }

private:
    static constexpr int N = 624, M = 397;
    std::array<uint32_t, N> mt_{};
    int index_ = N;

    void init_genrand(uint32_t s) {
        mt_[0] = s;
        for (int i = 1; i < N; i++) mt_[i] = 1812433253U * (mt_[i - 1] ^ (mt_[i - 1] >> 30)) + static_cast<uint32_t>(i);
        index_ = N;
    }

    void init_by_array(const uint32_t* key, int key_length) {
        init_genrand(19650218U);
        int i = 1, j = 0;
        for (int k = N > key_length ? N : key_length; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1664525U)) + key[j] + static_cast<uint32_t>(j);
            i++, j++;
            if (i >= N) mt_[0] = mt_[N - 1], i = 1;
            if (j >= key_length) j = 0;
        }
        for (int k = N - 1; k; k--) {
            mt_[i] = (mt_[i] ^ ((mt_[i - 1] ^ (mt_[i - 1] >> 30)) * 1566083941U)) - static_cast<uint32_t>(i);
            i++;
            if (i >= N) mt_[0] = mt_[N - 1], i = 1;
        }
        mt_[0] = 0x80000000U;
    }

    void twist() {
        constexpr uint32_t upper = 0x80000000U, lower = 0x7fffffffU, matrix_a = 0x9908b0dfU;
        auto mix = [&](int a, int b, int c) {
            const uint32_t y = (mt_[a] & upper) | (mt_[b] & lower);
            mt_[a] = mt_[c] ^ (y >> 1) ^ ((y & 1U) ? matrix_a : 0U);
        };
        int kk = 0;
        for (; kk < N - M; kk++) mix(kk, kk + 1, kk + M);
        for (; kk < N - 1; kk++) mix(kk, kk + 1, kk + M - N);
        mix(N - 1, 0, M - 1);
        index_ = 0;
    }
};

inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

class Xoshiro256 {
public:
    explicit Xoshiro256(uint64_t seed) {
        for (auto& word : s_) word = splitmix64(seed);
    }

    uint64_t next() {
        const uint64_t result = std::rotl(s_[1] * 5, 7) * 9;
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

    // n >= 1; Lemire's multiply-shift with rejection (unbiased)
    uint32_t randrange(uint32_t n) {
        uint64_t m = (next() >> 32) * n;
        auto low = static_cast<uint32_t>(m);
        if (low < n) {
            const uint32_t threshold = -n % n;
            while (low < threshold) {
                m = (next() >> 32) * n;
                low = static_cast<uint32_t>(m);
            }
        }
        return static_cast<uint32_t>(m >> 32);
    }

private:
    std::array<uint64_t, 4> s_{};
};

}  // namespace bpa
