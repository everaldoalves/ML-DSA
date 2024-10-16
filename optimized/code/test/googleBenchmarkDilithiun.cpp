// Comando para compilar a partir da pasta ref:
//g++ -O2 -std=c++11 -I /opt/homebrew/include test/googleBenchmarkDilithiun.cpp sign.c poly.c polyvec.c randombytes.c ntt.c reduce.c fips202.c fips202x2.c packing.c rounding.c symmetric-shake.c feat.S -L /opt/homebrew/lib -lbenchmark -lpthread -o test/googleBenchmarkDilithiun

#include <benchmark/benchmark.h>
#include <stdint.h>
#include "../sign.h"
#include "../poly.h"
#include "../polyvec.h"
#include "../params.h"

// Constantes
uint8_t pk[CRYPTO_PUBLICKEYBYTES];
uint8_t sk[CRYPTO_SECRETKEYBYTES];
uint8_t sig[CRYPTO_BYTES];
uint8_t seed[CRHBYTES];
polyvecl mat[K];
poly *a = &mat[0].vec[0];
poly *b = &mat[0].vec[1];
poly *c = &mat[0].vec[2];

// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}



// Função para medir o tempo de "polyvec_matrix_expand"
static void BM_polyvec_matrix_expand(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        polyvec_matrix_expand(mat, seed);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_polyvec_matrix_expand);

// Função para medir o tempo de "poly_uniform_eta"
static void BM_poly_uniform_eta(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_eta(a, seed, 0);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_eta);

// Função para medir o tempo de "poly_uniform_gamma1"
static void BM_poly_uniform_gamma1(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_gamma1(a, seed, 0);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_gamma1);

// Função para medir o tempo de "poly_ntt"
static void BM_poly_ntt(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_ntt(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_ntt);

// Função para medir o tempo de "poly_invntt_tomont"
static void BM_poly_invntt_tomont(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_invntt_tomont(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_invntt_tomont);

// Função para medir o tempo de "poly_pointwise_montgomery"
static void BM_poly_pointwise_montgomery(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_pointwise_montgomery(c, a, b);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();            
}
BENCHMARK(BM_poly_pointwise_montgomery);

// Função para medir o tempo de "poly_challenge"
static void BM_poly_challenge(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_challenge(c, seed);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_challenge);

// Função para medir o tempo de "crypto_sign_keypair"
static void BM_crypto_sign_keypair(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        crypto_sign_keypair(pk, sk);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_crypto_sign_keypair);

// Função para medir o tempo de "crypto_sign_signature"
static void BM_crypto_sign_signature(benchmark::State &state) {
    size_t siglen;
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        crypto_sign_signature(sig, &siglen, sig, CRHBYTES, NULL, 0, sk);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_crypto_sign_signature);

// Função para medir o tempo de "crypto_sign_verify"
static void BM_crypto_sign_verify(benchmark::State &state) {
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {        
        start_cycles = cpucycles();
        crypto_sign_verify(sig, CRYPTO_BYTES, sig, CRHBYTES, NULL, 0, pk);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_crypto_sign_verify);

// Função principal para executar os benchmarks
BENCHMARK_MAIN();
