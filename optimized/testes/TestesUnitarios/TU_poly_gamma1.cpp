//************************************************************************************************
// Autor: Everaldo Alves
// Data: 11 de Setembro/2024
// Função: poly_polygamma1
// Descrição: Esta função implementa um benchmark para medir o desempenho de funções de geração de polinômios
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜50% nos ciclos de CPU
//************************************************************************************************

#include <benchmark/benchmark.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include "symmetric.h"
#include "fips202.h"
#include "fips202x2.h"
#include <arm_neon.h>

// Definição de constantes e estruturas (reutilizar as definidas anteriormente)
#define K 4
#define L 4
#define ETA 2
#define TAU 39
#define BETA 78
#define GAMMA1 (1 << 17)
#define GAMMA2 ((Q-1)/88)
#define OMEGA 80
#define CTILDEBYTES 32
#define POLY_UNIFORM_GAMMA1_NBLOCKS ((POLYZ_PACKEDBYTES + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)
#define POLYZ_PACKEDBYTES   576

typedef struct {
  int32_t coeffs[N];
} poly;

// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}

// Implementações das funções polyz_unpack, poly_uniform_gamma1_2x, poly_uniform_gamma1_4x
void polyz_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  
#if GAMMA1 == (1 << 17) // Se gamma1 = 2^17
  for(i = 0; i < N/4; ++i) {
    r->coeffs[4*i+0]  = a[9*i+0];
    r->coeffs[4*i+0] |= (uint32_t)a[9*i+1] << 8;
    r->coeffs[4*i+0] |= (uint32_t)a[9*i+2] << 16;
    r->coeffs[4*i+0] &= 0x3FFFF;

    r->coeffs[4*i+1]  = a[9*i+2] >> 2;
    r->coeffs[4*i+1] |= (uint32_t)a[9*i+3] << 6;
    r->coeffs[4*i+1] |= (uint32_t)a[9*i+4] << 14;
    r->coeffs[4*i+1] &= 0x3FFFF;

    r->coeffs[4*i+2]  = a[9*i+4] >> 4;
    r->coeffs[4*i+2] |= (uint32_t)a[9*i+5] << 4;
    r->coeffs[4*i+2] |= (uint32_t)a[9*i+6] << 12;
    r->coeffs[4*i+2] &= 0x3FFFF;

    r->coeffs[4*i+3]  = a[9*i+6] >> 6;
    r->coeffs[4*i+3] |= (uint32_t)a[9*i+7] << 2;
    r->coeffs[4*i+3] |= (uint32_t)a[9*i+8] << 10;
    r->coeffs[4*i+3] &= 0x3FFFF;

    r->coeffs[4*i+0] = GAMMA1 - r->coeffs[4*i+0];
    r->coeffs[4*i+1] = GAMMA1 - r->coeffs[4*i+1];
    r->coeffs[4*i+2] = GAMMA1 - r->coeffs[4*i+2];
    r->coeffs[4*i+3] = GAMMA1 - r->coeffs[4*i+3];
  }
#elif GAMMA1 == (1 << 19)
  for(i = 0; i < N/2; ++i) {
    r->coeffs[2*i+0]  = a[5*i+0];
    r->coeffs[2*i+0] |= (uint32_t)a[5*i+1] << 8;
    r->coeffs[2*i+0] |= (uint32_t)a[5*i+2] << 16;
    r->coeffs[2*i+0] &= 0xFFFFF;

    r->coeffs[2*i+1]  = a[5*i+2] >> 4;
    r->coeffs[2*i+1] |= (uint32_t)a[5*i+3] << 4;
    r->coeffs[2*i+1] |= (uint32_t)a[5*i+4] << 12;
    /* r->coeffs[2*i+1] &= 0xFFFFF; */ /* No effect, since we're anyway at 20 bits */

    r->coeffs[2*i+0] = GAMMA1 - r->coeffs[2*i+0];
    r->coeffs[2*i+1] = GAMMA1 - r->coeffs[2*i+1];
  }
#endif

  
}


void poly_uniform_gamma1(poly *a,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce)
{
  uint8_t buf[POLY_UNIFORM_GAMMA1_NBLOCKS*STREAM256_BLOCKBYTES];
  stream256_state state;

  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);
  polyz_unpack(a, buf);
}

void poly_uniform_gamma1_2x(poly *a0,
                                   poly *a1,
                                   const uint8_t seed[CRHBYTES],
                                   uint16_t nonce0,
                                   uint16_t nonce1)
{
    // Calcula o tamanho do buffer necessário
    size_t buf_size = POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14;
    
    // Aloca buffers alinhados para as duas instâncias
    uint8_t buf0[POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14] __attribute__((aligned(16)));
    uint8_t buf1[POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14] __attribute__((aligned(16)));
    
    // Inicializa os estados do stream
    stream256_state state0, state1;
    stream256_init(&state0, seed, nonce0);
    stream256_init(&state1, seed, nonce1);
    
    // Gera os blocos de bytes para cada polinômio
    stream256_squeezeblocks(buf0, POLY_UNIFORM_GAMMA1_NBLOCKS, &state0);
    stream256_squeezeblocks(buf1, POLY_UNIFORM_GAMMA1_NBLOCKS, &state1);
    
    // Desempacota os bytes gerados para os polinômios
    polyz_unpack(a0, buf0);
    polyz_unpack(a1, buf1);
}




void poly_uniform_gamma1_2x_neon(poly *a0, poly *a1, const uint8_t seed[64], 
                            uint16_t nonce0, uint16_t nonce1) {
  uint8_t buf[2][POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14];
  uint64x2_t f0, f1;
  keccakx2_state state;

  // Carregar os primeiros 32 bytes do seed em registradores NEON
  f0 = vld1q_u64((const uint64_t *)&seed[0]);
  vst1q_u64((uint64_t *)&buf[0][0], f0);
  vst1q_u64((uint64_t *)&buf[1][0], f0);

  // Carregar os segundos 32 bytes do seed em registradores NEON
  f1 = vld1q_u64((const uint64_t *)&seed[32]);
  vst1q_u64((uint64_t *)&buf[0][16], f1);
  vst1q_u64((uint64_t *)&buf[1][16], f1);

  // Definir os nonces nos buffers
  buf[0][64] = nonce0 & 0xFF;
  buf[0][65] = (nonce0 >> 8) & 0xFF;
  buf[1][64] = nonce1 & 0xFF;
  buf[1][65] = (nonce1 >> 8) & 0xFF;

  // Absorver os dados para os 2 polinômios simultaneamente
  shake256x2_absorb(&state, buf[0], buf[1], 66);

  // Realizar squeezeblocks para obter os coeficientes
  shake256x2_squeezeblocks(buf[0], buf[1], POLY_UNIFORM_GAMMA1_NBLOCKS, &state);

  // Descompactar os coeficientes em polinômios
  polyz_unpack(a0, buf[0]);
  polyz_unpack(a1, buf[1]);
}


void poly_uniform_gamma1_4x_neon(poly *a0,
                                   poly *a1,
                                   poly *a2,
                                   poly *a3,
                                   const uint8_t seed[CRHBYTES],
                                   uint16_t nonce0,
                                   uint16_t nonce1,
                                   uint16_t nonce2,
                                   uint16_t nonce3)
{
    // Processa os dois primeiros polinômios em paralelo
    poly_uniform_gamma1_2x_neon(a0, a1, seed, nonce0, nonce1);
    
    // Processa os dois polinômios seguintes em paralelo
    poly_uniform_gamma1_2x_neon(a2, a3, seed, nonce2, nonce3);
}


// Benchmark para poly_uniform_gamma1_2x
static void BM_poly_uniform_gamma1(benchmark::State& state) {
    poly a0;
    uint8_t seed[CRHBYTES] = {0}; // Inicialize com dados de exemplo
    uint16_t nonce0 = 1;    
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_gamma1(&a0, seed, nonce0);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_gamma1);

// Benchmark para poly_uniform_gamma1_2x
static void BM_poly_uniform_gamma1_2x(benchmark::State& state) {
    poly a0, a1;
    uint8_t seed[CRHBYTES] = {0}; // Inicialize com dados de exemplo
    uint16_t nonce0 = 1;
    uint16_t nonce1 = 2;
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_gamma1_2x(&a0, &a1, seed, nonce0, nonce1);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_gamma1_2x);

// Benchmark para poly_uniform_gamma1_2x
static void BM_poly_uniform_gamma1_2x_neon(benchmark::State& state) {
    poly a0, a1;
    uint8_t seed[CRHBYTES] = {0}; // Inicialize com dados de exemplo
    uint16_t nonce0 = 1;
    uint16_t nonce1 = 2;
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_gamma1_2x_neon(&a0, &a1, seed, nonce0, nonce1);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_gamma1_2x_neon);

// Benchmark para poly_uniform_gamma1_4x
static void BM_poly_uniform_gamma1_4x_neon(benchmark::State& state) {
    poly a0, a1,a2,a3;
    uint8_t seed[CRHBYTES] = {0}; // Inicialize com dados de exemplo
    uint16_t nonce0 = 1;
    uint16_t nonce1 = 2;
    uint16_t nonce2 = 3;
    uint16_t nonce3 = 4;
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        poly_uniform_gamma1_4x_neon(&a0, &a1, &a2, &a3, seed, nonce0, nonce1, nonce2, nonce3);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}
BENCHMARK(BM_poly_uniform_gamma1_4x_neon);


// Função main para benchmarks
BENCHMARK_MAIN();
