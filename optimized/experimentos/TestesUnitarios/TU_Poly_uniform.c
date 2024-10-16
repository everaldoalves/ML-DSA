//************************************************************************************************
// Autor: Everaldo Alves
// Data: 14 de Setembro/2024
// Função: poly_uniform
// Descrição: Esta função amostra um polinômio com coeficientes uniformemente aleatórios
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Speed-up em relação à função de referência: 1.70x (39/70 ciclos)
//************************************************************************************************


#include <arm_neon.h>
#include "cpucycles.h"
#include "symmetric.h"
#include "fips202.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>

#define SEEDBYTES 32

#define N 256
#define Q 8380417
#define NTESTS 10000
#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)

// Estrutura de um polinômio
typedef struct {
    int32_t coeffs[N];
} poly;


// Função de benchmark para ciclos de CPU
uint64_t cpucycles_overhead(void) {
    uint64_t t0, t1, overhead = -1LL;
    for (unsigned int i = 0; i < 100000; i++) {
        t0 = cpucycles();
        __asm__ volatile(""); // Para evitar otimização
        t1 = cpucycles();
        if (t1 - t0 < overhead)
            overhead = t1 - t0;
    }
    return overhead;
}

// Função para calcular o tempo de execução de uma operação
static int cmp_uint64(const void *a, const void *b) {
    return (*(uint64_t *)a < *(uint64_t *)b) ? -1 : (*(uint64_t *)a > *(uint64_t *)b) ? 1 : 0;
}

static uint64_t median(uint64_t *l, size_t llen) {
    qsort(l, llen, sizeof(uint64_t), cmp_uint64);
    return (llen % 2) ? l[llen / 2] : (l[llen / 2 - 1] + l[llen / 2]) / 2;
}

static uint64_t average(uint64_t *t, size_t tlen) {
    uint64_t acc = 0;
    for (size_t i = 0; i < tlen; i++)
        acc += t[i];
    return acc / tlen;
}

void print_results(const char *s, uint64_t *t, size_t tlen) {
    static uint64_t overhead = -1;
    if (overhead == (uint64_t)-1)
        overhead = cpucycles_overhead();

    tlen--;
    for (size_t i = 0; i < tlen; ++i)
        t[i] = t[i + 1] - t[i] - overhead;

    printf("%s\n", s);
    printf("median: %llu cycles/ticks\n", (unsigned long long)median(t, tlen));
    printf("average: %llu cycles/ticks\n", (unsigned long long)average(t, tlen));
    printf("\n");
}

static unsigned int rej_uniform(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr, pos;
    uint32_t t;

    ctr = pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        t  = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= 0x7FFFFF;
     
        if (t < Q)
            a[ctr++] = t;
    }

    return ctr;
}


/*************************************************
* Name:        poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE128(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
void poly_uniform(poly *a,
                  const uint8_t seed[SEEDBYTES],
                  uint16_t nonce)
{
  unsigned int i, ctr, off;
  unsigned int buflen = POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES + 2];
  stream128_state state;

  stream128_init(&state, seed, nonce);
  stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

  ctr = rej_uniform(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    off = buflen % 3;
    for(i = 0; i < off; ++i)
      buf[i] = buf[buflen - off + i];

    stream128_squeezeblocks(buf + off, 1, &state);
    buflen = STREAM128_BLOCKBYTES + off;
    ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf, buflen);
  }
}

static unsigned int rej_uniform_neon2(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0;
    unsigned int pos = 0;

    uint32x4_t mask = vdupq_n_u32(0x7FFFFF);
    uint32x4_t Qvec = vdupq_n_u32(Q);

    while (ctr < len && pos + 12 <= buflen) {
        uint32x4_t t = {
            buf[pos] | ((uint32_t)buf[pos+1] << 8) | ((uint32_t)buf[pos+2] << 16),
            buf[pos+3] | ((uint32_t)buf[pos+4] << 8) | ((uint32_t)buf[pos+5] << 16),
            buf[pos+6] | ((uint32_t)buf[pos+7] << 8) | ((uint32_t)buf[pos+8] << 16),
            buf[pos+9] | ((uint32_t)buf[pos+10] << 8) | ((uint32_t)buf[pos+11] << 16)
        };
        t = vandq_u32(t, mask);

        uint32x4_t lt_Q = vcltq_u32(t, Qvec);

        for (int i = 0; i < 4 && ctr < len; i++) {
            if (lt_Q[i]) {
                a[ctr++] = t[i];
            }
        }

        pos += 12;
    }

    return ctr;
}

void poly_uniform_unrolling(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    unsigned int i, ctr, off;
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
    uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
    stream128_state state;

    // Inicializa o stream de acordo com o seed e nonce
    stream128_init(&state, seed, nonce);
    stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

    // Usa a versão otimizada da rejeição
    ctr = rej_uniform_neon2(a->coeffs, N, buf, buflen);

    // Loop de amostragem adicional até completar todos os coeficientes
    while (ctr < N) {
        // Calcula offset para a próxima iteração
        off = buflen % 3;
        for (i = 0; i < off; ++i) {
            buf[i] = buf[buflen - off + i];  // Reajusta o buffer
        }

        // Pré-carrega blocos adicionais do stream
        __builtin_prefetch(&buf, 0, 3);  // Prefetch para melhorar cache hit
        stream128_squeezeblocks(buf + off, 1, &state);
        buflen = STREAM128_BLOCKBYTES + off;

        // Rejeição usando a função otimizada
        ctr += rej_uniform_neon2(a->coeffs + ctr, N - ctr, buf, buflen);
    }
}

void poly_uniform_neon(poly *a, const uint8_t seed[SEEDBYTES], uint16_t nonce) {
    unsigned int i, ctr, off;
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
    uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
    stream128_state state;

    // Inicializa o stream com a semente e nonce
    stream128_init(&state, seed, nonce);
    stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

    // Usa a função otimizada de rejeição NEON
    ctr = rej_uniform_neon2(a->coeffs, N, buf, buflen);

    // Continua amostrando os coeficientes do polinômio até preencher
    while (ctr < N) {
        // Calcula o offset para blocos não preenchidos
        off = buflen % 3;
        for (i = 0; i < off; ++i) {
            buf[i] = buf[buflen - off + i];  // Ajusta o buffer
        }

        // Pré-carregamento do buffer para melhorar o uso de cache
        __builtin_prefetch(&buf, 0, 3);

        // Squeeze mais blocos do stream para preencher o buffer
        stream128_squeezeblocks(buf + off, 1, &state);
        buflen = STREAM128_BLOCKBYTES + off;

        // Executa rejeição utilizando a versão NEON otimizada
        ctr += rej_uniform_neon2(a->coeffs + ctr, N - ctr, buf, buflen);
    }
}


// Função para garantir que o compilador não elimine o resultado
void force_use(int32_t *p, unsigned int len) {
    volatile int32_t sum = 0;
    for (unsigned int i = 0; i < len; ++i) {
        sum += p[i];  // Soma os valores para evitar otimização
    }
}

// Função para comparação de dois polinômios
int compare_polys(const poly *a, const poly *b) {
    for (int i = 0; i < N; ++i) {
        if (a->coeffs[i] != b->coeffs[i])
            return 0;
    }
    return 1;
}

// Inicializa o polinômio com valores aleatórios
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % Q;  // Coeficientes aleatórios
    }
}

// Exibe o polinômio
void exibe_polinomio(const int32_t *p ) {
    for (unsigned int i = 0; i < N; i++) {
        printf("%d ", p[i]);
    }
    printf("\n");
}

int main() {
    poly a_ref, a_neon, a_unroll;
    uint8_t buf[1024];
    uint64_t t[NTESTS];
    uint8_t seed[SEEDBYTES];
    uint16_t nonce = 1234;
    unsigned int len = N;
    unsigned int sampled_ref = 0, sampled_neon = 0, sampled_unroll = 0;

    // Gera um seed aleatório
    for (int i = 0; i < SEEDBYTES; i++) {
        seed[i] = rand() % 256;
    }

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        poly_uniform(&a_ref, seed, nonce);
        force_use(a_ref.coeffs, sampled_ref);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_uniform (REF):", t, NTESTS);
    uint64_t media_ref = median(t, NTESTS);

    // Teste para a função que usa loop unrolling e prefetch manual
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        poly_uniform_unrolling(&a_unroll, seed, nonce);
        force_use(a_unroll.coeffs, sampled_unroll);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_uniform (UNROLLING):", t, NTESTS);
    uint64_t media_unroll = median(t, NTESTS);

    // Teste para a função otimizada com NEON
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        poly_uniform_neon(&a_neon, seed, nonce);
        force_use(a_neon.coeffs, sampled_neon);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_uniform (NEON):", t, NTESTS);
    uint64_t media_neon = median(t, NTESTS);

    printf("Antes do IF\n");

    // Verificar se os resultados são iguais
    if (sampled_ref != sampled_neon || !compare_polys(&a_ref, &a_neon) ||
        sampled_ref != sampled_unroll || !compare_polys(&a_ref, &a_unroll)) {
        printf("\nOs resultados dos polinômios são diferentes\n");
        printf("\n Polinômio A_REF:\n");
        exibe_polinomio(a_ref.coeffs);
        printf("\n Polinômio A_UNROLL:\n");
        exibe_polinomio(a_unroll.coeffs);
        printf("Polinômio A_NEON:\n");
        exibe_polinomio(a_neon.coeffs);
        return 1;
    }

    printf("\nOs resultados dos polinômios são iguais\n");

    uint64_t fastest_time = media_ref;
    char *fastest_version = "Referência";

    if (media_unroll < fastest_time) {
        fastest_time = media_unroll;
        fastest_version = "Unrolling";
    }
    if (media_neon < fastest_time) {
        fastest_time = media_neon;
        fastest_version = "NEON";
    }

    double speedup = (double)media_ref / fastest_time;

    printf("\nA versão mais rápida foi: %s\n", fastest_version);
    printf("Speed-up em relação à função de referência: %.2fx\n", speedup);

    exibe_polinomio(a_ref.coeffs);
    printf("\n");
    exibe_polinomio(a_unroll.coeffs);
    printf("\n");
    exibe_polinomio(a_neon.coeffs);

    return 0;
}
