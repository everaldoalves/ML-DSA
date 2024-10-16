//************************************************************************************************
// Autor: Everaldo Alves
// Data: 16 de Setembro/2024
// Função: poly_uniform_eta
// Descrição: Esta função realiza a amostragem de polinômios com coeficientes uniformemente distribuídos
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
#include <string.h>

#define SEEDBYTES 32

#define N 256
#define Q 8380417
#define ETA 2
#define NTESTS 1



#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)
#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#endif


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

// Função para garantir que o compilador não elimine o resultado
void force_use(int32_t *p, unsigned int len) {
    volatile int32_t sum = 0;
    for (unsigned int i = 0; i < len; ++i) {
        sum += p[i];  // Soma os valores para evitar otimização
    }
}

// Estrutura de estado do Stream256
void print_stream256_state(stream256_state *state) {
    printf("Estado do Stream256:\n");
    for (int i = 0; i < 25; i++) {
        printf("s[%d] = %llu\n", i, (unsigned long long)state->s[i]);
    }
    printf("pos = %u\n", state->pos);
}

/*************************************************
* Name:        rej_eta
*
* Description: Sample uniformly random coefficients in [-ETA, ETA] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
static unsigned int rej_eta(int32_t *a,
                            unsigned int len,
                            const uint8_t *buf,
                            unsigned int buflen)
{
  unsigned int ctr, pos;
  uint32_t t0, t1;
  //DBENCH_START();

  ctr = pos = 0;
  while(ctr < len && pos < buflen) {
    t0 = buf[pos] & 0x0F;
    t1 = buf[pos++] >> 4;

#if ETA == 2
    if(t0 < 15) {
      t0 = t0 - (205*t0 >> 10)*5;
      a[ctr++] = 2 - t0;
    }
    if(t1 < 15 && ctr < len) {
      t1 = t1 - (205*t1 >> 10)*5;
      a[ctr++] = 2 - t1;
    }
#elif ETA == 4
    if(t0 < 9)
      a[ctr++] = 4 - t0;
    if(t1 < 9 && ctr < len)
      a[ctr++] = 4 - t1;
#endif
  }

  //DBENCH_STOP(*tsample);
  return ctr;
}

/*************************************************
* Name:        poly_uniform_eta
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-ETA,ETA] by performing rejection sampling on the
*              output stream from SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/

void poly_uniform_eta(poly *a,
                      const uint8_t seed[CRHBYTES],
                      uint16_t nonce)
{
  unsigned int ctr;
  unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;

  printf("Nonce (REF): %u\n", nonce); // Log do nonce

  stream256_init(&state, seed, nonce);
  printf("\n Estado do Stream256 após stream256_init:\n");
  print_stream256_state(&state);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);
  printf("\n Estado do Stream256 após stream256_squeezeblocks:\n");
  print_stream256_state(&state);

  // Log do buffer gerado
  printf("Buffer após stream256_squeezeblocks (REF):\n");
  for (int i = 0; i < buflen; ++i) {
    printf("%02x ", buf[i]);
  }
  printf("\n");
    printf("\nTamanho de stream256_state: %zu bytes\n", sizeof(stream256_state));

  ctr = rej_eta(a->coeffs, N, buf, buflen);

  while (ctr < N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM256_BLOCKBYTES);
  }
}


static unsigned int rej_eta_unroll(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0, pos = 0;
    uint32_t t0, t1;
    
    // Ajuste o fator de desenrolamento (unroll_factor). Teste com diferentes valores como 8 ou 16.
    const int unroll_factor = 8;

    // Vamos pré-carregar dados em blocos para evitar cache misses
    while (ctr + unroll_factor <= len && pos + unroll_factor <= buflen) {
        // Prefetch manual: Carrega os bytes seguintes para o cache.
        __builtin_prefetch(&buf[pos + unroll_factor], 0, 1);

        for (int i = 0; i < unroll_factor; i++) {
            // Extrair t0 e t1 dos 4 bits menos e mais significativos de buf[pos]
            t0 = buf[pos] & 0x0F;
            t1 = buf[pos++] >> 4;

            // Para ETA = 2
            #if ETA == 2
                        if (t0 < 15) {
                            t0 = t0 - (205 * t0 >> 10) * 5;
                            a[ctr++] = 2 - t0;
                        }
                        if (t1 < 15 && ctr < len) {
                            t1 = t1 - (205 * t1 >> 10) * 5;
                            a[ctr++] = 2 - t1;
                        }
            #elif ETA == 4
                        // Para ETA = 4
                        if (t0 < 9)
                            a[ctr++] = 4 - t0;
                        if (t1 < 9 && ctr < len)
                            a[ctr++] = 4 - t1;
            #endif
        }
    }

    // Processar os coeficientes restantes
    while (ctr < len && pos < buflen) {
        t0 = buf[pos] & 0x0F;
        t1 = buf[pos++] >> 4;

#if ETA == 2
        if (t0 < 15) {
            t0 = t0 - (205 * t0 >> 10) * 5;
            a[ctr++] = 2 - t0;
        }
        if (t1 < 15 && ctr < len) {
            t1 = t1 - (205 * t1 >> 10) * 5;
            a[ctr++] = 2 - t1;
        }
#elif ETA == 4
        if (t0 < 9)
            a[ctr++] = 4 - t0;
        if (t1 < 9 && ctr < len)
            a[ctr++] = 4 - t1;
#endif
    }

    return ctr;
}

// Função otimizada para gerar polinômios com coeficientes uniformemente distribuídos
void poly_uniform_eta_optimized(poly *a,
                      const uint8_t seed[CRHBYTES],
                      uint16_t nonce)
{
  unsigned int ctr;
  unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS * STREAM256_BLOCKBYTES];
  stream256_state state;

  printf("Nonce (OPTIMIZED): %u\n", nonce); // Log do nonce

  stream256_init(&state, seed, nonce);
  printf("\n Estado do Stream256 após stream256_init (OPTIMIZED):\n");
  print_stream256_state(&state);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);
  printf("\n Estado do Stream256 após stream256_squeezeblocks (OPTIMIZED):\n");
  print_stream256_state(&state);

  // Log do buffer gerado
  printf("Buffer após stream256_squeezeblocks (OPTIMIZED):\n");
  for (int i = 0; i < buflen; ++i) {
    printf("%02x ", buf[i]);
  }
  printf("\n");

  ctr = rej_eta(a->coeffs, N, buf, buflen);

  while (ctr < N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM256_BLOCKBYTES);
    printf("\nTamanho de stream256_state: %zu bytes\n", sizeof(stream256_state));

  }
}


// Programa principal - Teste da Função poly_uniform_eta
int main() {
    // Declaração dos polinômios e variáveis de tempo
    poly a_ref, a_optimized;
    uint64_t t[NTESTS];
    uint8_t seed[SEEDBYTES] = {
        0x72, 0x9a, 0xc0, 0x9a, 0xcf, 0x6d, 0x27, 0x8c,
        0x76, 0x41, 0xc3, 0xfe, 0x66, 0x78, 0x1a, 0xee,
        0x94, 0x95, 0x16, 0x12, 0xa7, 0xd6, 0x52, 0xc2,
        0x1e, 0x5f, 0xec, 0x12, 0xd6, 0x13, 0x8c, 0xa3,
     
    };
    uint16_t nonce = 1234;

    /*
    // Gera uma seed aleatória (mesma seed para ambas as funções)
    for (int i = 0; i < SEEDBYTES; i++) {
        seed[i] = rand() % 256;
    }
    */

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; i++) {
        // Limpar a memória do polinômio para evitar restos de testes anteriores
        memset(&a_ref, 0, sizeof(poly)); 
        t[i] = cpucycles();
        poly_uniform_eta(&a_ref, seed, nonce); // Chama a função de referência
        force_use(a_ref.coeffs, N);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_uniform_eta (REF):", t, NTESTS);
    uint64_t media_ref = median(t, NTESTS);

    // Teste para a função otimizada
    for (int i = 0; i < NTESTS; i++) {
        // Limpar a memória do polinômio para evitar restos de testes anteriores
        memset(&a_optimized, 0, sizeof(poly));        
        t[i] = cpucycles();
        poly_uniform_eta_optimized(&a_optimized, seed, nonce); // Chama a função otimizada
        force_use(a_optimized.coeffs, N);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_uniform_eta_optimized:", t, NTESTS);
    uint64_t media_optimized = median(t, NTESTS);

    // Comparar polinômios
    printf("Comparando os resultados dos polinômios:\n");
    if (!compare_polys(&a_ref, &a_optimized)) {
        printf("\nOs polinômios diferem!\n");
        printf("\nPolinômio A_REF:\n");
        exibe_polinomio(a_ref.coeffs);  // Exibe o polinômio de referência
        printf("\nPolinômio A_OPTIMIZED:\n");
        exibe_polinomio(a_optimized.coeffs);  // Exibe o polinômio otimizado
        return 1;  // Finaliza o programa se houver divergência
    } else {
        printf("\nOs polinômios são iguais.\n");
    }

    // Comparar o desempenho
    double speedup = (double)media_ref / media_optimized;
    printf("\nSpeed-up da função otimizada em relação à referência: %.2fx\n", speedup);

    return 0;
}

