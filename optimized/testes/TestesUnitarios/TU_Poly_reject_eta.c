//************************************************************************************************
// Autor: Everaldo Alves
// Data: 15 de Setembro/2024
// Função: poly_reject_eta
// Descrição: Esta função amostra um polinômio com coeficientes uniformemente aleatórios
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Função otimizada com NEON está gerando resultados DIVERGENTES
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
#define ETA 4
#define N 256
#define Q 8380417
#define NTESTS 10
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
*************************************************/
static unsigned int rej_eta(int32_t *a,
                            unsigned int len,
                            const uint8_t *buf,
                            unsigned int buflen)
{
  unsigned int ctr, pos;
  uint32_t t0, t1;
 

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

   return ctr;
}

static unsigned int rej_eta_unroll(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0, pos = 0;
    uint32_t t0, t1;
    
    // Ajuste o fator de desenrolamento (unroll_factor). Teste com diferentes valores como 8 ou 16.
    const int unroll_factor = 4;

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

// Versão otimizada com NEON
static unsigned int rej_eta_neon(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0, pos = 0;
    uint8x16_t buf_vec;
    uint32x4_t t0_vec, t1_vec;
    uint32x4_t mask_t0, mask_t1;
    int32_t t0_array[4], t1_array[4];  // Arrays temporários para os valores de t0 e t1
    uint32_t mask_t0_array[4], mask_t1_array[4];  // Arrays temporários para as máscaras

    // ETA Flexível
    int32x4_t eta_vec, limit_vec;
#if ETA == 2
    eta_vec = vdupq_n_s32(2);   // ETA = 2
    limit_vec = vdupq_n_u32(15);  // Limite para t0 e t1 (valores abaixo de 15 são aceitos)
#elif ETA == 4
    eta_vec = vdupq_n_s32(4);   // ETA = 4
    limit_vec = vdupq_n_u32(9);  // Limite para t0 e t1 (valores abaixo de 9 são aceitos)
#endif

    while (ctr + 8 <= len && pos + 8 <= buflen) {
        // Carrega 16 bytes do buffer para processar 8 valores t0 e t1 (2 por byte)
        buf_vec = vld1q_u8(&buf[pos]);

        // Extrai os 4 bits menos e mais significativos (t0 e t1) para cada byte
        t0_vec = vandq_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(buf_vec)))), vdupq_n_u32(0x0F));
        t1_vec = vshrq_n_u32(vmovl_u16(vget_low_u16(vmovl_u8(vget_low_u8(buf_vec)))), 4);

        // Aplica a máscara para aceitar apenas valores t0 e t1 menores que os limites
        mask_t0 = vcltq_u32(t0_vec, limit_vec);  
        mask_t1 = vcltq_u32(t1_vec, limit_vec);  

        // Ajusta os valores de t0 e t1 para o intervalo [-ETA, ETA]
        t0_vec = vsubq_s32(eta_vec, vreinterpretq_s32_u32(t0_vec));
        t1_vec = vsubq_s32(eta_vec, vreinterpretq_s32_u32(t1_vec));

        // Depuração: Mostrar valores de t0 e t1 antes de armazená-los
        vst1q_s32(t0_array, t0_vec);
        vst1q_s32(t1_array, t1_vec);
        vst1q_u32(mask_t0_array, mask_t0);
        vst1q_u32(mask_t1_array, mask_t1);

        // Itera sobre os arrays e armazena os valores nos coeficientes
        for (int i = 0; i < 4 && ctr < len; i++) {
            if (mask_t0_array[i]) {
                a[ctr++] = t0_array[i];
                printf("NEON - a[%d]: %d\n", ctr-1, t0_array[i]);  // Depuração
            }
            if (ctr < len && mask_t1_array[i]) {
                a[ctr++] = t1_array[i];
                printf("NEON - a[%d]: %d\n", ctr-1, t1_array[i]);  // Depuração
            }
        }

        pos += 8;  // Avança o buffer
    }

    // Processar coeficientes restantes
    while (ctr < len && pos < buflen) {
        uint32_t t0 = buf[pos] & 0x0F;
        uint32_t t1 = buf[pos++] >> 4;

        printf("NEON (restante) - t0: %d, t1: %d\n", t0, t1);  // Depuração

#if ETA == 2
        if (t0 < 15) {
            t0 = t0 - (205*t0 >> 10)*5;  // Rejeição de t0 para ETA 2
            a[ctr++] = 2 - t0;
            printf("NEON (restante) - a[%d]: %d\n", ctr-1, 2 - t0);  // Depuração
        }
        if (t1 < 15 && ctr < len) {
            t1 = t1 - (205*t1 >> 10)*5;  // Rejeição de t1 para ETA 2
            a[ctr++] = 2 - t1;
            printf("NEON (restante) - a[%d]: %d\n", ctr-1, 2 - t1);  // Depuração
        }
#elif ETA == 4
        if (t0 < 9) {
            a[ctr++] = 4 - t0;
            printf("NEON (restante) - a[%d]: %d\n", ctr-1, 4 - t0);  // Depuração
        }
        if (t1 < 9 && ctr < len) {
            a[ctr++] = 4 - t1;
            printf("NEON (restante) - a[%d]: %d\n", ctr-1, 4 - t1);  // Depuração
        }
#endif
    }

    return ctr;
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
    unsigned int len = N;
    unsigned int sampled_ref = 0, sampled_neon = 0, sampled_unroll = 0;

    // Gera um buffer de bytes aleatórios para a função de rejeição
    for (int i = 0; i < sizeof(buf); i++) {
        buf[i] = rand() % 256;
    }

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_ref = rej_eta(a_ref.coeffs, len, buf, sizeof(buf));  // Função de referência
        force_use(a_ref.coeffs, sampled_ref);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_reject(REF):", t, NTESTS);
    uint64_t media_ref = median(t, NTESTS);

    // Teste para a função que usa loop unrolling e prefetch manual
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_unroll = rej_eta_unroll(a_unroll.coeffs, len, buf, sizeof(buf));  // Função otimizada com unrolling
        force_use(a_unroll.coeffs, sampled_unroll);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_reject (UNROLLING):", t, NTESTS);
    uint64_t media_unroll = median(t, NTESTS);

    // Teste para a função otimizada com NEON
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_neon = rej_eta_neon(a_neon.coeffs, len, buf, sizeof(buf));  // Função otimizada com NEON
        force_use(a_neon.coeffs, sampled_neon);  // Garante que os resultados não sejam otimizados
    }
    print_results("poly_reject (NEON):", t, NTESTS);
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

    // Exibir os polinômios
    exibe_polinomio(a_ref.coeffs);
    printf("\n");
    exibe_polinomio(a_unroll.coeffs);
    printf("\n");
    exibe_polinomio(a_neon.coeffs);

    return 0;
}