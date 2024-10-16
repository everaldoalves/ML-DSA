//************************************************************************************************
// Autor: Everaldo Alves
// Data: 13 de Setembro/2024
// Função: rej_uniform
// Descrição: Esta função faz a amostragem por rejeição de coeficientes uniformemente aleatórios
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: A versão mais rápida foi: NEON2. Speed-up em relação à função de referência: 1.42x
//************************************************************************************************
// Atenção:
//Processamento de vetores NEON: Quando utilizamos NEON, os valores são processados em blocos de 4 elementos simultaneamente, o que pode levar a diferenças se 
// o tamanho do buffer (buf) ou do polinômio (a) não for múltiplo de 4. Garanta que o processamento está correto e que os valores restantes são tratados adequadamente.


#include <arm_neon.h>
#include "cpucycles.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define N 256
#define Q 8380417
#define GAMMA1 (1 << 17)
#define GAMMA2 ((Q-1)/88)
#define NTESTS 100000

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
* Name:        rej_uniform
*
* Description: Sample uniformly random coefficients in [0, Q-1] by
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

// Versão otimizada para NEON - Esta versão está funcionando!!!!! Porém o ganho é modesto (cerca de 1 ciclo)
static unsigned int rej_uniform_neon(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0;
    unsigned int pos = 0;

    uint32x4_t mask = vdupq_n_u32(0x7FFFFF);  // Máscara para limitar a 23 bits
    uint32x4_t Qvec = vdupq_n_u32(Q);         // Vetor com o valor de Q
    uint8x16_t buf_vec;

    // Processar blocos de 12 bytes (4 valores de 3 bytes cada)
    while (ctr < len && pos + 12 <= buflen) {
        buf_vec = vld1q_u8(&buf[pos]);  // Carregar 16 bytes do buffer

        // Reconstituir 4 valores de 3 bytes em inteiros de 32 bits
        uint32x4_t t = {
            buf[pos] | ((uint32_t)buf[pos+1] << 8) | ((uint32_t)buf[pos+2] << 16),
            buf[pos+3] | ((uint32_t)buf[pos+4] << 8) | ((uint32_t)buf[pos+5] << 16),
            buf[pos+6] | ((uint32_t)buf[pos+7] << 8) | ((uint32_t)buf[pos+8] << 16),
            buf[pos+9] | ((uint32_t)buf[pos+10] << 8) | ((uint32_t)buf[pos+11] << 16)
        };

        t = vandq_u32(t, mask);  // Aplicar a máscara de 23 bits

        // Verificar se os valores são menores que Q
        uint32x4_t lt_Q = vcltq_u32(t, Qvec);  // Comparação vetorial (t < Q)

        // Usar NEON para evitar ramificações e armazenar resultados válidos
        if (vgetq_lane_u32(lt_Q, 0)) a[ctr++] = vgetq_lane_u32(t, 0);
        if (ctr < len && vgetq_lane_u32(lt_Q, 1)) a[ctr++] = vgetq_lane_u32(t, 1);
        if (ctr < len && vgetq_lane_u32(lt_Q, 2)) a[ctr++] = vgetq_lane_u32(t, 2);
        if (ctr < len && vgetq_lane_u32(lt_Q, 3)) a[ctr++] = vgetq_lane_u32(t, 3);

        pos += 12;
    }

    // Processar os bytes restantes fora do loop principal
    while (ctr < len && pos + 3 <= buflen) {
        uint32_t t = buf[pos] | ((uint32_t)buf[pos+1] << 8) | ((uint32_t)buf[pos+2] << 16);
        t &= 0x7FFFFF;

        if (t < Q) {
            a[ctr++] = t;
        }
        pos += 3;
    }

    return ctr;
}

// Esta foi A SEGUNDA MELHOR VERSÃO!!!! ELA FAZ USO DE LOOP UNROLLING PARA PROCESSAR 4 VALORES POR VEZ. ALÉM DISSO, ELA FAZ USO DE PREFETCH MANUAL PARA CARREGAR OS PRÓXIMOS 64 BYTES DO BUFFER
static unsigned int rej_uniform_unrolling(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0;
    unsigned int pos = 0;
    uint32_t t;

    // Desenrolar o loop para processar 4 entradas por vez
    while (ctr + 4 <= len && pos + 12 <= buflen) {
        // Prefetch manual para as próximas 64 bytes do buffer
        __builtin_prefetch(&buf[pos + 64], 0, 1);  // Carregar na cache

        // Processar 4 valores consecutivos
        for (int j = 0; j < 4; j++) {
            t  = buf[pos++];
            t |= (uint32_t)buf[pos++] << 8;
            t |= (uint32_t)buf[pos++] << 16;
            t &= 0x7FFFFF;

            if (t < Q) {
                a[ctr++] = t;
            }
        }
    }

    // Processar os elementos restantes (se houver)
    while (ctr < len && pos + 3 <= buflen) {
        t  = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            a[ctr++] = t;
        }
    }

    return ctr;
}



// Esta versão está funcionando!!!!!
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
    poly a_ref, a_neon,a_neon2,a_unroll;
    uint8_t buf[1024];
    uint64_t t[NTESTS];
    unsigned int len = N;
    unsigned int sampled_ref, sampled_neon, sampled_neon2,sampled_unroll;

    // Gerar bytes aleatórios para o buffer
    for (int i = 0; i < 1024; i++) {
        buf[i] = rand() % 256;
    }

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_ref = rej_uniform(a_ref.coeffs, len, buf, sizeof(buf));
        force_use(a_ref.coeffs, sampled_ref);  // Garante que os resultados não sejam otimizados
    }
    print_results("rej_uniform (REF):", t, NTESTS);
    uint64_t media_ref = median(t, NTESTS);

     // Teste para a função que usa loop unrolling e prefetch manual
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_unroll = rej_uniform_unrolling(a_unroll.coeffs, len, buf, sizeof(buf));
        force_use(a_unroll.coeffs, sampled_unroll);  // Garante que os resultados não sejam otimizados
    }
    print_results("rej_uniform (UNROLLING):", t, NTESTS);
    uint64_t media_unroll = median(t, NTESTS);

    // Teste para a função otimizada com NEON
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_neon = rej_uniform_neon(a_neon.coeffs, len, buf, sizeof(buf));
        force_use(a_neon.coeffs, sampled_neon);  // Garante que os resultados não sejam otimizados
    }
    print_results("rej_uniform (NEON):", t, NTESTS);
    uint64_t media_neon = median(t, NTESTS);

    // Teste para a função otimizada com NEON 2
    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        sampled_neon2 = rej_uniform_neon2(a_neon2.coeffs, len, buf, sizeof(buf));
        force_use(a_neon2.coeffs, sampled_neon2);  // Garante que os resultados não sejam otimizados
    }
    print_results("rej_uniform (NEON2):", t, NTESTS);
    uint64_t media_neon2 = median(t, NTESTS);

    // Verificar se os resultados são iguais
    if (sampled_ref != sampled_neon || !compare_polys(&a_ref, &a_neon) || sampled_ref != sampled_unroll || !compare_polys(&a_ref, &a_unroll) || sampled_ref != sampled_neon2 || !compare_polys(&a_ref, &a_neon2)) {
        printf("\nOs resultados dos polinômios são diferentes\n");
        printf("\n Polinômio A_REF:\n");
        exibe_polinomio(a_ref.coeffs);
        printf("\n Polinômio A_UNROLL:\n");
        exibe_polinomio(a_unroll.coeffs);
        printf("Polinômio A_NEON:\n");
        exibe_polinomio(a_neon.coeffs);    
        printf("Polinômio A_NEON2:\n");
        exibe_polinomio(a_neon2.coeffs);       
        return 1;
    }

    printf("\nOs resultados dos polinômios são iguais\n");

    uint64_t fastest_time = media_ref;
    char *fastest_version = "Referência";

    if (media_ref < media_unroll && media_ref < media_neon && media_ref < media_neon2) {
        printf("A função de referência é a mais rápida\n");                
    } else if (media_unroll < media_ref && media_unroll < media_neon && media_unroll < media_neon2) {
        printf("A função com loop unrolling é a mais rápida\n");
        fastest_version = "Unrolling";
        fastest_time = media_unroll;
    } else if (media_neon < media_ref && media_neon < media_unroll && media_neon < media_neon2) {
        printf("A função com NEON é a mais rápida\n");
        fastest_version = "NEON";
        fastest_time = media_neon;
    } else {
        printf("A função com NEON2 é a mais rápida\n");
        fastest_version = "NEON2";
        fastest_time = media_neon2;
    }

    double speedup = (double)media_ref / fastest_time;

    printf("\nA versão mais rápida foi: %s\n", fastest_version);
    printf("Speed-up em relação à função de referência: %.2fx\n", speedup);

    return 0;
}