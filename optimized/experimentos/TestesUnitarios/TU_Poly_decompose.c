//************************************************************************************************
// Autor: Everaldo Alves
// Data: 12 de Setembro/2024
// Função: poly_decompose
// Descrição: Esta função realiza a decomposição de dois polinômios 
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜25% no tempo de execução
//************************************************************************************************

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <arm_neon.h>
#include "cpucycles.h"

#define N 256
#define NTESTS 10000
#define Q 8380417
#define GAMMA1 (1 << 17)
#define GAMMA2 ((Q-1)/88)

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
* Name:        decompose
*
* Description: For finite field element a, compute high and low bits a0, a1 such
*              that a mod^+ Q = a1*ALPHA + a0 with -ALPHA/2 < a0 <= ALPHA/2 except
*              if a1 = (Q-1)/ALPHA where we set a1 = 0 and
*              -ALPHA/2 <= a0 = a mod^+ Q - Q < 0. Assumes a to be standard
*              representative.
*
* Arguments:   - int32_t a: input element
*              - int32_t *a0: pointer to output element a0
*
* Returns a1.
**************************************************/
int32_t decompose(int32_t *a0, int32_t a) {
    int32_t a1;

    a1 = (a + 127) >> 7;  // Calcular a1
    #if GAMMA2 == (Q - 1) / 32
        a1 = (a1 * 1025 + (1 << 21)) >> 22;
        a1 &= 15;
    #elif GAMMA2 == (Q - 1) / 88
        a1 = (a1 * 11275 + (1 << 23)) >> 24;
        a1 ^= ((43 - a1) >> 31) & a1;
    #endif

    *a0 = a - a1 * 2 * GAMMA2;   // Calcular a0
    *a0 -= (((Q - 1) / 2 - *a0) >> 31) & Q;

    return a1;
}

// Função de decompose vetorial adaptada para NEON
static inline void decompose_vector(int32x4_t *a1_vec, int32x4_t *a0_vec, int32x4_t a_vec) {
    int32x4_t a1 = vaddq_s32(a_vec, vdupq_n_s32(127));   // a1 = (a + 127)
    a1 = vshrq_n_s32(a1, 7);                            // a1 = (a + 127) >> 7

    // Para GAMMA2 == (Q-1)/32
    a1 = vmlaq_n_s32(vdupq_n_s32(1 << 21), a1, 1025);  // a1 * 1025 + (1 << 21)
    a1 = vshrq_n_s32(a1, 22);                          // a1 >> 22
    a1 = vandq_s32(a1, vdupq_n_s32(15));               // a1 & 15

    // Calcular a0: a0 = a - a1 * 2 * GAMMA2
    int32x4_t a0 = vmlsq_n_s32(a_vec, a1, 2 * GAMMA2);

    // Ajustar a0: a0 -= (((Q-1)/2 - a0) >> 31) & Q;
    a0 = vsubq_s32(a0, vandq_s32(vshrq_n_s32(vsubq_s32(vdupq_n_s32((Q-1)/2), a0), 31), vdupq_n_s32(Q)));

    // Salvar os resultados
    *a1_vec = a1;
    *a0_vec = a0;
}


void poly_decompose_neon(poly *a1, poly *a0, const poly *a) {
    unsigned int i;
    int32x4_t a_vec, a1_vec, a0_vec;

    for (i = 0; i < N; i += 4) {
        // Carregar 4 coeficientes do polinômio 'a'
        a_vec = vld1q_s32(&a->coeffs[i]);

        // Aplica decompose vetorialmente em 4 coeficientes
        decompose_vector(&a1_vec, &a0_vec, a_vec);

        // Salvar resultados
        vst1q_s32(&a1->coeffs[i], a1_vec);
        vst1q_s32(&a0->coeffs[i], a0_vec);
    }
}

// Função de referência (não otimizada)
void poly_decompose(poly *a1, poly *a0, const poly *a) {
    for (unsigned int i = 0; i < N; ++i) {
        a1->coeffs[i] = decompose(&a0->coeffs[i], a->coeffs[i]);
    }
}

// Função para inicializar polinômios com valores aleatórios
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % Q;
    }
}

// Função para garantir que o compilador não elimine o resultado
void force_use(poly *p) {
    volatile int32_t sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += p->coeffs[i];
    }
}

// Função para comparar dois polinômios
int compare_polys(poly *a, poly *b) {
for (int i = 0; i < N; ++i) {
    if (a->coeffs[i] != b->coeffs[i]) {
        return 0; // Diferença encontrada
    }
}
return 1; // Polinômios iguais
}

void exibe_polinomio(poly *p) {
    for (int i = 0; i < N; i++) {
        printf("%d ", p->coeffs[i]);
    }
    printf("\n");
}

int main() {
    poly a1, a0, a, b1, b0, b;
    uint64_t t[NTESTS];
    
    // Inicializar polinômio
    initialize_poly(&a);
    b = a;
    printf("Polinômio original: \n");
    exibe_polinomio(&a);
    printf("\n");
    printf("Polinômio Otimizado: \n");
    exibe_polinomio(&b);
    printf("\n");

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_decompose(&a1, &a0, &a);
        force_use(&a1);  // Garante que os resultados da função não sejam otimizados
        force_use(&a0);  // Garante que os resultados da função não sejam otimizados        
    }
    print_results("poly_decompose:", t, NTESTS);

    // Teste para a função otimizada com NEON
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_decompose_neon(&b1, &b0, &b);
        force_use(&a1);  // Garante que os resultados da função não sejam otimizados
        force_use(&a0);  // Garante que os resultados da função não sejam otimizados
    }
    print_results("poly_decompose_neon:", t, NTESTS);

    // Verificar se os resultados são iguais
        if (!compare_polys(&a1, &b1)) {
            printf("\nOs resultados de a1 e b1 são diferentes \n");
            //return 1;
        }
          if (!compare_polys(&a0, &b0)) {
            printf("\n Os resultados de a0 e b0 são diferentes \n");
            //return 1;
        }


    return 0;
}