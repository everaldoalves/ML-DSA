//************************************************************************************************
// Autor: Everaldo Alves
// Data: 13 de Setembro/2024
// Função: poly_make_hint
// Descrição: Esta função cria uma dica para o processo de verificação de assinatura 
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜20% nos ciclos de CPU
//************************************************************************************************


#include <arm_neon.h>
#include "cpucycles.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define N 256
#define Q 8380417
#define GAMMA1 (1 << 17)
#define GAMMA2 ((Q-1)/88)
#define NTESTS 10000

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



// Função de referência make_hint
unsigned int make_hint(int32_t a0, int32_t a1) {
    if (a0 > GAMMA2 || a0 < -GAMMA2 || (a0 == -GAMMA2 && a1 != 0))
        return 1;
    return 0;
}

// Função de referência poly_make_hint
unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {
    unsigned int i, s = 0;
    for (i = 0; i < N; ++i) {
        h->coeffs[i] = make_hint(a0->coeffs[i], a1->coeffs[i]);
        s += h->coeffs[i];
    }
    return s;
}

// Função otimizada NEON make_hint_vector
static inline uint32x4_t make_hint_vector(int32x4_t a0_vec, int32x4_t a1_vec) {
    int32x4_t gamma2_vec = vdupq_n_s32(GAMMA2);
    uint32x4_t hint_vec = vcgtq_s32(a0_vec, gamma2_vec);  // a0 > GAMMA2
    hint_vec = vorrq_u32(hint_vec, vcgtq_s32(vnegq_s32(a0_vec), gamma2_vec));  // a0 < -GAMMA2

    uint32x4_t condition = vceqq_s32(a0_vec, vdupq_n_s32(-GAMMA2));
    hint_vec = vorrq_u32(hint_vec, vandq_u32(condition, vmvnq_u32(vceqq_s32(a1_vec, vdupq_n_s32(0)))));
    return hint_vec;
}

// Função otimizada poly_make_hint_neon
unsigned int poly_make_hint_neon(poly *h, const poly *a0, const poly *a1) {
    unsigned int i, s = 0;
    uint32x4_t hint_vec, sum_vec = vdupq_n_u32(0);

    for (i = 0; i < N; i += 4) {
        int32x4_t a0_vec = vld1q_s32(&a0->coeffs[i]);
        int32x4_t a1_vec = vld1q_s32(&a1->coeffs[i]);

        hint_vec = make_hint_vector(a0_vec, a1_vec);
        vst1q_u32((uint32_t *)&h->coeffs[i], hint_vec);

        sum_vec = vaddq_u32(sum_vec, hint_vec);
    }

    // Reduz o vetor sum_vec para um único valor
    s = vgetq_lane_u32(sum_vec, 0) + vgetq_lane_u32(sum_vec, 1) + vgetq_lane_u32(sum_vec, 2) + vgetq_lane_u32(sum_vec, 3);
    return s;
}


// Função para garantir que o compilador não elimine o resultado
void force_use(poly *p) {
    volatile int32_t sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += p->coeffs[i];
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
        p->coeffs[i] = rand() % (2 * GAMMA2) - GAMMA2;  // Valores aleatórios entre -GAMMA2 e GAMMA2
    }
}

// Exibe o polinômio
void exibe_polinomio(const poly *p) {
    for (int i = 0; i < N; i++) {
        printf("%d ", p->coeffs[i]);
    }
    printf("\n");
}

// Função principal de teste
int main() {
    poly a1, a0, a, b1, b0, b, h_ref, h_opt;
    uint64_t t[NTESTS];

    // Inicializar polinômios de entrada
    initialize_poly(&a0);
    initialize_poly(&a1);
    b0 = a0;
    b1 = a1;

    // Teste da função de referência
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_make_hint(&h_ref, &a0, &a1);
        force_use(&h_ref);  // Garante que os resultados da função não sejam otimizados
    }
    print_results("poly_make_hint (ref):", t, NTESTS);

    // Teste da função otimizada NEON
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_make_hint_neon(&h_opt, &b0, &b1);
        force_use(&h_opt);  // Garante que os resultados da função não sejam otimizados
    }
    print_results("poly_make_hint (NEON):", t, NTESTS);

    // Verificação dos resultados
    if (!compare_polys(&h_ref, &h_opt)) {
        printf("Os resultados dos polinômios são diferentes!\n");
    } else {
        printf("Os resultados dos polinômios são iguais.\n");
    }

    return 0;
}
