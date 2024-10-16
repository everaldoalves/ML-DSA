//************************************************************************************************
// Autor: Everaldo Alves
// Data: 13 de Setembro/2024
// Função: poly_make_hint
// Descrição: Esta função usa uma dica para o processo de verificação de assinatura 
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

/*************************************************
* Name:        use_hint
*
* Description: Correct high bits according to hint.
*
* Arguments:   - int32_t a: input element
*              - unsigned int hint: hint bit
*
* Returns corrected high bits.
**************************************************/
int32_t use_hint(int32_t a, unsigned int hint) {
  int32_t a0, a1;

  a1 = decompose(&a0, a);
  if(hint == 0)
    return a1;

#if GAMMA2 == (Q-1)/32
  if(a0 > 0)
    return (a1 + 1) & 15;
  else
    return (a1 - 1) & 15;
#elif GAMMA2 == (Q-1)/88
  if(a0 > 0)
    return (a1 == 43) ?  0 : a1 + 1;
  else
    return (a1 ==  0) ? 43 : a1 - 1;
#endif
}

/*************************************************
* Name:        poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
void poly_use_hint(poly *b, const poly *a, const poly *h) {
  unsigned int i;  
  for(i = 0; i < N; ++i)
    b->coeffs[i] = use_hint(a->coeffs[i], h->coeffs[i]);
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

// Função auxiliar para ajustar o valor de a1 com base no hint
static inline int32x4_t use_hint_vector(int32x4_t a1_vec, int32x4_t a0_vec) {
    int32x4_t one_vec = vdupq_n_s32(1);  // Definindo o vetor 'one_vec' com valor 1
#if GAMMA2 == (Q - 1) / 32
    int32x4_t mask = vcgtq_s32(a0_vec, vdupq_n_s32(0));  // Verifica se a0 > 0
    a1_vec = vbslq_s32(mask, vaddq_s32(a1_vec, one_vec), vsubq_s32(a1_vec, one_vec));
    return vandq_s32(a1_vec, vdupq_n_s32(15));  // Aplicar máscara & 15
#elif GAMMA2 == (Q - 1) / 88
    int32x4_t zero_vec = vdupq_n_s32(0);
    int32x4_t forty_three_vec = vdupq_n_s32(43);

    int32x4_t mask_pos = vcgtq_s32(a0_vec, zero_vec);  // Verifica se a0 > 0
    int32x4_t mask_eq_43 = vceqq_s32(a1_vec, forty_three_vec);
    int32x4_t mask_eq_0 = vceqq_s32(a1_vec, zero_vec);

    a1_vec = vbslq_s32(mask_pos, vbslq_s32(mask_eq_43, zero_vec, vaddq_s32(a1_vec, one_vec)),
                       vbslq_s32(mask_eq_0, forty_three_vec, vsubq_s32(a1_vec, one_vec)));
    return a1_vec;
#endif
}

// Função use_hint vetorial adaptada para NEON
void poly_use_hint_neon(poly *b, const poly *a, const poly *h) {
    unsigned int i;
    int32x4_t a_vec, h_vec, b_vec, a0_vec, a1_vec;

    for (i = 0; i < N; i += 4) {
        // Carregar 4 coeficientes de 'a' e 'h' para vetores NEON
        a_vec = vld1q_s32(&a->coeffs[i]);
        h_vec = vld1q_s32((const int32_t *)&h->coeffs[i]);

        // Chamar a função decompose_vector para obter a1 (bits altos) e a0 (bits baixos)
        decompose_vector(&a1_vec, &a0_vec, a_vec);  // Altere aqui para a nova assinatura

        // Aplicar a dica (hint) para corrigir os bits altos
        int32x4_t hint_mask = vceqq_s32(h_vec, vdupq_n_s32(0));  // Verificar se hint é 0
        b_vec = vbslq_s32(hint_mask, a1_vec, use_hint_vector(a1_vec, a0_vec));  // Se hint for 0, usar a1; caso contrário, corrigir

        // Armazenar os resultados no polinômio de saída
        vst1q_s32(&b->coeffs[i], b_vec);
    }
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
    poly a, h, b_ref, b_neon;
    uint64_t t[NTESTS];

    // Inicializa os polinômios com dados aleatórios ou fixos
    initialize_poly(&a);
    initialize_poly(&h);

    // Teste para a função de referência
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_use_hint(&b_ref, &a, &h);
        force_use(&b_ref);
    }
    print_results("poly_use_hint (ref):", t, NTESTS);

    // Teste para a função otimizada com NEON
    for (int i = 0; i < NTESTS; ++i) {
        t[i] = cpucycles();
        poly_use_hint_neon(&b_neon, &a, &h);
        force_use(&b_neon);
    }
    print_results("poly_use_hint (NEON):", t, NTESTS);

    // Verifica se os resultados são idênticos
    if (!compare_polys(&b_ref, &b_neon)) {
        printf("\nOs resultados dos polinômios não são iguais\n");
        return 1;
    } else {
        printf("\nOs resultados dos polinômios são iguais\n");
    }

    return 0;
}
