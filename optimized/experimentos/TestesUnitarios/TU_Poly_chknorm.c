//************************************************************************************************
// Autor: Everaldo Alves
// Data: 13 de Setembro/2024
// Função: poly_chknorm
// Descrição: Esta função usa uma dica para o processo de verificação de assinatura 
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜50% nos ciclos de CPU
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


/*************************************************
* Name:        poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input coefficients were reduced by reduce32().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
int poly_chknorm(const poly *a, int32_t B) {
  unsigned int i;
  int32_t t;
  
  if(B > (Q-1)/8)
    return 1;

  /* It is ok to leak which coefficient violates the bound since
     the probability for each coefficient is independent of secret
     data but we must not leak the sign of the centralized representative. */
  for(i = 0; i < N; ++i) {
    /* Absolute value */
    t = a->coeffs[i] >> 31;
    t = a->coeffs[i] - (t & 2*a->coeffs[i]);

    if(t >= B) {
      
      return 1;
    }
  }
  
  return 0;
}

int poly_chknorm_neon(const poly *a, int32_t B) {
    unsigned int i;
    int32x4_t B_vec = vdupq_n_s32(B); // Vetor com valor B replicado
    int32x4_t t_vec, abs_vec, mask_vec;

    if (B > (Q - 1) / 8)
        return 1;

    for (i = 0; i < N; i += 4) {
        // Carrega 4 coeficientes
        int32x4_t coeff_vec = vld1q_s32(&a->coeffs[i]);

        // Calcula o valor absoluto
        mask_vec = vshrq_n_s32(coeff_vec, 31);      // t = a[i] >> 31
        abs_vec = vsubq_s32(coeff_vec, vandq_s32(mask_vec, vaddq_s32(coeff_vec, coeff_vec)));  // t = a - (t & 2*a)

        // Verifica se algum coeficiente é maior ou igual a B
        uint32x4_t cmp_result = vcgeq_s32(abs_vec, B_vec);  // Compara abs_vec >= B
        if (vmaxvq_u32(cmp_result) != 0) { // vmaxvq_u32 retorna o valor máximo no vetor
            return 1;
        }
    }

    return 0;
}

// Função para garantir que o compilador não elimine o resultado
void force_use_int(int result) {
    volatile int dummy = result;  // 'dummy' garante que o valor de 'result' seja utilizado
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
void exibe_polinomio(const poly *p) {
    for (int i = 0; i < N; i++) {
        printf("%d ", p->coeffs[i]);
    }
    printf("\n");
}
// Teste da função original
void test_poly_chknorm(const poly *a, int32_t B) {
    uint64_t t[NTESTS];
    volatile int result;

    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        result = poly_chknorm(a, B);
        
    }
    print_results("poly_chknorm (ref):", t, NTESTS);
    printf("Resultado: %d\n", result);
    
}

// Teste da função otimizada com NEON
void test_poly_chknorm_neon(const poly *a, int32_t B) {
    uint64_t t[NTESTS];
    volatile int result;

    for (int i = 0; i < NTESTS; i++) {
        t[i] = cpucycles();
        result = poly_chknorm_neon(a, B);
        
    }
    print_results("poly_chknorm (NEON):", t, NTESTS);
    printf("Resultado: %d\n", result);
    
}

// Função principal para rodar os testes
int main() {
    poly a;
    int32_t B = (Q - 1) / 8;

    // Inicializar o polinômio
    initialize_poly(&a);

    // Executar os testes
    test_poly_chknorm(&a, B);           // Teste da função de referência
    test_poly_chknorm_neon(&a, B);      // Teste da função otimizada com NEON


    return 0;
}