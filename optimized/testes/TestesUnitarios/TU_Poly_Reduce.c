#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

// Definições de constantes e estruturas
#define N 256
#define Q 8380417
#define REPEAT 10000
#define INNER_REPEAT 1000

typedef struct {
    int32_t coeffs[N] __attribute__((aligned(16)));
} poly;

// Função para inicializar o polinômio com valores aleatórios
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % (2 * Q) - Q;  // Valores aleatórios em [-Q, Q]
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

// Função de referência: reduce32
int32_t reduce32(int32_t a) {
    int32_t t;
    t = (a + (1 << 22)) >> 23;
    t = a - t * Q;
    return t;
}

// Função de referência: poly_reduce_ref
void poly_reduce_ref(poly *a) {
    for (unsigned int i = 0; i < N; ++i) {
        a->coeffs[i] = reduce32(a->coeffs[i]);
    }
}

void poly_reduce_everaldo(poly *a) {
    int32x4_t q_vec = vdupq_n_s32(Q);
    int32x4_t shift_vec = vdupq_n_s32(1 << 22);

    for (unsigned int i = 0; i < N; i+=8) {
        // Carregar 8 coeficientes
        int32x4_t a_vec1 = vld1q_s32(&a->coeffs[i]);
        int32x4_t a_vec2 = vld1q_s32(&a->coeffs[i + 4]);

        // Calcular t = (a + (1 << 22)) >> 23 para ambos
        int32x4_t t_vec1 = vshrq_n_s32(vaddq_s32(a_vec1, shift_vec), 23);
        int32x4_t t_vec2 = vshrq_n_s32(vaddq_s32(a_vec2, shift_vec), 23);

        // Calcular a - t * Q
        int32x4_t result1 = vsubq_s32(a_vec1, vmulq_s32(t_vec1, q_vec));
        int32x4_t result2 = vsubq_s32(a_vec2, vmulq_s32(t_vec2, q_vec));

        // Armazenar resultados
        vst1q_s32(&a->coeffs[i], result1);
        vst1q_s32(&a->coeffs[i + 4], result2);
    }
}

// Função otimizada usando NEON: poly_reduce
void poly_reduce_optimized(poly *a) {
    unsigned int i;
    int32x4_t q_vec = vdupq_n_s32(Q);
    int32x4_t shift_vec = vdupq_n_s32(1 << 22);

    for (i = 0; i < N; i += 4) {
        int32x4_t a_vec = vld1q_s32(&a->coeffs[i]);

        int32x4_t t_vec = vaddq_s32(a_vec, shift_vec);
        t_vec = vshrq_n_s32(t_vec, 23);  // t = (a + (1 << 22)) >> 23
        t_vec = vmulq_s32(t_vec, q_vec); // t * Q

        int32x4_t result = vsubq_s32(a_vec, t_vec); // a - t * Q
        vst1q_s32(&a->coeffs[i], result);
    }
}

// Função otimizada usando NEON: poly_reduce_optimized_v2
void poly_reduce_optimized_v2(poly *a) {
    unsigned int i;
    int32x4_t q_vec = vdupq_n_s32(Q);
    int32x4_t shift_vec = vdupq_n_s32(1 << 22);

    for (i = 0; i < N; i += 8) {
        // Carregar 8 coeficientes
        int32x4_t a_vec1 = vld1q_s32(&a->coeffs[i]);
        int32x4_t a_vec2 = vld1q_s32(&a->coeffs[i + 4]);

        // Calcular t_vec para ambos os vetores
        int32x4_t t_vec1 = vshrq_n_s32(vaddq_s32(a_vec1, shift_vec), 23);
        int32x4_t t_vec2 = vshrq_n_s32(vaddq_s32(a_vec2, shift_vec), 23);

        // Calcular resultado usando vmlsq_s32
        int32x4_t result1 = vmlsq_s32(a_vec1, t_vec1, q_vec);
        int32x4_t result2 = vmlsq_s32(a_vec2, t_vec2, q_vec);

        // Armazenar resultados
        vst1q_s32(&a->coeffs[i], result1);
        vst1q_s32(&a->coeffs[i + 4], result2);
    }
}


//******************************************************************************** Versão FINAL ************************************************************************************************

int32x4_t reduce32_vector(int32x4_t a) {
    int32x4_t t;
    int32x4_t q = vdupq_n_s32(Q);                 // Vetor constante com o valor de Q
    int32x4_t shift_val = vdupq_n_s32(1 << 22);    // Valor a ser somado
    int32x4_t shift_bits = vdupq_n_s32(23);        // Número de bits para o deslocamento

    // Primeiro passo: (a + (1 << 22)) >> 23
    t = vaddq_s32(a, shift_val);                   // Soma 1 << 22 a cada elemento de 'a'
    t = vshrq_n_s32(t, 23);                        // Desloca os valores 23 bits à direita

    // Segundo passo: t * Q
    t = vmulq_s32(t, q);                           // Multiplica t pelo valor de Q

    // Terceiro passo: a - t * Q
    a = vsubq_s32(a, t);                           // Subtrai t * Q de a

    return a;                                      // Retorna o valor reduzido
}

void poly_reduce_final(poly *a) {
    unsigned int i;
    // Processa 4 coeficientes por vez
    for (i = 0; i < N; i += 4) {
        // Carregar 4 coeficientes de 'a' nos registradores NEON
        int32x4_t vec_a = vld1q_s32(&a->coeffs[i]);

        // Aplicar a redução a 4 coeficientes simultaneamente
        vec_a = reduce32_vector(vec_a);

        // Armazenar o resultado de volta em 'a'
        vst1q_s32(&a->coeffs[i], vec_a);
    }
}

//******************************************************************************************************************************************************************************************************


// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}

int main() {
    // Semente para números aleatórios
    srand(time(NULL));

    // Inicializar dois polinômios com os mesmos valores
    poly __attribute__ ((aligned (16))) poly_ref, poly_opt;
   

    // Medir CICLOS da função de referência e otimizada
    uint64_t ref_time = 0, opt_time = 0;
    uint64_t start_cycles, end_cycles;

    // Inicializar os polinômios
    initialize_poly(&poly_ref);
    poly_opt = poly_ref;

    // Medir ciclos da função de referência
    for (int i = 0; i < REPEAT; i++) {
        start_cycles = cpucycles();
        poly_reduce_ref(&poly_ref);
        end_cycles = cpucycles();
        ref_time += end_cycles - start_cycles;
    }

    // Medir ciclos da função otimizada
    for (int i = 0; i < REPEAT; i++) {
        start_cycles = cpucycles();
        poly_reduce_everaldo(&poly_opt);
        end_cycles = cpucycles();
        opt_time += end_cycles - start_cycles;
    }

    compare_polys(&poly_ref, &poly_opt);

    // Calcular o speed-up
    double avg_ref_time = (double) ref_time / REPEAT;
    double avg_opt_time = (double) opt_time / REPEAT;
    double speedup = avg_ref_time / avg_opt_time;

    // Mostrar o speed-up
    printf("Média de ciclos da função de referência: %.2f\n", avg_ref_time);
    printf("Média de ciclos da função otimizada: %.2f\n", avg_opt_time);
    printf("Speed-up: %.2f X\n", speedup);

    return 0;
}
