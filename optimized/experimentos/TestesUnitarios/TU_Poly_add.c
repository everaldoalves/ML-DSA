#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define N 256
#define Q 8380417
#define REPEAT 100000

typedef struct {
    int32_t coeffs[N];
} poly;

#define DBENCH_START(time_var) time_var = cpucycles()
#define DBENCH_STOP(t, time_var) t += cpucycles() - time_var - timing_overhead

// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}

// Função para calcular a sobrecarga de medição de tempo
uint64_t timing_overhead = 0;
void measure_timing_overhead(void) {
    uint64_t start = cpucycles();
    uint64_t end = cpucycles();
    timing_overhead = end - start;
}

// Função de referência: poly_add
void poly_add_ref(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    for (i = 0; i < N; ++i) {
        c->coeffs[i] = a->coeffs[i] + b->coeffs[i];
    }
}

// Função otimizada usando NEON: poly_add
void poly_add_optimized(poly *c, const poly *a, const poly *b) {
    unsigned int i;

    for (i = 0; i < N; i += 4) {
        int32x4_t a_vec = vld1q_s32(&a->coeffs[i]);  // Carrega 4 coeficientes de a
        int32x4_t b_vec = vld1q_s32(&b->coeffs[i]);  // Carrega 4 coeficientes de b

        int32x4_t c_vec = vaddq_s32(a_vec, b_vec);   // Soma vetorial dos coeficientes

        vst1q_s32(&c->coeffs[i], c_vec);             // Armazena o resultado em c
    }
}

void poly_add_unrolled(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    for (i = 0; i < N; i += 16) {
        // Carregar 8 coeficientes de 'a' e 'b' (2 blocos de 8)
        int32x4x2_t vec_a1 = vld1q_s32_x2(&a->coeffs[i]);
        int32x4x2_t vec_b1 = vld1q_s32_x2(&b->coeffs[i]);

        int32x4x2_t vec_a2 = vld1q_s32_x2(&a->coeffs[i+8]);
        int32x4x2_t vec_b2 = vld1q_s32_x2(&b->coeffs[i+8]);

        // Somar os coeficientes
        int32x4x2_t vec_c1, vec_c2;
        vec_c1.val[0] = vaddq_s32(vec_a1.val[0], vec_b1.val[0]);
        vec_c1.val[1] = vaddq_s32(vec_a1.val[1], vec_b1.val[1]);

        vec_c2.val[0] = vaddq_s32(vec_a2.val[0], vec_b2.val[0]);
        vec_c2.val[1] = vaddq_s32(vec_a2.val[1], vec_b2.val[1]);

        // Armazenar o resultado de volta em 'c'
        vst1q_s32_x2(&c->coeffs[i], vec_c1);
        vst1q_s32_x2(&c->coeffs[i+8], vec_c2);
    }
}


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

int main() {
    // Medir a sobrecarga de medição de tempo
    measure_timing_overhead();

    // Inicializar três polinômios
    poly poly_ref_a, poly_ref_b, poly_ref_c, poly_opt_a, poly_opt_b, poly_opt_c, poly_opt_unroll_a, poly_opt_unroll_b, poly_opt_unroll_c;
    initialize_poly(&poly_ref_a);
    initialize_poly(&poly_ref_b);
    
    poly_opt_a = poly_ref_a;  // Copiar valores para os polinômios otimizados
    poly_opt_b = poly_ref_b;
    
    // Variáveis para medir o tempo
    uint64_t ref_time = 0, opt_time = 0, opt_unroll_time = 0, dbench_time;

    // Repetir o teste 1000 vezes
    for (int i = 0; i < REPEAT; ++i) {
        // Inicializar novamente os polinômios para cada iteração
        initialize_poly(&poly_ref_a);
        initialize_poly(&poly_ref_b);
        poly_opt_a = poly_ref_a;
        poly_opt_b = poly_ref_b;
        poly_opt_unroll_a = poly_ref_a;
        poly_opt_unroll_b = poly_ref_b;

        // Medir o tempo da função de referência
        DBENCH_START(dbench_time);
        poly_add_ref(&poly_ref_c, &poly_ref_a, &poly_ref_b);
        DBENCH_STOP(ref_time, dbench_time);

        // Medir o tempo da função otimizada
        DBENCH_START(dbench_time);
        poly_add_optimized(&poly_opt_c, &poly_opt_a, &poly_opt_b);
        DBENCH_STOP(opt_time, dbench_time);

        // Medir o tempo da função otimizada
        DBENCH_START(dbench_time);
        poly_add_unrolled(&poly_opt_unroll_c, &poly_opt_unroll_a, &poly_opt_unroll_b);
        DBENCH_STOP(opt_unroll_time, dbench_time);

        // Verificar se os resultados são iguais
        if (!compare_polys(&poly_ref_c, &poly_opt_c)) {
            printf("ERRO!!!! - OTIMIZADO X REFERÊNCIA - Os resultados são diferentes na iteração %d\n", i);
            return 1;
        }
        if (!compare_polys(&poly_ref_c, &poly_opt_unroll_c)) {
            printf("ERRO!!!! - UNROLL x REFERÊNCIA - Os resultados são diferentes na iteração %d\n", i);
            return 1;
        }
    }

    // Calcular a média dos tempos
    uint64_t avg_ref_time    = ref_time / REPEAT;
    uint64_t avg_opt_time    = opt_time / REPEAT;
    uint64_t avg_unroll_time = opt_unroll_time / REPEAT;

    // Imprimir os tempos de execução médios
    printf("Tempo médio de execução da função de referência: %llu ciclos\n", avg_ref_time);
    printf("Tempo médio de execução da função otimizada: %llu ciclos\n", avg_opt_time);
    printf("Tempo médio de execução da função com UNROLL: %llu ciclos\n", avg_unroll_time);

    return 0;
}
