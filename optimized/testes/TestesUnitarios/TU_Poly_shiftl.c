//*******************************************************************************************************
// Houve um ganho de desempenho ao usar NEON para otimizar esta função em aproximadamente 100%
//*******************************************************************************************************

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define N 256
#define Q 8380417
#define D 5  // Valor de D para deslocamento
#define REPEAT 10000

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

// Função de referência: poly_shiftl
void poly_shiftl_ref(poly *a) {
    unsigned int i;
    for (i = 0; i < N; ++i) {
        a->coeffs[i] <<= D;  // Desloca os coeficientes à esquerda em D bits
    }
}

// Função otimizada usando NEON: poly_shiftl
void poly_shiftl_optimized(poly *a) {
    unsigned int i;
    for (i = 0; i < N; i += 4) {
        int32x4_t a_vec = vld1q_s32(&a->coeffs[i]);  // Carrega 4 coeficientes de a
        a_vec = vshlq_n_s32(a_vec, D);               // Desloca à esquerda em D bits
        vst1q_s32(&a->coeffs[i], a_vec);             // Armazena o resultado
    }
}

// Função para inicializar o polinômio com valores aleatórios
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % (2 * Q) - Q;  // Valores aleatórios em [-Q, Q]
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

int main() {
    // Medir a sobrecarga de medição de tempo
    measure_timing_overhead();

    // Inicializar dois polinômios
    poly poly_ref, poly_opt;
    initialize_poly(&poly_ref);

    poly_opt = poly_ref;  // Copiar valores para o polinômio otimizado

    // Variáveis para medir o tempo
    uint64_t ref_time = 0, opt_time = 0, dbench_time;

    // Repetir o teste 1000 vezes
    for (int i = 0; i < REPEAT; ++i) {
        // Inicializar novamente os polinômios para cada iteração
        initialize_poly(&poly_ref);
        poly_opt = poly_ref;

        // Medir o tempo da função de referência
        DBENCH_START(dbench_time);
        poly_shiftl_ref(&poly_ref);
        DBENCH_STOP(ref_time, dbench_time);
        force_use(&poly_ref);

        // Medir o tempo da função otimizada
        DBENCH_START(dbench_time);
        poly_shiftl_optimized(&poly_opt);
        DBENCH_STOP(opt_time, dbench_time);
        force_use(&poly_opt);

        // Verificar se os resultados são iguais
        if (!compare_polys(&poly_ref, &poly_opt)) {
            printf("Os resultados são diferentes na iteração %d\n", i);
            return 1;
        }
    }

    // Calcular a média dos tempos
    uint64_t avg_ref_time = ref_time / REPEAT;
    uint64_t avg_opt_time = opt_time / REPEAT;

    // Imprimir os tempos de execução médios
    printf("Tempo médio de execução da função de referência: %llu ciclos\n", avg_ref_time);
    printf("Tempo médio de execução da função otimizada: %llu ciclos\n", avg_opt_time);

    return 0;
}
