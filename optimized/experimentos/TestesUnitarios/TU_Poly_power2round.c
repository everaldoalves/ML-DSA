// NENHUM RESULTADO SATISFATÓRIO FOI OBTIDO COM NEON

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <stdio.h>


#define N 256  // Tamanho do polinômio
#define D 10   // Valor de D para deslocamento
#define MASK ((1 << D) - 1)  // Máscara para limitar a0
#define Q 8380417
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define REPEAT 1000

// Estrutura para armazenar um polinômio
typedef struct {
    int32_t coeffs[N];
} poly;

// CICLOS - Macros para medir ciclos de CPU
#define DBENCH_START_CYCLES(time_var) time_var = cpucycles()
#define DBENCH_STOP_CYCLES(t, time_var) t += cpucycles() - time_var

// TEMPO - Macros para medir o tempo
#define DBENCH_START_TIME(ts) \
    clock_gettime(CLOCK_MONOTONIC, &ts);

#define DBENCH_STOP_TIME(total_time, start_ts) { \
    struct timespec end_ts; \
    clock_gettime(CLOCK_MONOTONIC, &end_ts); \
    total_time += timespec_diff_ns(&start_ts, &end_ts); \
}

// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}
// Função para calcular a diferença de tempo em nanosegundos
uint64_t timespec_diff_ns(struct timespec *start, struct timespec *end) {
    return (end->tv_sec - start->tv_sec) * 1000000000ULL + (end->tv_nsec - start->tv_nsec);
}

// Função para calcular a sobrecarga de medição de TEMPO
uint64_t timing_overhead = 0;
void measure_timing_overhead(void) {
    uint64_t start = cpucycles();
    uint64_t end = cpucycles();
    timing_overhead = end - start;
    printf("Sobrecarga de medição de tempo: %llu ciclos\n", timing_overhead);
}

/*************************************************
* Name:        power2round
*
* Description: For finite field element a, compute a0, a1 such that
*              a mod^+ Q = a1*2^D + a0 with -2^{D-1} < a0 <= 2^{D-1}.
*              Assumes a to be standard representative.
*
* Arguments:   - int32_t a: input element
*              - int32_t *a0: pointer to output element a0
*
* Returns a1.
**************************************************/
int32_t power2round(int32_t *a0, int32_t a)  {
  int32_t a1;

  a1 = (a + (1 << (D-1)) - 1) >> D;
  *a0 = a - (a1 << D);
  return a1;
}
/*************************************************
* Name:        poly_power2round
*
* Description: For all coefficients c of the input polynomial,
*              compute c0, c1 such that c mod Q = c1*2^D + c0
*              with -2^{D-1} < c0 <= 2^{D-1}. Assumes coefficients to be
*              standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
void poly_power2round(poly *a1, poly *a0, const poly *a) {
  unsigned int i;

  for(i = 0; i < N; ++i)
    a1->coeffs[i] = power2round(&a0->coeffs[i], a->coeffs[i]);
  
}


// Função power2round adaptada para operar vetorialmente
static inline void power2round_vector(int32x4_t *a1_vec, int32x4_t *a0_vec, int32x4_t a_vec) {
    // Calcula a1: (a + (1 << (D-1)) - 1) >> D
    int32x4_t offset = vdupq_n_s32((1 << (D-1)) - 1);  // Constante (1 << (D-1)) - 1
    int32x4_t a1 = vaddq_s32(a_vec, offset);           // a + (1 << (D-1)) - 1
    a1 = vshrq_n_s32(a1, D);                           // a1 = (a + offset) >> D

    // Calcula a0: a0 = a - a1 * 2^D
    int32x4_t a0 = vsubq_s32(a_vec, vshlq_n_s32(a1, D));  // a0 = a - (a1 * 2^D)

    // Salva os resultados
    *a1_vec = a1;  // Vetor a1
    *a0_vec = a0;  // Vetor a0
}

void poly_power2round_neon(poly *a1, poly *a0, const poly *a) {
    unsigned int i;
    int32x4_t a_vec, a1_vec, a0_vec;   

    // Processa blocos de 4 coeficientes de cada vez usando NEON
    for(i = 0; i < N; i += 4) {
        // Carrega 4 coeficientes do polinômio de entrada 'a'
        a_vec = vld1q_s32(&a->coeffs[i]);

        // Aplica power2round em 4 coeficientes simultaneamente
        power2round_vector(&a1_vec, &a0_vec, a_vec);

        // Armazena os resultados no polinômio de saída
        vst1q_s32(&a1->coeffs[i], a1_vec);  // Salva a1
        vst1q_s32(&a0->coeffs[i], a0_vec);  // Salva a0
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

// Função para exibir um polinômio
void exibe_poly(poly *p) {
    printf("Polinômio: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", p->coeffs[i]);
    }
    printf("\n\n");
}

// Função para inicializar o polinômio com valores no intervalo [-3, 3]
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % 7 - 3;  // Valores aleatórios em [-3, 3]
    }
}

int main() {
    // Variáveis para medir o tempo
    uint64_t ref_ciclos = 0, opt_ciclos = 0, ref_time = 0, opt_time = 0, dbench_ciclos;
    // Medir a sobrecarga de medição de tempo
    uint64_t timing_overhead = 0;
    struct timespec dbench_time;
    DBENCH_START_TIME(dbench_time);
    DBENCH_STOP_TIME(timing_overhead, dbench_time);

    printf("Testando a função poly_power2round otimizada...\n");

    // Inicializar polinômios
    poly poly_ref_a1, poly_ref_a0, poly_opt_a1, poly_opt_a0;
    poly poly_input;
    initialize_poly(&poly_input);

    // Copiar valores para os polinômios de referência e otimizado
    poly poly_ref = poly_input;
    poly poly_opt = poly_input;

    

    // Repetir o teste 1000 vezes
    for (int i = 0; i < REPEAT; ++i) {
        initialize_poly(&poly_input);
        poly_ref = poly_input;
        poly_opt = poly_input;

        // CICLOS - Testando a função de referência
        DBENCH_START_CYCLES(dbench_ciclos);
        poly_power2round(&poly_ref_a1, &poly_ref_a0, &poly_ref);
        DBENCH_STOP_CYCLES(ref_ciclos, dbench_ciclos);

        // CICLOS - Testando a função otimizada
        DBENCH_START_CYCLES(dbench_ciclos);
        poly_power2round_neon(&poly_opt_a1, &poly_opt_a0, &poly_opt);
        DBENCH_STOP_CYCLES(opt_ciclos, dbench_ciclos);

        // TEMPO - função de referência
        DBENCH_START_TIME(dbench_time);
        poly_power2round(&poly_ref_a1, &poly_ref_a0, &poly_ref);
        DBENCH_STOP_TIME(ref_time, dbench_time);

        // TEMPO - função otimizada
        DBENCH_START_TIME(dbench_time);
        poly_power2round_neon(&poly_opt_a1, &poly_opt_a0, &poly_opt);
        DBENCH_STOP_TIME(opt_time, dbench_time);

        // Comparando os resultados
        if (!compare_polys(&poly_ref_a1, &poly_opt_a1) || !compare_polys(&poly_ref_a0, &poly_opt_a0)) {
            printf("Resultados divergentes na iteração %d\n", i);
            return 1;
        }
    }

    // Média de ciclos
    uint64_t avg_ref_ciclos = ref_ciclos / REPEAT;
    uint64_t avg_opt_ciclos = opt_ciclos / REPEAT;

    // Média de tempo
    uint64_t avg_ref_time = ref_time / REPEAT;
    uint64_t avg_opt_time = opt_time / REPEAT;

    // Exibir resultados
    printf("Ciclos - referência: %llu, otimizado: %llu\n", avg_ref_ciclos, avg_opt_ciclos);
    printf("Tempo (ns) - referência: %llu ns, otimizado: %llu ns\n", avg_ref_time, avg_opt_time);

    return 0;
}