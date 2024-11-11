// Contém uma abordagem de profile que utiliza o Google Benchmark
// para comparar o desempenho de duas funções que implementam POLY2ROUND.
// **** >>> "NENHUMA" aceleração foi obtida até agora com a função otimizada.
// Para compilar e testar utilize:  g++ GoogleBenchmarkPoly.cpp -o benchmark_test -I/opt/homebrew/include -L/opt/homebrew/lib -lbenchmark -pthread -march=armv8-a+simd -O3 -ftree-vectorize -std=c++11
// ./benchmark_test  

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <stdio.h>
#include <benchmark/benchmark.h>


#define N 256  // Tamanho do polinômio
#define D 10   // Valor de D para deslocamento
#define MASK ((1 << D) - 1)  // Máscara para limitar a0
#define Q 8380417

// Estrutura para armazenar um polinômio
typedef struct {
    int32_t coeffs[N];
} poly;

void* aligned_malloc(size_t alignment, size_t size) {
    void *ptr = NULL;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return NULL;
    }
    return ptr;
}

// Função de referência
int32_t power2round(int32_t *a0, int32_t a)  {
  int32_t a1;

  a1 = (a + (1 << (D-1)) - 1) >> D;
  *a0 = a - (a1 << D);
  return a1;
}

// Função de referência
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

// Função que lê o contador de ciclos de CPU no ARM
inline uint64_t read_cycle_count() {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}

// Função para inicializar polinômios
void initialize_poly(poly *p) {
    // Aloca memória alinhada para a estrutura inteira
    poly *aligned_p = (poly *)aligned_malloc(16, sizeof(poly));
    if (aligned_p == NULL) {
        // Tratamento de erro
        perror("aligned_malloc falhou");
        exit(EXIT_FAILURE);
    }

    // Copia a estrutura alinhada para o ponteiro original
    *p = *aligned_p;
    free(aligned_p);  // Libera a memória temporária

    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % 8380417;  // Usar um valor aleatório no intervalo [-Q, Q]
    }
}

// Benchmark para a função de referência
static void BM_poly_power2round(benchmark::State& state) {
    poly poly_ref_a1, poly_ref_a0, poly_ref;
    initialize_poly(&poly_ref);  // Inicializa o polinômio de entrada

    for (auto _ : state) {
        uint64_t start = read_cycle_count();  // Inicia a contagem de ciclos

        poly_power2round(&poly_ref_a1, &poly_ref_a0, &poly_ref);

        uint64_t end = read_cycle_count();  // Finaliza a contagem de ciclos

        state.counters["Cycles"] = end - start;  // Calcula os ciclos
    }

    // Processamento de bytes
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(sizeof(poly) * 3));  // 3 polinômios usados
}

// Benchmark para a função otimizada
static void BM_poly_power2round_neon(benchmark::State& state) {
    poly poly_opt_a1, poly_opt_a0, poly_opt;
    initialize_poly(&poly_opt);  // Inicializa o polinômio de entrada

    for (auto _ : state) {
        uint64_t start = read_cycle_count();  // Inicia a contagem de ciclos

        poly_power2round_neon(&poly_opt_a1, &poly_opt_a0, &poly_opt);

        uint64_t end = read_cycle_count();  // Finaliza a contagem de ciclos

        state.counters["Cycles"] = end - start;  // Calcula os ciclos
    }

    // Processamento de bytes
    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(sizeof(poly) * 3));  // 3 polinômios usados
}

// Registra os benchmarks
BENCHMARK(BM_poly_power2round)->Unit(benchmark::kNanosecond)->UseRealTime();
BENCHMARK(BM_poly_power2round_neon)->Unit(benchmark::kNanosecond)->UseRealTime();

// Função principal do benchmark
BENCHMARK_MAIN();