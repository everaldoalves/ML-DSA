#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>

#define N 256
#define Q 8380417
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
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
    // Repita para obter uma medição média mais precisa
    for (int i = 0; i < 100; i++) {
        start = cpucycles();
        end = cpucycles();
        timing_overhead += (end - start);
    }
    timing_overhead /= 101; // Média
}

// Função de impressão para int32x2_t
void print_int32x2_t(int32x2_t v, const char* name) {
    int32_t vals[2];
    vst1_s32(vals, v);
    printf("%s = [%d, %d]\n", name, vals[0], vals[1]);
}
// Função de impressão para int32x4_t
void print_int32x4_t(int32x4_t v, const char* name) {
    int32_t vals[4];
    vst1q_s32(vals, v);
    printf("%s : %d %d %d %d\n", name, vals[0], vals[1], vals[2], vals[3]);
}

void print_int32_t(int32_t v[4], const char* name) {
    printf("%s : ", name );
    for (int i = 0; i < 4; i++) {
        printf("%d ", v[i]);
    }    
    printf("\n");
}

// Função montgomery_reduce fornecida no código original
int32_t montgomery_reduce(int64_t a) {
    int32_t t;
    t = (int64_t)(int32_t)a * QINV; //     
    t = (a - (int64_t)t * Q) >> 32;
    return t;
}

// Função montgomery_reduce otimizada com NEON
int32_t montgomery_reduce_neon(int64_t a) {
    int32x2_t a_low = vmov_n_s32((int32_t)a);          // a em 32 bits
    int32x2_t qinv_v = vdup_n_s32(QINV);               // vetor com QINV
    int32x2_t t = vmul_s32(a_low, qinv_v);             // t = (a * QINV)
    int64_t temp = (int64_t)vget_lane_s32(t, 0) * Q;   // calcular t * Q
    int32_t result = (a - temp) >> 32;                 // (a - t * Q) >> 32
    return result;
}

// Redução de Montgomery otimizada com NEON para 2 coeficientes
int32x2_t montgomery_reduce_neon3(int64x2_t a) {
    int32x2_t a_low = vmovn_s64(a);  // Extrai os 32 bits inferiores
    int32x2_t qinv = vdup_n_s32(QINV);
    
    // t = a * QINV (parte baixa)
    int32x2_t t = vmul_s32(a_low, qinv);

    // t * Q (resultando em 64 bits)
    int64x2_t t_full = vmull_s32(t, vdup_n_s32(Q));

    // a - t * Q e realiza o shift de 32 bits
    int64x2_t res = vsubq_s64(a, t_full);
    
    // Shift à direita de 32 bits para fazer a redução e retorna dois valores de 32 bits
    return vshrn_n_s64(res, 32);
}

// Redução de Montgomery otimizada com NEON para 4 coeficientes
int32x4_t montgomery_reduce_neon4(int64x2x2_t a) {
    // Extrair os 32 bits inferiores de cada bloco
    int32x2_t a_low1 = vmovn_s64(a.val[0]);
    int32x2_t a_low2 = vmovn_s64(a.val[1]);

    int32x2_t qinv = vdup_n_s32(QINV);

    // t = a * QINV (parte baixa)
    int32x2_t t1 = vmul_s32(a_low1, qinv);
    int32x2_t t2 = vmul_s32(a_low2, qinv);

    // t * Q (resultando em 64 bits)
    int64x2_t t_full1 = vmull_s32(t1, vdup_n_s32(Q));
    int64x2_t t_full2 = vmull_s32(t2, vdup_n_s32(Q));

    // a - t * Q e realizar o shift de 32 bits
    int64x2_t res1 = vsubq_s64(a.val[0], t_full1);
    int64x2_t res2 = vsubq_s64(a.val[1], t_full2);

    // Shift à direita de 32 bits para fazer a redução e retorna dois blocos de 4 valores de 32 bits
    int32x2_t reduced1 = vshrn_n_s64(res1, 32);
    int32x2_t reduced2 = vshrn_n_s64(res2, 32);

    // Combinar os resultados em um vetor de 4 valores
    int32x4_t result;
    result = vcombine_s32(reduced1, reduced2);


    return result;
}

// Inicializa polinômio com valores aleatórios
void initialize_poly(poly* p) {
    for (int i = 0; i < N; i++) {
        p->coeffs[i] = rand() % Q*Q;
    }
}

// Função para garantir que o compilador não elimine o resultado
void force_use_int32(int32_t val) {
    volatile int32_t sum = val;
}

void force_use_int32x2(int32x2_t val) {
    volatile int32_t sum = vget_lane_s32(val, 0) + vget_lane_s32(val, 1);
}

void force_use_int32x4(int32x4_t val) {
    volatile int32_t sum = vgetq_lane_s32(val, 0) + vgetq_lane_s32(val, 1) + vgetq_lane_s32(val, 2) + vgetq_lane_s32(val, 3);
}

int main() {
    // Medir a sobrecarga de medição de tempo
    measure_timing_overhead();

    int64_t poly_ref[4];
    int64x2x2_t poly_opt;
    int32_t poly_ref32[4];
    int32x4_t poly_opt32;
    

    poly_ref[0] = -19646401311217;
    poly_ref[1] = 20298163801216;
    poly_ref[2] = -886464013112135;
    poly_ref[3] = 90298163801255;

    poly_opt.val[0] = vcombine_s64(vcreate_s64(poly_ref[0]), vcreate_s64(poly_ref[1]));
    poly_opt.val[1] = vcombine_s64(vcreate_s64(poly_ref[2]), vcreate_s64(poly_ref[3]));
   

    // Variáveis para medir o tempo
    uint64_t ref_time = 0, opt_time = 0, dbench_time;

    // Repetir o teste várias vezes
    for (int i = 0; i < REPEAT; ++i) {     

        printf("Iteração %d\n", i);

            // Medir o tempo da função de referência
        DBENCH_START(dbench_time);
        for (int j = 0; j < 4; j++) {
            poly_ref32[j] = montgomery_reduce(poly_ref[j]);
            force_use_int32(poly_ref32[j]);
        }
        DBENCH_STOP(ref_time, dbench_time);

        // Medir o tempo da função otimizada
        DBENCH_START(dbench_time);
        for (int j = 0; j < 4; j++) {
            poly_opt32 = montgomery_reduce_neon4(poly_opt);
            force_use_int32x4(poly_opt32);
        }
        DBENCH_STOP(opt_time, dbench_time);

        // Comparar os resultados
        for (int j = 0; j < 4; j++) {
            int32_t opt_val;
            if (j == 0) {
                opt_val = vgetq_lane_s32(poly_opt32, 0); // corrigir vget_lane_s32 para vgetq_lane_s32
            } else if (j == 1) {
                opt_val = vgetq_lane_s32(poly_opt32, 1);
            }
            else if (j == 2) {
                opt_val = vgetq_lane_s32(poly_opt32, 2);
            }
            else if (j == 3) {
                opt_val = vgetq_lane_s32(poly_opt32, 3);
            } 
            if (poly_ref32[j] != opt_val) {
                printf("Erro na otimização na posição %d: ref=%lld, opt=%d\n", j, poly_ref32[j], opt_val);
                return 1;
            }
        }
    }
    printf("Os resultados são iguais\n");

    print_int32_t(poly_ref32, "poly_ref32");
    print_int32x4_t(poly_opt32, "poly_opt32");
    

    // Exibir resultados de tempo
    printf("\nTempo de execução da função de referência: %llu ciclos\n", ref_time / REPEAT);
    printf("Tempo de execução da função otimizada: %llu ciclos\n", opt_time / REPEAT);

    // Calcular speed-up
    double speedup = (double)ref_time / opt_time;
    printf("\nSpeed-up: %.2fx\n", speedup);

    return 0;
}
