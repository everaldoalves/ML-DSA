
//****************************************************************************************************************************************************************
// Autor: Everaldo Alves
// Data: 19 de Setembro/2024
// Função: poly_ntt
// Descrição: Esta função realiza a transformação de um polinômio para o domínio da NTT. 
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para um speed-up de 1.7x em relação à implementação de referência. 
//****************************************************************************************************************************************************************

#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
//#include <omp.h>
//#include <benchmark/benchmark.h>

#define N 256
#define Q 8380417
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define REPEAT 10000


static const int32_t zetas[N] = {
         0,    25847, -2608894,  -518909,   237124,  -777960,  -876248,   466468,
   1826347,  2353451,  -359251, -2091905,  3119733, -2884855,  3111497,  2680103,
   2725464,  1024112, -1079900,  3585928,  -549488, -1119584,  2619752, -2108549,
  -2118186, -3859737, -1399561, -3277672,  1757237,   -19422,  4010497,   280005,
   2706023,    95776,  3077325,  3530437, -1661693, -3592148, -2537516,  3915439,
  -3861115, -3043716,  3574422, -2867647,  3539968,  -300467,  2348700,  -539299,
  -1699267, -1643818,  3505694, -3821735,  3507263, -2140649, -1600420,  3699596,
    811944,   531354,   954230,  3881043,  3900724, -2556880,  2071892, -2797779,
  -3930395, -1528703, -3677745, -3041255, -1452451,  3475950,  2176455, -1585221,
  -1257611,  1939314, -4083598, -1000202, -3190144, -3157330, -3632928,   126922,
   3412210,  -983419,  2147896,  2715295, -2967645, -3693493,  -411027, -2477047,
   -671102, -1228525,   -22981, -1308169,  -381987,  1349076,  1852771, -1430430,
  -3343383,   264944,   508951,  3097992,    44288, -1100098,   904516,  3958618,
  -3724342,    -8578,  1653064, -3249728,  2389356,  -210977,   759969, -1316856,
    189548, -3553272,  3159746, -1851402, -2409325,  -177440,  1315589,  1341330,
   1285669, -1584928,  -812732, -1439742, -3019102, -3881060, -3628969,  3839961,
   2091667,  3407706,  2316500,  3817976, -3342478,  2244091, -2446433, -3562462,
    266997,  2434439, -1235728,  3513181, -3520352, -3759364, -1197226, -3193378,
    900702,  1859098,   909542,   819034,   495491, -1613174,   -43260,  -522500,
   -655327, -3122442,  2031748,  3207046, -3556995,  -525098,  -768622, -3595838,
    342297,   286988, -2437823,  4108315,  3437287, -3342277,  1735879,   203044,
   2842341,  2691481, -2590150,  1265009,  4055324,  1247620,  2486353,  1595974,
  -3767016,  1250494,  2635921, -3548272, -2994039,  1869119,  1903435, -1050970,
  -1333058,  1237275, -3318210, -1430225,  -451100,  1312455,  3306115, -1962642,
  -1279661,  1917081, -2546312, -1374803,  1500165,   777191,  2235880,  3406031,
   -542412, -2831860, -1671176, -1846953, -2584293, -3724270,   594136, -3776993,
  -2013608,  2432395,  2454455,  -164721,  1957272,  3369112,   185531, -1207385,
  -3183426,   162844,  1616392,  3014001,   810149,  1652634, -3694233, -1799107,
  -3038916,  3523897,  3866901,   269760,  2213111,  -975884,  1717735,   472078,
   -426683,  1723600, -1803090,  1910376, -1667432, -1104333,  -260646, -3833893,
  -2939036, -2235985,  -420899, -2286327,   183443,  -976891,  1612842, -3545687,
   -554416,  3919660,   -48306, -1362209,  3937738,  1400424,  -846154,  1976782
};


typedef struct {
    int32_t coeffs[N] __attribute__((aligned(16)));
} poly;


// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}

int32_t montgomery_reduce(int64_t a) {
    int32_t t;    

    t = (int64_t)(int32_t)a * QINV;
    t = (a - (int64_t)t * Q) >> 32;
    
    return t;
}

// NTT original 
void ntt(int32_t a[N]) {
  unsigned int len, start, j, k;
  int32_t zeta, t;

  k = 0;
  for(len = 128; len > 0; len >>= 1) {
    for(start = 0; start < N; start = j + len) {
      zeta = zetas[++k];
      for(j = start; j < start + len; ++j) {
        t = montgomery_reduce((int64_t)zeta * a[j + len]);
        a[j + len] = a[j] - t;
        a[j] = a[j] + t;
      }
    }
  }
}

int32x4_t montgomery_reduce_neon_4(int64x2x2_t a) {
    int32x2_t a_low1 = vmovn_s64(a.val[0]);
    int32x2_t a_low2 = vmovn_s64(a.val[1]);

    int32x2_t qinv = vdup_n_s32(QINV);

    int32x2_t t1 = vmul_s32(a_low1, qinv);
    int32x2_t t2 = vmul_s32(a_low2, qinv);

    int64x2_t t_full1 = vmull_s32(t1, vdup_n_s32(Q));
    int64x2_t t_full2 = vmull_s32(t2, vdup_n_s32(Q));

    int64x2_t res1 = vsubq_s64(a.val[0], t_full1);
    int64x2_t res2 = vsubq_s64(a.val[1], t_full2);

    int32x2_t reduced1 = vshrn_n_s64(res1, 32);
    int32x2_t reduced2 = vshrn_n_s64(res2, 32);

    int32x4_t result = vcombine_s32(reduced1, reduced2);
    return result;
}

// NTT Otimizada com Paralelismo
void ntt_neon_4(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    k = 0;

    // Estágios maiores, onde podemos usar NEON para 8 elementos
    for (len = 128; len >= 8 ; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            int32x4x2_t zeta_vec;
            zeta_vec.val[0] = vdupq_n_s32(zetas[++k]);
            zeta_vec.val[1] = vdupq_n_s32(zetas[++k]);

            for (j = start; j < start + len; j += 8) {
                int32x4x2_t a_vec = vld1q_s32_x2(&a[j]);         // Carregar 8 coeficientes
                int32x4x2_t a_len_vec = vld1q_s32_x2(&a[j + len]); // Carregar 8 coeficientes com offset len

                // Multiplicação de Montgomery (usando a abordagem de Sanal)
                int32x4x2_t t_vec;
                t_vec.val[0] = vmulq_s32(a_len_vec.val[0], zeta_vec.val[0]);
                t_vec.val[1] = vmulq_s32(a_len_vec.val[1], zeta_vec.val[1]);

                // Subtração e Adição
                vst1q_s32(&a[j + len], vsubq_s32(a_vec.val[0], t_vec.val[0]));
                vst1q_s32(&a[j], vaddq_s32(a_vec.val[0], t_vec.val[0]));

                vst1q_s32(&a[j + len + 4], vsubq_s32(a_vec.val[1], t_vec.val[1]));
                vst1q_s32(&a[j + 4], vaddq_s32(a_vec.val[1], t_vec.val[1]));
            }
        }
    }

    // Estágios menores, onde NEON não faz sentido
    for (len = 4; len > 0; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = zetas[++k];

            for (j = start; j < start + len; ++j) {
                int32_t t = montgomery_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
    }
}

void ntt_radix4(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32x4_t zeta_vec, t_vec1, t_vec2, a_vec1, a_vec2, a_vec3, a_vec4;
    int32_t zeta;

    k = 0;
    for(len = 64; len > 0; len >>= 2) { // Radix-4, reduzindo o len por 4 a cada iteração
        for(start = 0; start < N; start += 4 * len) {
            for(j = start; j < start + len; j += 4) {
                // Carregar quatro coeficientes de cada vez
                a_vec1 = vld1q_s32(&a[j]);
                a_vec2 = vld1q_s32(&a[j + len]);
                a_vec3 = vld1q_s32(&a[j + 2 * len]);
                a_vec4 = vld1q_s32(&a[j + 3 * len]);

                // Carregar os valores de zeta para cada butterfly
                int32_t zeta1 = zetas[++k];
                int32_t zeta2 = zetas[++k];
                int32_t zeta3 = zetas[++k];
                int32_t zeta4 = zetas[++k];

                // Aplicar os valores de zeta
                int32x4_t zeta_vec1 = vdupq_n_s32(zeta1);
                int32x4_t zeta_vec2 = vdupq_n_s32(zeta2);
                int32x4_t zeta_vec3 = vdupq_n_s32(zeta3);
                int32x4_t zeta_vec4 = vdupq_n_s32(zeta4);

                // Multiplicação de Montgomery
                int64x2x2_t t_prod1 = {
                    vmull_s32(vget_low_s32(zeta_vec1), vget_low_s32(a_vec2)),
                    vmull_s32(vget_high_s32(zeta_vec1), vget_high_s32(a_vec2))
                };
                t_vec1 = montgomery_reduce_neon_4(t_prod1);

                int64x2x2_t t_prod2 = {
                    vmull_s32(vget_low_s32(zeta_vec2), vget_low_s32(a_vec4)),
                    vmull_s32(vget_high_s32(zeta_vec2), vget_high_s32(a_vec4))
                };
                t_vec2 = montgomery_reduce_neon_4(t_prod2);

                // Atualizar os coeficientes
                a_vec2 = vsubq_s32(a_vec1, t_vec1);
                a_vec1 = vaddq_s32(a_vec1, t_vec1);

                a_vec4 = vsubq_s32(a_vec3, t_vec2);
                a_vec3 = vaddq_s32(a_vec3, t_vec2);

                // Armazenar os resultados de volta no vetor
                vst1q_s32(&a[j], a_vec1);
                vst1q_s32(&a[j + len], a_vec2);
                vst1q_s32(&a[j + 2 * len], a_vec3);
                vst1q_s32(&a[j + 3 * len], a_vec4);
            }
        }
    }
}
// Esta implementação está incorreta.
void ntt_memoria_limpa(int32_t a[N]) {
    int32x4_t zeta_vec1, zeta_vec2;
    int32x4_t a1_vec, a2_vec, b1_vec, b2_vec;
    int64x2_t t_vec1_low, t_vec1_high, t_vec2_low, t_vec2_high;
    int64x2x2_t t_vec1, t_vec2;

    int len, start, j, k = 0;

    // Estágios maiores, desenrolamento de loop para máxima eficiência
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            // Carregar duas zetas por vez em vetores NEON
            zeta_vec1 = vdupq_n_s32(zetas[++k]);
            zeta_vec2 = vdupq_n_s32(zetas[++k]);

            for (j = start; j < start + len; j += 4) {
                // Carregar 4 coeficientes de 'a' e 'b' em vetores NEON
                a1_vec = vld1q_s32(&a[j]);            // a[j]
                a2_vec = vld1q_s32(&a[j + len]);      // a[j + len]
                b1_vec = vld1q_s32(&a[j + 2 * len]);  // a[j + 2*len]
                b2_vec = vld1q_s32(&a[j + 3 * len]);  // a[j + 3*len]

                // Multiplicação por zeta e redução de Montgomery com vetores NEON
                t_vec1_low = vmull_s32(vget_low_s32(a2_vec), vget_low_s32(zeta_vec1));
                t_vec1_high = vmull_s32(vget_high_s32(a2_vec), vget_high_s32(zeta_vec1));
                t_vec2_low = vmull_s32(vget_low_s32(b2_vec), vget_low_s32(zeta_vec2));
                t_vec2_high = vmull_s32(vget_high_s32(b2_vec), vget_high_s32(zeta_vec2));

                // Organizar os resultados em t_vec1 e t_vec2 para Montgomery Reduction
                t_vec1.val[0] = t_vec1_low;
                t_vec1.val[1] = t_vec1_high;
                t_vec2.val[0] = t_vec2_low;
                t_vec2.val[1] = t_vec2_high;

                // Chamada para função montgomery_reduce_neon_4 para processar 4 coeficientes
                a2_vec = montgomery_reduce_neon_4(t_vec1);
                b2_vec = montgomery_reduce_neon_4(t_vec2);

                // Somar e subtrair resultados nos vetores NEON
                a1_vec = vaddq_s32(a1_vec, a2_vec);   // a[j] = a[j] + t
                a2_vec = vsubq_s32(a1_vec, a2_vec);   // a[j + len] = a[j] - t

                b1_vec = vaddq_s32(b1_vec, b2_vec);   // a[j + 2*len] = a[j + 2*len] + t
                b2_vec = vsubq_s32(b1_vec, b2_vec);   // a[j + 3*len] = a[j + 3*len] - t

                // Armazenar os resultados processados de volta na memória
                vst1q_s32(&a[j], a1_vec);
                vst1q_s32(&a[j + len], a2_vec);
                vst1q_s32(&a[j + 2 * len], b1_vec);
                vst1q_s32(&a[j + 3 * len], b2_vec);
            }
        }
    }
}



/*
void poly_ntt_neon_paralela(poly *a) {
    #pragma omp parallel for    
    
        ntt_neon_4(a->coeffs);    
    
}
*/
void poly_ntt_radix4(poly *a) {    
    ntt_radix4(a->coeffs);  // Chamando a versão otimizada uma única vez para processar todos os 256 coeficientes
}

void poly_ntt_neon(poly *a) {    
    ntt_memoria_limpa(a->coeffs);  // Chamando a versão otimizada uma única vez para processar todos os 256 coeficientes
}



void poly_ntt_referencia(poly *a) {
    ntt(a->coeffs);  // Chamando NTT uma única vez para processar os 256 coeficientes
}

void print_poly(poly *a) {
    for (int i = 0; i < N; i++) {
        printf("%d ", a->coeffs[i]);
    }
    printf("\n");
}
void compara_poly(poly *a, poly *b) {
    for (int i = 0; i < N; i++) {
        if (a->coeffs[i] != b->coeffs[i]) {
            printf("Erro: a[%d] = %d, b[%d] = %d\n", i, a->coeffs[i], i, b->coeffs[i]);
            return;
        }
    }
    printf("Os polinômios são iguais\n");
}

// Função para medir o tempo
double get_time(clock_t start, clock_t end) {
    return (double)(end - start) / CLOCKS_PER_SEC;
}

int main() {
    poly a,b,c;
    srand(time(NULL));
    double tempo1,tempo2,tempo3 = 0;
    clock_t start, end;

    // Inicializar os coeficientes aleatoriamente para o teste
    for (int i = 0; i < N; i++) {
        a.coeffs[i] = rand() % Q;
    }

    // Copiar os coeficientes de a para b/c
    for (int i = 0; i < N; i++) {
        b.coeffs[i] = a.coeffs[i];
        c.coeffs[i] = a.coeffs[i];
    }

    // Aplicar a NTT para ambos os polinômios e comparar os resultados
    printf("Verificando se os polinômios de entrada a,b,c são iguais....\n");
    compara_poly(&a, &b);
    compara_poly(&a, &c);
    
    // Aplicando a NTT aos polinômios de entrada
    poly_ntt_referencia(&a);
    poly_ntt_neon(&b);
    poly_ntt_radix4(&c);
    printf("Aplicando a NTT....\n");
    printf("Comparando os os polinômios a e b...\n");
    compara_poly(&a, &b);
    printf("\nComparando os os polinômios a e c...\n");
    compara_poly(&a, &c);

    for (int i = 0; i < REPEAT; i++) {
        // Teste para a versão single-thread
        start = clock();
        poly_ntt_referencia(&a);
        end = clock();
        tempo1 += get_time(start, end);       
    }
       
    for (int i = 0; i < REPEAT; i++) {
        // Teste para a versão de referência
        start = clock();
        poly_ntt_neon(&b);
        end = clock();
        tempo2 += get_time(start, end);
        
    }

     for (int i = 0; i < REPEAT; i++) {
        // Teste para a versão de referência
        start = clock();
        poly_ntt_radix4(&b);
        end = clock();
        tempo3 += get_time(start, end);
        
    }

    
     printf("Tempo de referência: %f segundos\n", tempo1);
     printf("Tempo da função OTIMIZADA: %f segundos\n", tempo2);
     printf("Tempo da função RADIX-4: %f segundos\n", tempo3);


     if ((tempo1 < tempo2) && (tempo1 < tempo3)) {
         printf("A versão de referência é mais rápida\n");
     } else {
        if(tempo2 < tempo3) {
            printf("A versão OTIMIZADA é mais rápida\n");
        } else {
         printf("A versão RADIX-4 é mais rápida\n");
        }
     }

     compara_poly(&a, &b);
     compara_poly(&a, &c);


     // Medir os ciclos da função de referência
        uint64_t ref_time = 0, opt_time = 0, rd4_time = 0;
        uint64_t start_cycles, end_cycles;

        // Executar e medir ciclos da função de referência
        for (int i = 0; i < REPEAT; i++) {
            start_cycles = cpucycles();
            poly_ntt_referencia(&a);
            end_cycles = cpucycles();
            ref_time += end_cycles - start_cycles;
        }

        // Executar e medir ciclos da função otimizada
        for (int i = 0; i < REPEAT; i++) {
            start_cycles = cpucycles();
            poly_ntt_neon(&b);
            end_cycles = cpucycles();
            opt_time += end_cycles - start_cycles;
        }

        // Executar e medir ciclos da função otimizada
        for (int i = 0; i < REPEAT; i++) {
            start_cycles = cpucycles();
            poly_ntt_radix4(&c);
            end_cycles = cpucycles();
            rd4_time += end_cycles - start_cycles;
        }


    // Calcular o speed-up
    double avg_ref_time  = (double) ref_time / REPEAT;
    double avg_opt_time  = (double) opt_time / REPEAT;
    double avg_rd4_time  = (double) rd4_time / REPEAT;
    
    uint64_t cycles_saved1 = avg_ref_time - avg_opt_time;
    double speedup1 = avg_ref_time / avg_opt_time;

    uint64_t cycles_saved2 = avg_ref_time - avg_rd4_time;
    double speedup2 = avg_ref_time / avg_rd4_time;

    // Mostrar o speed-up
    printf("\n\nCiclos médios da função de referência: %.2f\n", avg_ref_time);
    printf("\nCiclos médios da função OTIMIZADA %.2f\n", avg_opt_time);
    printf("Ciclos economizados: %llu\n", cycles_saved1);
    printf("Speed-up: %.2f X\n", speedup1);
    printf("\n\n");
    printf("Ciclos médios da função RADIX-4 %.2f\n", avg_rd4_time);
    printf("Ciclos economizados: %llu\n", cycles_saved2);
    printf("Speed-up: %.2f X\n", speedup2);

    return 0;
}