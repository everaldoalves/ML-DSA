
//****************************************************************************************************************************************************************
// Autor: Everaldo Alves
// Data: 14 de Outubro/2024
// Função: ntt e inversa com radix-4 comparada a radix-2 USANDO O GOOGLE BENCHMARK
// Descrição: Esta função realiza a transformação de um polinômio para o domínio da NTT. 
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para um speed-up de 1.7x em relação à implementação de referência. 
// Comando para COMPILAR: g++ -O2 -std=c++11 -I /opt/homebrew/include TU_Poly_ntt.cpp -L /opt/homebrew/lib -lbenchmark -lpthread -o ntt_google_benchmark
//****************************************************************************************************************************************************************

#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <benchmark/benchmark.h>

#include <mach/mach.h>
#include <mach/thread_policy.h>
#include <pthread.h>

void set_cpu_affinity(int cpu) {
    thread_port_t thread = pthread_mach_thread_np(pthread_self());
    thread_affinity_policy_data_t policy = { cpu };
    thread_policy_set(thread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1);
}

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


int32x4x2_t montgomery_reduce_neon_8(int64x2x2_t a1, int64x2x2_t a2) {
    // Extrair 32 bits mais baixos dos 64 bits resultantes
    int32x2_t a_low1_1 = vmovn_s64(a1.val[0]);
    int32x2_t a_low1_2 = vmovn_s64(a1.val[1]);
    int32x2_t a_low2_1 = vmovn_s64(a2.val[0]);
    int32x2_t a_low2_2 = vmovn_s64(a2.val[1]);

    // Multiplicar com QINV para obter o valor 't'
    int32x2_t qinv = vdup_n_s32(QINV);
    int32x2_t t1_1 = vmul_s32(a_low1_1, qinv);
    int32x2_t t1_2 = vmul_s32(a_low1_2, qinv);
    int32x2_t t2_1 = vmul_s32(a_low2_1, qinv);
    int32x2_t t2_2 = vmul_s32(a_low2_2, qinv);

    // Multiplicar por Q
    int64x2_t t_full1_1 = vmull_s32(t1_1, vdup_n_s32(Q));
    int64x2_t t_full1_2 = vmull_s32(t1_2, vdup_n_s32(Q));
    int64x2_t t_full2_1 = vmull_s32(t2_1, vdup_n_s32(Q));
    int64x2_t t_full2_2 = vmull_s32(t2_2, vdup_n_s32(Q));

    // Subtrair
    int64x2_t res1_1 = vsubq_s64(a1.val[0], t_full1_1);
    int64x2_t res1_2 = vsubq_s64(a1.val[1], t_full1_2);
    int64x2_t res2_1 = vsubq_s64(a2.val[0], t_full2_1);
    int64x2_t res2_2 = vsubq_s64(a2.val[1], t_full2_2);

    // Shift right 32 bits
    int32x2_t reduced1_1 = vshrn_n_s64(res1_1, 32);
    int32x2_t reduced1_2 = vshrn_n_s64(res1_2, 32);
    int32x2_t reduced2_1 = vshrn_n_s64(res2_1, 32);
    int32x2_t reduced2_2 = vshrn_n_s64(res2_2, 32);

    // Combinar resultados
    int32x4_t result1 = vcombine_s32(reduced1_1, reduced1_2);
    int32x4_t result2 = vcombine_s32(reduced2_1, reduced2_2);

    return (int32x4x2_t) { result1, result2 };
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
                // Garantir que não ultrapasse o tamanho do vetor zetas
                if (k + 4 > N) {
                    break; // Parar o laço se não houver valores de zetas suficientes
                }

                // Carregar quatro coeficientes de cada vez
                a_vec1 = vld1q_s32(&a[j]);
                a_vec2 = vld1q_s32(&a[j + len]);
                a_vec3 = vld1q_s32(&a[j + 2 * len]);
                a_vec4 = vld1q_s32(&a[j + 3 * len]);

                // Carregar os valores de zeta para cada butterfly
                int32_t zeta1 = zetas[k++];
                int32_t zeta2 = zetas[k++];
                int32_t zeta3 = zetas[k++];
                int32_t zeta4 = zetas[k++];

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



void ntt_everaldo1(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    k = 0;

    // Estágios maiores, otimizados usando as técnicas de Yongbeom
    for (len = 128; len >= 8; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = zetas[++k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Duplicar o valor de zeta para vetor NEON

            for (j = start; j < start + len; j += 8) {
                // Carregar 8 coeficientes (4 a cada vez) usando NEON
                int32x4_t a_vec1 = vld1q_s32(&a[j]);
                int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);
                int32x4_t a_len1 = vld1q_s32(&a[j + len]);
                int32x4_t a_len2 = vld1q_s32(&a[j + len + 4]);

                // Aplicação da borboleta (Butterfly 1) - técnica de Yongbeom
                int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_len1), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len1), vget_high_s32(zeta_vec)) };
                int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_len2), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len2), vget_high_s32(zeta_vec)) };

                int32x4_t t_vec1 = montgomery_reduce_neon_4(prod1);
                int32x4_t t_vec2 = montgomery_reduce_neon_4(prod2);

                // Adição e subtração com redução modular para Borboleta 2
                vst1q_s32(&a[j + len], vsubq_s32(a_vec1, t_vec1));
                vst1q_s32(&a[j], vaddq_s32(a_vec1, t_vec1));

                vst1q_s32(&a[j + len + 4], vsubq_s32(a_vec2, t_vec2));
                vst1q_s32(&a[j + 4], vaddq_s32(a_vec2, t_vec2));

                // Máscara para garantir o ajuste dos coeficientes (Butterfly 2)
                int64_t w9 = (int64_t)a[j + len] * zeta;
                int64_t t = montgomery_reduce(w9);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
    }

    // Estágios menores, onde não vale a pena usar NEON
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

void ntt_everaldo2(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    k = 0;

    // Estágios maiores, onde podemos usar NEON para 8 elementos
    for (len = 128; len >= 8 ; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = zetas[++k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Duplicar o valor de zeta para vetor NEON

            for (j = start; j < start + len; j += 8) {
                int32x4_t a_vec1 = vld1q_s32(&a[j]);         // Carregar 4 coeficientes
                int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);     // Carregar próximos 4 coeficientes
                int32x4_t a_len1 = vld1q_s32(&a[j + len]);   // Carregar 4 coeficientes com offset len
                int32x4_t a_len2 = vld1q_s32(&a[j + len + 4]);

                // Multiplicação de Montgomery
                int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_len1), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len1), vget_high_s32(zeta_vec)) };
                int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_len2), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len2), vget_high_s32(zeta_vec)) };

                int32x4_t t_vec1 = montgomery_reduce_neon_4(prod1);
                int32x4_t t_vec2 = montgomery_reduce_neon_4(prod2);

                // Atualizar os valores de a[j] e a[j+len]
                vst1q_s32(&a[j + len], vsubq_s32(a_vec1, t_vec1));
                vst1q_s32(&a[j], vaddq_s32(a_vec1, t_vec1));

                vst1q_s32(&a[j + len + 4], vsubq_s32(a_vec2, t_vec2));
                vst1q_s32(&a[j + 4], vaddq_s32(a_vec2, t_vec2));
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

// Função de NTT radix-2 otimizada com NEON
void ntt_optimized_radix2(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    k = 0;

    // Estágios maiores onde o NEON pode ser efetivamente utilizado
    // Desenrolamento de loop para processar 8 elementos por iteração
    for (len = 128; len >= 8 ; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = zetas[++k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Duplicar o valor de zeta para vetor NEON

            // Desenrolar o loop interno para processar 8 elementos de uma vez
            for (j = start; j < start + len; j += 8) {
                // Carregar 8 coeficientes de 'a' e 8 coeficientes deslocados por 'len'
                int32x4_t a_vec_low = vld1q_s32(&a[j]);             // a[j] a[j+1] a[j+2] a[j+3]
                int32x4_t a_vec_high = vld1q_s32(&a[j + 4]);        // a[j+4] a[j+5] a[j+6] a[j+7]

                int32x4_t a_len_vec_low = vld1q_s32(&a[j + len]);    // a[j+len] a[j+len+1] a[j+len+2] a[j+len+3]
                int32x4_t a_len_vec_high = vld1q_s32(&a[j + len + 4]); // a[j+len+4] a[j+len+5] a[j+len+6] a[j+len+7]

                // Multiplicação de Montgomery para os coeficientes deslocados
                // vmull_s32 realiza a multiplicação de 32-bit para 64-bit
                int64x2_t prod_low1 = vmull_s32(vget_low_s32(a_len_vec_low), vget_low_s32(zeta_vec));
                int64x2_t prod_high1 = vmull_s32(vget_high_s32(a_len_vec_low), vget_high_s32(zeta_vec));

                int64x2_t prod_low2 = vmull_s32(vget_low_s32(a_len_vec_high), vget_low_s32(zeta_vec));
                int64x2_t prod_high2 = vmull_s32(vget_high_s32(a_len_vec_high), vget_high_s32(zeta_vec));

                // Redução de Montgomery para os primeiros 4 coeficientes
                int32x4_t t_vec_low = montgomery_reduce_neon_4((int64x2x2_t){prod_low1, prod_high1});
                // Redução de Montgomery para os próximos 4 coeficientes
                int32x4_t t_vec_high = montgomery_reduce_neon_4((int64x2x2_t){prod_low2, prod_high2});

                // Atualizar os coeficientes com as operações butterfly
                // a[j] = a[j] + t
                // a[j + len] = a[j + len] - t
                int32x4_t a_new_low = vaddq_s32(a_vec_low, t_vec_low);
                int32x4_t a_new_high = vaddq_s32(a_vec_high, t_vec_high);

                int32x4_t a_len_new_low = vsubq_s32(a_len_vec_low, t_vec_low);
                int32x4_t a_len_new_high = vsubq_s32(a_len_vec_high, t_vec_high);

                // Armazenar os resultados de volta no array 'a'
                vst1q_s32(&a[j], a_new_low);
                vst1q_s32(&a[j + 4], a_new_high);
                vst1q_s32(&a[j + len], a_len_new_low);
                vst1q_s32(&a[j + len + 4], a_len_new_high);
            }
        }
    }

    // Estágios menores, onde NEON não faz sentido ou é menos eficiente
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

void ntt_yeongbeom(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    k = 0;

    // Estágios maiores, otimizados usando as técnicas de Yongbeom
    for (len = 128; len >= 8; len >>= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = zetas[++k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Duplicar o valor de zeta para vetor NEON

            for (j = start; j < start + len; j += 8) {
                // Carregar 8 coeficientes (4 a cada vez) usando NEON
                int32x4_t a_vec1 = vld1q_s32(&a[j]);
                int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);
                int32x4_t a_len1 = vld1q_s32(&a[j + len]);
                int32x4_t a_len2 = vld1q_s32(&a[j + len + 4]);

                // Aplicação da borboleta (Butterfly 1) - técnica de Yongbeom
                int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_len1), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len1), vget_high_s32(zeta_vec)) };
                int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_len2), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_len2), vget_high_s32(zeta_vec)) };

                int32x4_t t_vec1 = montgomery_reduce_neon_4(prod1);
                int32x4_t t_vec2 = montgomery_reduce_neon_4(prod2);

                // Adição e subtração com redução modular para Borboleta 2
                vst1q_s32(&a[j + len], vsubq_s32(a_vec1, t_vec1));
                vst1q_s32(&a[j], vaddq_s32(a_vec1, t_vec1));

                vst1q_s32(&a[j + len + 4], vsubq_s32(a_vec2, t_vec2));
                vst1q_s32(&a[j + 4], vaddq_s32(a_vec2, t_vec2));

                // Máscara para garantir o ajuste dos coeficientes (Butterfly 2)
                int64_t w9 = (int64_t)a[j + len] * zeta;
                int64_t t = montgomery_reduce(w9);
                a[j + len] = a[j] - t;
                a[j] = a[j] + t;
            }
        }
    }

    // Estágios menores, onde não vale a pena usar NEON
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



void invntt_tomont(int32_t a[N]) {
  unsigned int start, len, j, k;
  int32_t t, zeta;
  const int32_t f = 41978; // mont^2/256

  k = 256;
  for(len = 1; len < N; len <<= 1) {
    for(start = 0; start < N; start = j + len) {
      zeta = -zetas[--k];
      for(j = start; j < start + len; ++j) {
        t = a[j];
        a[j] = t + a[j + len];
        a[j + len] = t - a[j + len];
        a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
      }
    }
  }

  for(j = 0; j < N; ++j) {
    a[j] = montgomery_reduce((int64_t)f * a[j]);
  }
}

void invntt_tomont_radix2(int32_t a[N]) {
   unsigned int len, start, j, k;
    int32_t zeta;
    const int32_t f = 41978;  // mont^2 / 256
    k = 256;

    // Estágios menores, onde NEON não faz sentido - Executados primeiro
    for (len = 1; len < 8; len <<= 1) {
        for (start = 0; start < N; start += 2 * len) {
            if (k == 0) break; // Garantir que k não seja decrementado para um valor inválido
            zeta = -zetas[--k];
            for (j = start; j < start + len; ++j) {
                int32_t t = a[j];
                a[j] = t + a[j + len];
                a[j + len] = t - a[j + len];
                a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
            }
        }
    }

    // Estágios maiores, otimizados com NEON
    for (len = 8; len <= 128; len <<= 1) {
        for (start = 0; start < N; start += 2 * len) {
            if (k < 2) break; // Garantir que k não seja decrementado para um valor inválido
            int32x4x2_t zeta_vec;
            zeta_vec.val[0] = vdupq_n_s32(-zetas[--k]);
            zeta_vec.val[1] = vdupq_n_s32(-zetas[--k]);

            for (j = start; j < start + len; j += 8) {
                // Carregar 8 coeficientes de a
                if (j + len + 7 < N) { // Garantir que os acessos estejam dentro dos limites do array 'a'
                    int32x4x2_t a_vec = vld1q_s32_x2(&a[j]);
                    int32x4x2_t b_vec = vld1q_s32_x2(&a[j + len]);

                    // c ← a
                    int32x4x2_t c_vec = a_vec;

                    // a ← b + c
                    a_vec.val[0] = vaddq_s32(c_vec.val[0], b_vec.val[0]);
                    a_vec.val[1] = vaddq_s32(c_vec.val[1], b_vec.val[1]);

                    // Barrett Reduction otimizado para evitar latências
                    a_vec.val[0] = vqrdmulhq_n_s32(a_vec.val[0], f);
                    a_vec.val[1] = vqrdmulhq_n_s32(a_vec.val[1], f);

                    // b ← b - c
                    b_vec.val[0] = vsubq_s32(c_vec.val[0], b_vec.val[0]);
                    b_vec.val[1] = vsubq_s32(c_vec.val[1], b_vec.val[1]);

                    // b ← b * ζ (utilizando instrução FQMUL otimizada)
                    b_vec.val[0] = vmulq_s32(b_vec.val[0], zeta_vec.val[0]);
                    b_vec.val[1] = vmulq_s32(b_vec.val[1], zeta_vec.val[1]);

                    // Guardar os resultados no array 'a'
                    vst1q_s32(&a[j], a_vec.val[0]);
                    vst1q_s32(&a[j + 4], a_vec.val[1]);
                    vst1q_s32(&a[j + len], b_vec.val[0]);
                    vst1q_s32(&a[j + len + 4], b_vec.val[1]);
                }
            }
        }
    }

    // Multiplicação final por mont^2 / 256 utilizando NEON para 8 coeficientes
    for (j = 0; j < N; j += 8) {
        if (j + 7 < N) { // Garantir que os acessos estejam dentro dos limites do array 'a'
            int32x4_t a_vec1 = vld1q_s32(&a[j]);
            int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);

            int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_vec1), vdup_n_s32(f)),
                                  vmull_s32(vget_high_s32(a_vec1), vdup_n_s32(f)) };
            int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_vec2), vdup_n_s32(f)),
                                  vmull_s32(vget_high_s32(a_vec2), vdup_n_s32(f)) };

            int32x4_t reduced_vec1 = montgomery_reduce_neon_4(prod1);
            int32x4_t reduced_vec2 = montgomery_reduce_neon_4(prod2);

            vst1q_s32(&a[j], reduced_vec1);
            vst1q_s32(&a[j + 4], reduced_vec2);
        }
    }
}

void invntt_tomont_radix4(int32_t a[N]) {  
    const int32_t f = 41978;  // mont^2 / 256        
    unsigned int len, start, j, k;
    int32x4_t zeta_vec1, zeta_vec2, t_vec1, t_vec2, a_vec1, a_vec2, a_vec3, a_vec4;
    int32_t zeta;

    k = 256; // Inicializando k para percorrer os valores de zetas de forma correta
    for(len = 64; len > 0; len >>= 2) { // Radix-4 inverso, reduzindo len por 4 a cada iteração
        for(start = 0; start < N; start += 4 * len) {
            for(j = start; j < start + len; j += 4) {
                // Garantir que não ultrapasse o tamanho do vetor zetas
                if (k < 4) {
                    continue; // Continuar o laço para evitar acessar valores inválidos de zeta
                }

                // Carregar quatro coeficientes de cada vez
                if (j + 3 * len < N) { // Garantir que os acessos estejam dentro dos limites do array 'a'
                    a_vec1 = vld1q_s32(&a[j]);
                    a_vec2 = vld1q_s32(&a[j + len]);
                    a_vec3 = vld1q_s32(&a[j + 2 * len]);
                    a_vec4 = vld1q_s32(&a[j + 3 * len]);

                    // Carregar os valores de zeta para cada butterfly
                    int32_t zeta1 = -zetas[--k];
                    int32_t zeta2 = -zetas[--k];
                    int32_t zeta3 = -zetas[--k];
                    int32_t zeta4 = -zetas[--k];

                    // Aplicar os valores de zeta
                    zeta_vec1 = vdupq_n_s32(zeta1);
                    zeta_vec2 = vdupq_n_s32(zeta2);

                    // Atualizar os coeficientes (adição e subtração)
                    int32x4_t tmp_vec1 = vaddq_s32(a_vec1, a_vec2);
                    a_vec2 = vsubq_s32(a_vec1, a_vec2);
                    a_vec1 = tmp_vec1;

                    int32x4_t tmp_vec2 = vaddq_s32(a_vec3, a_vec4);
                    a_vec4 = vsubq_s32(a_vec3, a_vec4);
                    a_vec3 = tmp_vec2;

                    // Multiplicação de Montgomery inversa para zeta_vec1 e zeta_vec2
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

                    // Multiplicar por f para normalizar após a NTT inversa
                    int32x4_t f_vec = vdupq_n_s32(f);
                    t_vec1 = vmulq_s32(t_vec1, f_vec);
                    t_vec2 = vmulq_s32(t_vec2, f_vec);

                    // Armazenar os resultados de volta no vetor
                    vst1q_s32(&a[j], a_vec1);
                    vst1q_s32(&a[j + len], t_vec1);
                    vst1q_s32(&a[j + 2 * len], a_vec3);
                    vst1q_s32(&a[j + 3 * len], t_vec2);
                }
            }
        }
    }
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


// Função de benchmark usando Google Benchmark
static void BM_ntt_referencia(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_radix2(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_neon_4(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_optimized_radix2(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_optimized_radix2(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_everaldo1(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_everaldo1(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_everaldo2(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_everaldo2(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_neon_4(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_neon_4(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_ntt_radix4(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_radix4(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}


// Função de benchmark usando Google Benchmark
static void BM_ntt_yeongbeom(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        ntt_yeongbeom(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}


// Função de benchmark usando Google Benchmark
static void BM_invntt_tomont_referencia(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        invntt_tomont(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

// Função de benchmark usando Google Benchmark
static void BM_invntt_tomont_radix4(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        invntt_tomont_radix4(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

static void BM_invntt_tomont_radix2(benchmark::State& state) {    
    int32_t a[N] = {0}; // Inicialização do vetor de entrada
    uint64_t start_cycles, end_cycles, total_cycles = 0;
    for (auto _ : state) {
        start_cycles = cpucycles();
        invntt_tomont_radix2(a);
        end_cycles = cpucycles();
        total_cycles += (end_cycles - start_cycles);
    }
    state.counters["Ciclos"] = total_cycles / state.iterations();
}

    
BENCHMARK(BM_ntt_referencia);
BENCHMARK(BM_ntt_everaldo1);
BENCHMARK(BM_ntt_everaldo2);
BENCHMARK(BM_ntt_radix2);
BENCHMARK(BM_ntt_yeongbeom);
BENCHMARK(BM_ntt_optimized_radix2);
BENCHMARK(BM_ntt_neon_4);
BENCHMARK(BM_ntt_radix4);

BENCHMARK(BM_invntt_tomont_referencia);
BENCHMARK(BM_invntt_tomont_radix2);
BENCHMARK(BM_invntt_tomont_radix4);


int main(int argc, char** argv) {
    // Definir afinidade para o núcleo de desempenho 0
    set_cpu_affinity(4);

    // Inicializar o Google Benchmark
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
}


