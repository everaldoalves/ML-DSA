#include <stdint.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"
#include <stdio.h>
#include <arm_neon.h>


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



/*************************************************
* Name:        ntt
*
* Description: Forward NTT, in-place. No modular reduction is performed after
*              additions or subtractions. Output vector is in bitreversed order.
*
* Arguments:   - uint32_t p[N]: input/output coefficient array
**************************************************/
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


/*************************************************
* Name:        ntt NEON
*
* Description: Forward NTT, in-place. No modular reduction is performed after
*              additions or subtractions. Output vector is in bitreversed order.
*
* Arguments:   - uint32_t p[N]: input/output coefficient array
**************************************************/

void ntt_neon2(int32_t a[N]) { // GERANDO VALORES INCORRETOS
    unsigned int len, start, j, k;
    int32_t zeta;

    k = 0;
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = zetas[++k];  // Fator zeta é escalar

            for (j = start; j < start + len; j += 4) {
                // Verificar se o endereço está alinhado para carregamento NEON
                if (((uintptr_t)(&a[j]) % 16 == 0) && ((uintptr_t)(&a[j + len]) % 16 == 0)) {
                    // Carregar os valores de a[j] e a[j + len] nos registros NEON (alinhado)
                    int32x4_t a_vec = vld1q_s32(&a[j]);
                    int32x4_t a_len_vec = vld1q_s32(&a[j + len]);

                    // Multiplicação escalar usando zeta para cada elemento
                    int32x4_t t_vec = vmulq_n_s32(a_len_vec, zeta);  // Multiplicação vetorizada com zeta

                    // Aplicar a redução de Montgomery vetorizada (função em 32 bits)
                    t_vec = montgomery_reduce_neon(t_vec);

                    // Realizar as operações de adição e subtração
                    int32x4_t a_result = vaddq_s32(a_vec, t_vec);       // a[j] = a[j] + t
                    int32x4_t a_len_result = vsubq_s32(a_vec, t_vec);   // a[j + len] = a[j] - t

                    // Armazenar os resultados de volta no array a
                    vst1q_s32(&a[j], a_result);                        // Armazenar a[j]
                    vst1q_s32(&a[j + len], a_len_result);               // Armazenar a[j+len]
                } else {
                    // Caso o endereço não esteja alinhado, faça o carregamento manual (não vetorizado)
                    for (int i = 0; i < 4; ++i) {
                        int32_t t = montgomery_reduce((int64_t)zeta * a[j + len + i]);
                        a[j + i] = a[j + i] + t;
                        a[j + len + i] = a[j + len + i] - t;
                    }
                }
            }
        }
    }
}

// Implementação direta com NEON sem uso de redução de montgomery
inline int32x4_t barrett_reduce(int32x4_t t, int32x4_t q_vec) {
    // Aplica a redução manualmente garantindo que o resultado esteja no intervalo [0, Q)
    int32x4_t r = t;
    int32x4_t mask = vcltq_s32(r, vdupq_n_s32(0));  // Verifica se r é negativo
    r = vaddq_s32(r, vandq_s32(mask, q_vec));       // Se negativo, adiciona Q

    mask = vcgeq_s32(r, q_vec);                     // Verifica se r >= Q
    r = vsubq_s32(r, vandq_s32(mask, q_vec));       // Se r >= Q, subtrai Q

    return r;
}

void ntt_neon(int32_t *a) {
    int32x4_t q_vec = vdupq_n_s32(Q);  // Vetor com valor do módulo Q

    for (int len = 128; len >= 1; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            for (int j = 0; j < len; j += 4) {
                int32x4_t a0 = vld1q_s32(&a[start + j]);         // Carrega a primeira metade
                int32x4_t a1 = vld1q_s32(&a[start + j + len]);   // Carrega a segunda metade

                int32x4_t zeta = vld1q_s32(&zetas[j]);           // Carrega os twiddles
                
                // Operação Butterfly: t = zeta * a1 (multiplicação modular)
                int32x4_t t = vmulq_s32(a1, zeta);
                t = barrett_reduce(t, q_vec);  // Redução modular

                // Atualiza a primeira metade
                int32x4_t a_new0 = vaddq_s32(a0, t);
                a_new0 = barrett_reduce(a_new0, q_vec);  // Redução modular
                vst1q_s32(&a[start + j], a_new0);
                
                // Atualiza a segunda metade
                int32x4_t a_new1 = vsubq_s32(a0, t);
                a_new1 = barrett_reduce(a_new1, q_vec);  // Redução modular
                vst1q_s32(&a[start + j + len], a_new1);
            }
        }
    }
}


// Função para realizar o NTT usando NEON
void ntt_neon1(int32_t a[N]) {
    for (int len = N / 2; len >= 1; len >>= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            for (int i = start; i < start + len; i += 2) {
                // Carregar coeficientes em vetores NEON (2 coeficientes de cada vez)
                int32x2_t a1 = vld1_s32(&a[i]);         // Carregar coeficientes a[i], a[i+1]
                int32x2_t a2 = vld1_s32(&a[i + len]);   // Carregar coeficientes a[i+len], a[i+len+1]

                // Carregar fator de multiplicação Zeta (2 twiddle factors)
                int32x2_t zeta = vld1_s32(&zetas[i - start]);

                // Multiplicar a2 pelos fatores zeta
                int64x2_t t = vmull_s32(a2, zeta);  // t = a2 * zeta (vetorial, dois valores ao mesmo tempo)

                // Aplicar a redução de Montgomery para cada resultado
                int32x2_t t_reduced;
                t_reduced = vset_lane_s32(montgomery_reduce(vgetq_lane_s64(t, 0)), t_reduced, 0);  // Reduzir o primeiro valor
                t_reduced = vset_lane_s32(montgomery_reduce(vgetq_lane_s64(t, 1)), t_reduced, 1);  // Reduzir o segundo valor

                // Realizar a operação borboleta (butterfly)
                int32x2_t result_a2 = vsub_s32(a1, t_reduced);  // a[i+len] = a[i] - t
                int32x2_t result_a1 = vadd_s32(a1, t_reduced);  // a[i] = a[i] + t

                // Armazenar os resultados de volta no array a
                vst1_s32(&a[i], result_a1);         // Salvar resultado em a[i], a[i+1]
                vst1_s32(&a[i + len], result_a2);   // Salvar resultado em a[i+len], a[i+len+1]
            }
        }
    }
}


/*************************************************
* Name:        invntt_tomont
*
* Description: Inverse NTT and multiplication by Montgomery factor 2^32.
*              In-place. No modular reductions after additions or
*              subtractions; input coefficients need to be smaller than
*              Q in absolute value. Output coefficient are smaller than Q in
*              absolute value.
*
* Arguments:   - uint32_t p[N]: input/output coefficient array
**************************************************/
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


/*************************************************
* Name:        invntt_tomont NEON
*
* Description: Inverse NTT and multiplication by Montgomery factor 2^32.
*              In-place. No modular reductions after additions or
*              subtractions; input coefficients need to be smaller than
*              Q in absolute value. Output coefficient are smaller than Q in
*              absolute value.
*
* Arguments:   - uint32_t p[N]: input/output coefficient array
**************************************************/
void invntt_tomont_neon(int32_t a[N]) {
    unsigned int start, len, j, k;
    int32_t zeta;
    const int32_t f = 41978; // mont^2/256

    k = 256;
    for (len = 1; len < N; len <<= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = -zetas[--k];

            for (j = start; j < start + len; j += 4) {
                int32x4_t a_vec = vld1q_s32(&a[j]);
                int32x4_t a_len_vec = vld1q_s32(&a[j + len]);

                // Operações SIMD de soma e subtração
                int32x4_t sum = vaddq_s32(a_vec, a_len_vec);
                int32x4_t diff = vsubq_s32(a_vec, a_len_vec);

                // Multiplicação escalar
                int64_t mul_res0 = (int64_t)vgetq_lane_s32(diff, 0) * zeta;
                int64_t mul_res1 = (int64_t)vgetq_lane_s32(diff, 1) * zeta;
                int64_t mul_res2 = (int64_t)vgetq_lane_s32(diff, 2) * zeta;
                int64_t mul_res3 = (int64_t)vgetq_lane_s32(diff, 3) * zeta;

                // Redução de Montgomery escalar
                int32_t t0 = montgomery_reduce(mul_res0);
                int32_t t1 = montgomery_reduce(mul_res1);
                int32_t t2 = montgomery_reduce(mul_res2);
                int32_t t3 = montgomery_reduce(mul_res3);

                // Criar o vetor de resultados de 128 bits a partir dos resultados escalonados
                int32x4_t t_vec = {t0, t1, t2, t3};

                // Atualizar o array 'a'
                vst1q_s32(&a[j], sum);
                vst1q_s32(&a[j + len], t_vec);
            }
        }
    }

    for (j = 0; j < N; j += 4) {
        int32x4_t a_vec = vld1q_s32(&a[j]);

        // Multiplicação escalar
        int64_t mul_res0 = (int64_t)vgetq_lane_s32(a_vec, 0) * f;
        int64_t mul_res1 = (int64_t)vgetq_lane_s32(a_vec, 1) * f;
        int64_t mul_res2 = (int64_t)vgetq_lane_s32(a_vec, 2) * f;
        int64_t mul_res3 = (int64_t)vgetq_lane_s32(a_vec, 3) * f;

        // Redução de Montgomery escalar
        int32_t r0 = montgomery_reduce(mul_res0);
        int32_t r1 = montgomery_reduce(mul_res1);
        int32_t r2 = montgomery_reduce(mul_res2);
        int32_t r3 = montgomery_reduce(mul_res3);

        // Criar o vetor de resultados de 128 bits a partir dos resultados escalonados
        int32x4_t result = {r0, r1, r2, r3};

        vst1q_s32(&a[j], result);
    }
}
