#include <stdint.h>
#include "params.h"
#include "ntt.h"
#include "reduce.h"
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


// NTT Otimizada com Paralelismo. Esta versão utiliza NEON para 8 elementos e estã funcionando corretamente.
// única versão funcional
/*
void ntt(int32_t a[N]) {
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
*/

void ntt(int32_t a[N]) {
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


// Função de INTT radix-2 otimizada com NEON
void invntt_tomont(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta;
    const int32_t f = 41978;  // mont^2 / 256
    k = 256;

    // Estágios menores, onde NEON não faz sentido - Executados primeiro
    for (len = 1; len < 8; len <<= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = -zetas[--k];
            for (j = start; j < start + len; ++j) {
                int32_t t = a[j];
                a[j] = t + a[j + len];
                a[j + len] = t - a[j + len];
                a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
            }
        }
    }

    // Estágios maiores, agora para 8 coeficientes
    for (len = 8; len <= 128; len <<= 1) {
        for (start = 0; start < N; start += 2 * len) {
            zeta = -zetas[--k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Duplicar o valor de zeta para vetor NEON

            for (j = start; j < start + len; j += 8) {
                // Carregar 8 coeficientes de a
                int32x4_t a_vec1 = vld1q_s32(&a[j]);
                int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);
                int32x4_t a_len1 = vld1q_s32(&a[j + len]);
                int32x4_t a_len2 = vld1q_s32(&a[j + len + 4]);

                // Soma e subtração
                int32x4_t t_vec1 = vaddq_s32(a_vec1, a_len1);
                int32x4_t t_vec2 = vaddq_s32(a_vec2, a_len2);
                int32x4_t a_sub_vec1 = vsubq_s32(a_vec1, a_len1);
                int32x4_t a_sub_vec2 = vsubq_s32(a_vec2, a_len2);

                // Multiplicação de Montgomery para os valores subtraídos
                int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_sub_vec1), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_sub_vec1), vget_high_s32(zeta_vec)) };
                int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_sub_vec2), vget_low_s32(zeta_vec)),
                                      vmull_s32(vget_high_s32(a_sub_vec2), vget_high_s32(zeta_vec)) };

                int32x4x2_t t_reduce = montgomery_reduce_neon_8(prod1, prod2);

                // Guardar os resultados no array 'a'
                vst1q_s32(&a[j], t_vec1);
                vst1q_s32(&a[j + 4], t_vec2);
                vst1q_s32(&a[j + len], t_reduce.val[0]);
                vst1q_s32(&a[j + len + 4], t_reduce.val[1]);
            }
        }
    }

   // Multiplicação final por mont^2 / 256 utilizando NEON para 8 coeficientes
    for (j = 0; j < N; j += 8) {
        int32x4_t a_vec1 = vld1q_s32(&a[j]);
        int32x4_t a_vec2 = vld1q_s32(&a[j + 4]);

        int64x2x2_t prod1 = { vmull_s32(vget_low_s32(a_vec1), vdup_n_s32(f)),
                              vmull_s32(vget_high_s32(a_vec1), vdup_n_s32(f)) };
        int64x2x2_t prod2 = { vmull_s32(vget_low_s32(a_vec2), vdup_n_s32(f)),
                              vmull_s32(vget_high_s32(a_vec2), vdup_n_s32(f)) };

        int32x4x2_t reduced_vec = montgomery_reduce_neon_8(prod1, prod2);

        vst1q_s32(&a[j], reduced_vec.val[0]);
        vst1q_s32(&a[j + 4], reduced_vec.val[1]);
    }
}
