//****************************************************************************************************************************************************************
// Autor: Everaldo Alves
// Data: 19 de Setembro/2024
// Função: poly_ntt
// Descrição: Esta função realiza a transformação de um polinômio para o domínio da NTT. A função montgomery_reduce também foi otimizada para NEON.
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para um speed-up de 1.27x em relação à implementação de referência
//****************************************************************************************************************************************************************

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <string.h>


#define N 256
#define Q 8380417
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define REPEAT 10000
#define NUM_CORES 4  // Número de núcleos de alto desempenho do M1


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



// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}
// Função para a Lei de Amdahl
double amdahl_law(double P, int n) {
    return 1.0 / ((1 - P) + (P / n));
}

// Função para medir eficiência
double calculate_efficiency(double speedup, int num_cores) {
    return speedup / num_cores;
}

// Função para medir o tempo de execução de uma função NTT
double measure_time(void (*ntt_func)(int32_t *), int32_t *a) {
    clock_t start, end;
    start = clock();
    ntt_func(a);
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}

//****************************************************************************************************************************************************************
typedef struct {
    int32_t coeffs[N];
} poly;

// Função montgomery_reduce fornecida no código original
int32_t montgomery_reduce(int64_t a) {
    int32_t t;    

    t = (int64_t)(int32_t)a * QINV;
    t = (a - (int64_t)t * Q) >> 32;
    
    return t;
    
}

// NTT original 
void ntt_reference(int32_t *a) {
  unsigned int len, start, j, k,i,w;
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
//****************************************************************************************************************************************************************

// Função para realizar a NTT com Decimation In Frequency (DIF) usando SIMD NEON

// Função de bit-reversal para reorganizar a saída
uint16_t bit_reverse(uint16_t x, int logn) {
    uint16_t n = 0;
    for (int i = 0; i < logn; i++) {
        n <<= 1;
        n |= (x & 1);
        x >>= 1;
    }
    return n;
}

// Reorganiza os coeficientes de entrada para a ordem de bit-reverso
void bit_reverse_order(int32_t *a) {
    int logn = 8;  // Para N=256, log_2(256) = 8
    for (uint16_t i = 0; i < N; i++) {
        uint16_t j = bit_reverse(i, logn);
        if (i < j) {
            int32_t temp = a[i];
            a[i] = a[j];
            a[j] = temp;
        }
    }
}

// Função para aplicar a redução de Montgomery em um vetor NEON
void apply_montgomery_reduce(int32x4_t *t2) {
    int32_t vals[4];
    vst1q_s32(vals, *t2);  // Extrair os valores do vetor NEON para um array
    for (int i = 0; i < 4; i++) {
        vals[i] = montgomery_reduce((int64_t)vals[i]);  // Aplicar a redução de Montgomery
    }
    *t2 = vld1q_s32(vals);  // Carregar os valores de volta para o vetor NEON
}

// Função para realizar a NTT com DIF usando SIMD NEON
void ntt_dif(int32_t *a) {
    int len, j, k;
    int32x4_t t1, t2, z;

    k = 0;
    for (len = 2; len <= N; len <<= 1) {
        printf("Stage with len = %d\n", len);
        for (j = 0; j < N; j += len) {
            for (int i = 0; i < len / 2; i += 4) {
                // Garantir que k não ultrapasse o tamanho do vetor zetas
                if (k + 4 > N) {
                    printf("Fim do vetor zetas, k = %d\n", k);
                    return;
                }
                
                // Carregar 4 valores de zetas em um vetor NEON
                int32x4_t zetas_vec = vld1q_s32(&zetas[k]);
                
                // Extrair e printar os valores de zetas_vec para verificação
                int32_t zetas_vals[4];
                vst1q_s32(zetas_vals, zetas_vec);
                printf("k = %d, zetas_vec = [%d, %d, %d, %d]\n", k, zetas_vals[0], zetas_vals[1], zetas_vals[2], zetas_vals[3]);

                k += 4;

                // Carregar 4 elementos do vetor de entrada usando NEON
                t1 = vld1q_s32(&a[j + i]);
                t2 = vld1q_s32(&a[j + i + len / 2]);

                // Multiplicação de t2 com os valores de zeta no domínio NEON
                t2 = vmulq_s32(t2, zetas_vec);

                // Aplicar a redução de Montgomery após a multiplicação
                apply_montgomery_reduce(&t2);

                // Cálculos de adição e subtração sem modular reduction (como na referência)
                int32x4_t res1 = vaddq_s32(t1, t2);  // a[j] + zeta * a[j + len/2]
                int32x4_t res2 = vsubq_s32(t1, t2);  // a[j] - zeta * a[j + len/2]

                // Armazenar os resultados de volta no vetor original
                vst1q_s32(&a[j + i], res1);
                vst1q_s32(&a[j + i + len / 2], res2);
            }
        }
    }
}

// Função montgomery_reduce aplicada a vetores NEON
static inline int32x4_t neon_montgomery_reduce(int64x2_t a_lo, int64x2_t a_hi) {
    // Quebra o vetor de 64 bits em dois vetores de 32 bits para aplicar montgomery_reduce
    int32_t a0[2], a1[2];
    int32x4_t result;
    uint8_t i = 0;
    
    if (i==0) {
        int64_t vetor_lo[2];        
        int64_t vetor_hi[2];
        vst1q_s64(vetor_lo, a_lo);
        vst1q_s64(vetor_hi, a_hi);        
        printf("\nEntrada para função neon_montgomery_reduce: ");
        printf("%lld, %lld, %lld, %lld", vetor_lo[0],vetor_lo[1],vetor_hi[0],vetor_hi[1]);
    }
    // Extrai os resultados dos vetores de 64 bits
    a0[0] = montgomery_reduce(vgetq_lane_s64(a_lo, 0));
    a0[1] = montgomery_reduce(vgetq_lane_s64(a_lo, 1));
    a1[0] = montgomery_reduce(vgetq_lane_s64(a_hi, 0));
    a1[1] = montgomery_reduce(vgetq_lane_s64(a_hi, 1));

    // Reconstroi o vetor int32x4_t com os resultados
    result = vsetq_lane_s32(a0[0], result, 0);
    result = vsetq_lane_s32(a0[1], result, 1);
    result = vsetq_lane_s32(a1[0], result, 2);
    result = vsetq_lane_s32(a1[1], result, 3);

    if (i==0) {
        int32_t vetor1[4];
        vst1q_s32(vetor1, result);
        printf("\nSaída da função neon_montgomery_reduce: ");
        printf("%d, %d, %d, %d", vetor1[0],vetor1[1],vetor1[2],vetor1[3]);
        i++;
    }

    return result;
}

void ntt_2(int32_t a[N]) {
    int len, start, j;
    int32x4_t z, t, w_j, w_j_len;

    len = 128;
    while (len >= 1) {
        start = 0;
        while (start < 256) {
            // Carrega zeta
            z = vdupq_n_s32(zetas[len]);  // Carrega zeta em vetor

            for (j = start; j < start + len; j += 4) {
                // Carrega quatro valores de w[j] e w[j + len]
                w_j = vld1q_s32(&a[j]);
                w_j_len = vld1q_s32(&a[j + len]);

                // t <- (z * w[j + len]) mod q
                int64x2_t prod_lo = vmull_s32(vget_low_s32(z), vget_low_s32(w_j_len));
                int64x2_t prod_hi = vmull_s32(vget_high_s32(z), vget_high_s32(w_j_len));
                t = neon_montgomery_reduce(prod_lo, prod_hi);  // Usar montgomery_reduce
                
                // Atualiza os valores
                w_j_len = vsubq_s32(w_j, t);
                w_j_len = neon_montgomery_reduce(vmull_s32(vget_low_s32(w_j_len), vget_low_s32(vdupq_n_s32(1))),
                                                 vmull_s32(vget_high_s32(w_j_len), vget_high_s32(vdupq_n_s32(1))));
                w_j = vaddq_s32(w_j, t);
                w_j = neon_montgomery_reduce(vmull_s32(vget_low_s32(w_j), vget_low_s32(vdupq_n_s32(1))),
                                             vmull_s32(vget_high_s32(w_j), vget_high_s32(vdupq_n_s32(1))));

                // Armazena de volta os resultados
                vst1q_s32(&a[j], w_j);
                vst1q_s32(&a[j + len], w_j_len);
            }
            start += 2 * len;
        }
        len /= 2;
    }
}

//****************************************************************************************************************************************************************
// Esta implementação otimizada de montgomery_reduce está gerando saídas CORRETAS e opera em conjunto com a função NTT_3

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

// Implementação da NTT com NEON, unrolling e pré-computação
// Função NTT otimizada com NEON - DEU CERTO!!!!!
void ntt_3(int32_t *a) {
    unsigned int len, start, j, k;
    int32_t zeta, t;

    k = 0;
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = zetas[++k];

            for (j = start; j < start + len; j++) {
                // Carregar os valores de a[j] e a[j + len] em NEON
                int32_t a_j = a[j];             // Valor de a[j]
                int32_t a_j_len = a[j + len];   // Valor de a[j + len]

                // Multiplicar zeta * a[j + len]
                int64_t zeta_product = (int64_t)zeta * a_j_len;

                // Aplicar montgomery_reduce usando a função otimizada NEON
                int32x2_t zeta_vec = { zeta_product, 0 };  // Vetor com zeta_product
                int32x2_t t_vec = montgomery_reduce_neon3(vcombine_s64(vcreate_s64(zeta_product), vcreate_s64(0)));

                // Atualizar os valores a[j] e a[j + len]
                int32_t t = vget_lane_s32(t_vec, 0);
                a[j] = a_j + t;
                a[j + len] = a_j - t;
            }
        }
    }
}



//****************************************************************************************************************************************************************
// implementação mais performática

// Implementação para 4 coeficientes
int32x4_t laço_externo_ntt(int32_t a[4], int32_t a_len[4], int32_t zeta) {
    // Carregar 4 coeficientes de a e a + len em registradores NEON
    int32x4_t a_vec = vld1q_s32(a);
    int32x4_t a_len_vec = vld1q_s32(a_len);

    // Multiplicar zeta * a_len para os 4 coeficientes
    int64x2_t product_low = vmull_s32(vget_low_s32(a_len_vec), vdup_n_s32(zeta));
    int64x2_t product_high = vmull_s32(vget_high_s32(a_len_vec), vdup_n_s32(zeta));

    // Aplicar montgomery_reduce_neon para os dois blocos de coeficientes
    int32x2_t reduced_low = montgomery_reduce_neon3(product_low);
    int32x2_t reduced_high = montgomery_reduce_neon3(product_high);

    // Combinar os dois resultados em um vetor de 4 inteiros
    return vcombine_s32(reduced_low, reduced_high);
}
// Implementação para 4 coeficientes por vez
void ntt_neon_optimized(int32_t *a) {
    unsigned int len, start, j, k;
    int32_t zeta;

    k = 0;
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = zetas[++k];

            for (j = start; j < start + len; j += 4) {
                // Chamar laço_externo_ntt para processar 4 coeficientes
                int32x4_t result = laço_externo_ntt(&a[j], &a[j + len], zeta);

                // Armazenar os resultados em a[j] e a[j + len]
                vst1q_s32(&a[j], vaddq_s32(vld1q_s32(&a[j]), result));
                vst1q_s32(&a[j + len], vsubq_s32(vld1q_s32(&a[j]), result));
            }
        }
    }
}

// Implementação para 8 coeficientes
int32x4x2_t montgomery_reduce_neon4(int64x2x2_t a) {
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
    int32x4x2_t result;
    result.val[0] = vcombine_s32(reduced1, reduced1);
    result.val[1] = vcombine_s32(reduced2, reduced2);

    return result;
}

// Implementação para 8 coeficientes
int32x4x2_t laço_externo_ntt4(int32_t a[8], int32_t a_len[8], int32_t zeta) {
    // Carregar 8 coeficientes de a e a_len em registradores NEON
    int32x4x2_t a_vec;
    a_vec.val[0] = vld1q_s32(&a[0]);  // Carrega os primeiros 4 coeficientes
    a_vec.val[1] = vld1q_s32(&a[4]);  // Carrega os próximos 4 coeficientes

    int32x4x2_t a_len_vec;
    a_len_vec.val[0] = vld1q_s32(&a_len[0]);
    a_len_vec.val[1] = vld1q_s32(&a_len[4]);

    // Multiplicar zeta * a_len para os 8 coeficientes
    int64x2x2_t product;
    product.val[0] = vmull_s32(vget_low_s32(a_len_vec.val[0]), vdup_n_s32(zeta));
    product.val[1] = vmull_s32(vget_high_s32(a_len_vec.val[1]), vdup_n_s32(zeta));

    // Aplicar montgomery_reduce_neon3 para os 8 coeficientes de uma vez
    return montgomery_reduce_neon4(product);
}

// Implementação para 4 coeficientes por vez
void ntt_neon_optimized4(int32_t *a) {
    unsigned int len, start, j, k;
    int32_t zeta;

    k = 0; // Certifique-se de iniciar com k = 0
    for (len = 128; len > 0; len >>= 1) {
        printf("Stage with len = %d \n", len);
        for (start = 0; start < N; start = j + len) {
            zeta = zetas[++k];  // Usar zetas[k] antes de incrementar k

            for (j = start; j < start + len; j += 8) {
                // Chamar laço_externo_ntt para processar 8 coeficientes
                int32x4x2_t result = laço_externo_ntt4(&a[j], &a[j + len], zeta);

                // Armazenar os resultados nos respectivos blocos de 8 coeficientes
                vst1q_s32(&a[j], vaddq_s32(vld1q_s32(&a[j]), result.val[0]));
                vst1q_s32(&a[j + 4], vaddq_s32(vld1q_s32(&a[j + 4]), result.val[1]));

                vst1q_s32(&a[j + len], vsubq_s32(vld1q_s32(&a[j]), result.val[0]));
                vst1q_s32(&a[j + len + 4], vsubq_s32(vld1q_s32(&a[j + 4]), result.val[1]));

                // Printar os valores de k, zeta, j, a[j] e a[j + len] para verificar a correção
                printf("k: %d, zeta: %d, j: %d ", k, zeta, j);
                printf("a[j]: %d %d %d %d ", a[j], a[j+1], a[j+2], a[j+3]);
                printf("a[j + len]: %d %d %d %d\n", a[j + len], a[j + 1+ len], a[j + 2 + len], a[j + 3 + len]);
            }
        }
    }
}



// Esta versão trouxe assertividade e aceleração de 1.5x em relação à implementação de referência
// Função Montgomery de Redução Otimizada para 4 coeficientes
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
void ntt_neon_4(int32_t *a) {
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




// Função para comparar os resultados das duas NTTs
int compare_ntt(const int32_t *a, const int32_t *b) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            return 0;  // As saídas são diferentes
        }
    }
    return 1;  // As saídas são iguais
}



//****************************************************************************************************************************************************************
// Programa principal
//****************************************************************************************************************************************************************

int main() {
    // Declarar dois polinômios para testar as funções
    poly poly_ref, poly_dif;
    
     
        // Preencher ambos os polinômios com os mesmos coeficientes
        for (int i = 0; i < N; i++) {
            poly_ref.coeffs[i] = rand() % Q;  // Inicialização aleatória
            poly_dif.coeffs[i] = poly_ref.coeffs[i];  // Mesmos valores para garantir a comparação justa
        }

        if (compare_ntt(poly_ref.coeffs, poly_dif.coeffs)) {
            printf("\nOs polinômios de entrada são iguais.\n");
        } else {
            printf("\nOs polinômios de entrada são diferentes.\n");
        }


        //**********************************************************************************************************************************************************************
        // Medição de TEMPO
        //**********************************************************************************************************************************************************************
        printf("\n\nfunção NTT de Referência\n");
        double time_ref = measure_time(ntt_reference, poly_ref.coeffs);

        // Medir o tempo de execução da função ntt_otimizada
        printf("\n\nfunção NTT OTIMIZADA\n");
        double time_dif = measure_time(ntt_neon_4, poly_dif.coeffs);

        // Comparar as saídas das duas funções
        if (compare_ntt(poly_ref.coeffs, poly_dif.coeffs)) {
            printf("\nSUCESSO! - As saídas das duas funções NTT são iguais.\n");
        } else {
            printf("\n * * * ERRO !!!!!!!!!!!!!!!!!!!! - As saídas das duas funções NTT são diferentes.\n");
            /*
            printf("\n\nSaída da função NTT :\n");
            for (int i = 0; i < N; i++) {
                printf("%d ", poly_ref.coeffs[i]);
                if ((i + 1) % 16 == 0) printf("\n");  // Quebrar em linhas de 8 coeficientes
            }
            printf("\n\nSaída da função NTT OTIMIZADA :\n");
            for (int i = 0; i < N; i++) {
                printf("%d ", poly_dif.coeffs[i]);
                if ((i + 1) % 16 == 0) printf("\n");  // Quebrar em linhas de 8 coeficientes
            }
            */
            
        }

        // Exibir os tempos de execução
        printf("\n\nTempo da função de referência: %f segundos\n", time_ref);
        printf("Tempo da função NTT OTIMIZADA: %f segundos\n", time_dif);

        // Comparar os tempos de execução
        if (time_dif < time_ref) {
            printf("Sucesso! - A função NTT OTIMIZADA é mais rápida.\n");
        } else {
            printf("Erro! - A função de referência é mais rápida.\n");
        }


        //**********************************************************************************************************************************************************************
        // Medição de CICLOS
        //**********************************************************************************************************************************************************************
        
        // Preencher ambos os polinômios com os mesmos coeficientes
        for (int i = 0; i < N; i++) {
            poly_ref.coeffs[i] = rand() % Q;  // Inicialização aleatória
            poly_dif.coeffs[i] = poly_ref.coeffs[i];  // Mesmos valores para garantir a comparação justa
        }
        // Medir o tempo da função de referência
        uint64_t ref_time = 0, opt_time = 0;
        uint64_t start_cycles, end_cycles;

        // Executar e medir ciclos da função de referência
        for (int i = 0; i < REPEAT; i++) {
            start_cycles = cpucycles();
            ntt_reference(poly_ref.coeffs);
            end_cycles = cpucycles();
            ref_time += end_cycles - start_cycles;
        }

        // Executar e medir ciclos da função otimizada
        for (int i = 0; i < REPEAT; i++) {
            start_cycles = cpucycles();
            ntt_neon_4(poly_dif.coeffs);
            end_cycles = cpucycles();
            opt_time += end_cycles - start_cycles;
        }

   

    // Calcular o speed-up
    double avg_ref_time = (double) ref_time / REPEAT;
    double avg_opt_time = (double) opt_time / REPEAT;
    uint64_t cycles_saved = avg_ref_time - avg_opt_time;
    double speedup = avg_ref_time / avg_opt_time;

    // Mostrar o speed-up
    printf("\n\nCiclos médios da função de referência: %.2f\n", avg_ref_time);
    printf("Ciclos médios da função otimizada: %.2f\n", avg_opt_time);
    printf("Ciclos economizados: %llu\n", cycles_saved);
    printf("Speed-up: %.2f X\n", speedup);
    /*
    // Calcular e mostrar a eficiência
    double efficiency = calculate_efficiency(speedup, NUM_CORES);
    printf("\nEficiência: %.2f\n", efficiency);
    // Avaliação da eficiência da otimização com base no speed-up e eficiência
    if (efficiency > 0.75) {
        printf("A eficiência da otimização é excelente (%.2f%%). A otimização explorou bem o paralelismo disponível, o que sugere que a fração paralelizável foi corretamente identificada e explorada.\n", efficiency * 100);
    } else if (efficiency > 0.5) {
        printf("A eficiência da otimização é boa (%.2f%%). Embora o paralelismo tenha trazido melhorias significativas, ainda há espaço para otimizar a fração paralelizável do código.\n", efficiency * 100);
    } else if (efficiency > 0.3) {
        printf("A eficiência da otimização é moderada (%.2f%%). A fração paralelizável foi limitada, sugerindo que o código contém muitas seções sequenciais, o que restringe o ganho de desempenho.\n", efficiency * 100);
    } else {
        printf("A eficiência da otimização é baixa (%.2f%%). A maior parte do código parece ser sequencial, limitando drasticamente o impacto do paralelismo. Seria interessante revisar se mais partes do código podem ser paralelizadas.\n", efficiency * 100);
    }
    */

    // Aplicar a Lei de Amdahl
    double P = 0.45;  // Estimativa inicial de que 80% do código é paralelizável
    double amdahl_speedup = amdahl_law(P, NUM_CORES);
    printf("\n\nSpeed-up teórico pela Lei de Amdahl (com P=%.2f): %.2f\n", P, amdahl_speedup);

    // Considerações da Lei de Amdahl
    printf("\nConsiderações sobre a Lei de Amdahl:\n");
    printf("Com base em uma fração paralelizável de %.2f do código e %d núcleos de alto desempenho,\n", P, NUM_CORES);
    printf("o speed-up teórico máximo seria de %.2f. Portanto, o que obtivemos (%.2f) ", amdahl_speedup, speedup);
    if(amdahl_speedup <= speedup || amdahl_speedup - speedup <= 0.20) {
        printf(" pode ser considerado um BOM trabalho.\n");
    } else {
        printf(" significativamente ABAIXO do esperado. Provavelmente ainda é possível evoluir a otimização do código.\n");
    }    

    return 0;
}