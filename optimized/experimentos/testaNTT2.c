#include <stdint.h>
#include <stdio.h>
#include <arm_neon.h>  // Biblioteca para intrínsecos NEON

#include <time.h>      // Para medição de tempo


#define N 256
#define Q 8380417
#define ZETA 1753 // A raíz 512ª da unidade módulo q
#define R (1ULL << 32) % Q // 2^32 mod Q, usado para o domínio de Montgomery
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define R2INV 4091  // R^-2 mod Q, com R=2^16

// Precomputed zetas array - VALORES DO CODIGO DE REFERENCIA - NÃO ALTERAR
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

// Função que inverte os bits de um byte de 8 bits
uint8_t BitRev8(uint8_t m) {
    uint8_t r = 0;
    for (int i = 0; i < 8; i++) {
        r <<= 1;
        r |= (m & 1);
        m >>= 1;
    }
    return r;
}

// Redução de Montgomery ingênua - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
int32_t montgomery_reduce1(int64_t a) {
    int32_t t;

    t = (int64_t)(int32_t)a * QINV;
    t = (a - (int64_t)t * Q) >> 32;

    // Normalização para garantir que o resultado esteja dentro de -Q a Q
    if (t < 0) {
        t += Q;
    } else if (t >= Q) {
        t -= Q;
    }

    return t;
}

// Redução de Montgomery otimizada - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
static inline int32_t montgomery_reduce(int64_t a) {
    int32_t t = (int32_t)((a * QINV) & 0xFFFFFFFF);  // Multiplicação com QINV e mask para 32 bits
    int32_t result = (a - (int64_t)t * Q) >> 32;     // Subtração e shift para 32 bits
    return (result < 0) ? (result + Q) : result;     // Normalização se necessário
}

// Função para aplicar redução em vetores NEON
static int32x4_t montgomery_reduce_vector(int64x2_t vec_low, int64x2_t vec_high) {
    int32x4_t result;

    int32_t t0 = montgomery_reduce(vgetq_lane_s64(vec_low, 0));
    int32_t t1 = montgomery_reduce(vgetq_lane_s64(vec_low, 1));
    int32_t t2 = montgomery_reduce(vgetq_lane_s64(vec_high, 0));
    int32_t t3 = montgomery_reduce(vgetq_lane_s64(vec_high, 1));

    result = vsetq_lane_s32(t0, result, 0);
    result = vsetq_lane_s32(t1, result, 1);
    result = vsetq_lane_s32(t2, result, 2);
    result = vsetq_lane_s32(t3, result, 3);

    return result;
}


// Converte para o domínio de Montgomery - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
int32_t to_montgomery(int32_t a) {
    return montgomery_reduce((int64_t)a * R);
}

// Converte de volta do domínio de Montgomery - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
int32_t from_montgomery(int32_t a) {
    return montgomery_reduce((int64_t)a);
}

//**************************************************************************
// Função NTT padrão - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
void ntt(int32_t a[N]) {
    unsigned int len, start, j, k;
    int32_t zeta, t;

    k = 0;
    for (len = 128; len > 0; len >>= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = zetas[++k];
            for (j = start; j < start + len; ++j) {
                t = montgomery_reduce((int64_t)zeta * a[j + len]);
                a[j + len] = a[j] - t;
                if (a[j + len] < 0) a[j + len] += Q;
                a[j] = a[j] + t;
                if (a[j] >= Q) a[j] -= Q;
            }
        }
    }
}


// Função inversa NTT no domínio de Montgomery - ESTÁ FUNCIONANDO OK - NÃO ALTERAR
void invntt_tomont1(int32_t a[N]) {
    unsigned int start, len, j, k;
    int32_t t, zeta;
    const int32_t f = 41978; // mont^2/256

    k = 256;
    for (len = 1; len < N; len <<= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = -zetas[--k];
            for (j = start; j < start + len; ++j) {
                t = a[j];
                a[j] = t + a[j + len];
                if (a[j] >= Q) a[j] -= Q;
                a[j + len] = t - a[j + len];
                if (a[j + len] < 0) a[j + len] += Q;
                a[j + len] = montgomery_reduce((int64_t)zeta * a[j + len]);
            }
        }
    }

    for (j = 0; j < N; ++j) {
        a[j] = montgomery_reduce((int64_t)f * a[j]);
    }
}
//***************************************************************************

// Função de redução modular customizada
static inline int32x4_t custom_reduce(int32x4_t x) {
    // Multiplica x por QINV e pega os 16 bits menos significativos
    int32x4_t u = vmulq_n_s32(x, QINV);
    u = vshrq_n_s32(u, 16);  // Desloca para direita 16 bits (u * Q)
    
    int32x4_t qx = vmulq_n_s32(u, Q);
    int32x4_t result = vsubq_s32(x, qx);  // x - u * Q

    // Correção final se o resultado for negativo
    int32x4_t mask = vcltzq_s32(result);  // Verifica negativos
    result = vaddq_s32(result, vandq_s32(mask, vdupq_n_s32(Q)));

    return result;
}

// Função NTT otimizada usando NEON
void ntt_otimizada(int32_t *a, const int32_t *zetas, int n) {
    int logn = 0;
    for (int i = n; i > 1; i >>= 1) logn++;

    // Processamento NTT utilizando SIMD e lógica customizada de redução
    for (int len = n / 2; len > 0; len >>= 1) {
        int32_t *end = a + n;
        for (int i = 0; i < n; i += (len << 1)) {
            int32x4_t psi = vld1q_s32(&zetas[i >> (logn - len)]);
            for (int j = 0; j < len; j += 4) {
                int32x4_t x1 = vld1q_s32(&a[i + j]);
                int32x4_t x2 = vld1q_s32(&a[i + j + len]);

                // Redução modular direta nos produtos de x2 e psi
                x2 = vmulq_s32(x2, psi);
                x2 = custom_reduce(x2);

                // Atualização dos valores de a[i + j] e a[i + j + len]
                int32x4_t x = vaddq_s32(x1, x2);
                int32x4_t y = vsubq_s32(x1, x2);

                vst1q_s32(&a[i + j], custom_reduce(x));
                vst1q_s32(&a[i + j + len], custom_reduce(y));
            }
        }
        logn--;
    }
}



// Função inversa NTT otimizada
void invntt_tomont(int32_t a[N]) {
    unsigned int start, len, j, k;
    int32_t t, zeta;
    const int32_t f = 41978; // mont^2/256

    k = 256;
    for (len = 1; len < N; len <<= 1) {
        for (start = 0; start < N; start = j + len) {
            zeta = -zetas[--k];
            int32x4_t zeta_vec = vdupq_n_s32(zeta);  // Broadcast zeta

            for (j = start; j < start + len; j += 4) {
                int32x4_t a_vec1 = vld1q_s32(&a[j]);  // Carrega 4 elementos
                int32x4_t a_vec2 = vld1q_s32(&a[j + len]);  // Carrega 4 elementos
                int32x4_t t_vec = vsubq_s32(a_vec1, a_vec2);  // Subtração
                vst1q_s32(&a[j], vaddq_s32(a_vec1, a_vec2));  // Adiciona e salva
                t_vec = vmulq_s32(zeta_vec, t_vec);  // Multiplicação
                t_vec = vshrq_n_s32(t_vec, 16);  // Shift para redução (ajustado para evitar overflow)
                vst1q_s32(&a[j + len], t_vec);  // Salva
            }
        }
    }

    int32x4_t f_vec = vdupq_n_s32(f);  // Broadcast de f para vetor
    for (j = 0; j < N; j += 4) {
        int32x4_t a_vec = vld1q_s32(&a[j]);  // Carrega 4 elementos
        a_vec = vmulq_s32(f_vec, a_vec);  // Multiplicação com f
        a_vec = vshrq_n_s32(a_vec, 16);  // Shift para redução (ajustado para evitar overflow)
        vst1q_s32(&a[j], a_vec);  // Salva
    }
}


//********************************************************************************************************************


// Função para inicializar o vetor com uma entrada determinística
void initialize_input(int32_t a[N]) {
    printf("Inicializando vetor de entrada com conversão para montgomery\n");
    for (int i = 0; i < N; i++) {
        a[i] = to_montgomery(i);  // Exemplo simples de inicialização determinística
    }
}

void initialize_input2(int32_t a[N]) {
    for (int i = 0; i < N; i++) {
        a[i] = montgomery_reduce((int64_t)i * R);  // Usando a função de redução para inicialização
    }
}

// Função para comparar resultados
int compare_arrays(int32_t a[N], int32_t b[N]) {
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            printf("Diferença encontrada no índice %d: a = %d, b = %d\n", i, a[i], b[i]);
            return 0; // Arrays não são iguais
        }
    }
    return 1; // Arrays são iguais
}

// Função para imprimir o array para debug
void print_array(const char* label, int32_t a[N]) {
    printf("%s: ", label);
    for (int i = 0; i < N; i++) {
        printf("%d ", a[i]);
    }
    printf("\n\n");
}

// Função para medir o tempo de execução usando clock_gettime()
double measure_time(void (*func)(int32_t*), int32_t a[N]) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    func(a);
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (end.tv_sec - start.tv_sec) * 1e6 + (end.tv_nsec - start.tv_nsec) / 1e3; // Tempo em microsegundos
}

int main() {
    int32_t poly[N], poly_copy[N], poly_otimizada[N]; 
    initialize_input(poly);
    initialize_input(poly_copy);
    initialize_input(poly_otimizada);
    compare_arrays(poly, poly_otimizada);

   // Medindo tempo para a NTT padrão
    //double time_standard = measure_time(ntt, poly);
    //print_array("Resultado NTT Padrão", poly);
    //printf("Tempo NTT Padrão: %.2f microsegundos\n\n", time_standard);
    ntt(poly);

    // Medindo tempo para a NTT otimizada
    //double time_optimized = measure_time(ntt_otimizada, poly_otimizada);
    //print_array("Resultado NTT Otimizada", poly_otimizada);
    //printf("Tempo NTT Otimizada: %.2f microsegundos\n\n", time_optimized);
    ntt_otimizada(poly_otimizada,zetas,N);

    // Comparar resultados
    if (compare_arrays(poly, poly_otimizada)) {
        printf("NTT padrão e otimizada produziram resultados idênticos.\n");
    } else {
        printf("Diferenças encontradas entre NTT padrão e otimizada.\n");
    }


    /*
    // Realizar a NTT inversa
    invntt_tomont(poly);
    print_array("Resultado INVNTT", poly);

    // Converter de volta do domínio de Montgomery
    for (int i = 0; i < N; i++) {
        poly[i] = from_montgomery(poly[i]);
    }

    print_array("Resultado Final", poly);

    // Comparar resultados para verificar se o array retornou ao seu estado original
    if (compare_arrays(poly, poly_copy)) {
        printf("Teste NTT e INTT bem-sucedido\n");
    } else {
        printf("Teste NTT e INTT falhou\n");
    }
    */

    return 0;
}
