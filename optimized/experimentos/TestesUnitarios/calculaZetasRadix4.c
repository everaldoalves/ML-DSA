#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdbool.h>
#include <inttypes.h>
#include <arm_neon.h>


// Definições e funções auxiliares (já definidas anteriormente)
#define N 256
#define Q 8380417
#define ROOT_OF_UNITY 1753
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32

const unsigned int brv[N] = { /* valores fornecidos */ };

// Funções auxiliares
int32_t pow_mod(int32_t base, unsigned int exp, int32_t mod);
int32_t centerlift(int32_t x, int32_t q);
int32_t mod_inverse(int32_t a, int32_t m);

// Funções de NTT
int32_t* precomp_radix4();
void ntt_radix4_neon(int32_t a[N], const int32_t zetas_radix4[768]);
void ntt_radix4_reference(int32_t a[N], const int32_t zetas_radix4[768], int32_t q);

// Função de multiplicação modular para referência
int32_t fqmul_reference(int32_t a, int32_t b, int32_t q) {
    int64_t product = (int64_t)a * (int64_t)b;
    // Garantir que o resultado esteja no intervalo [0, q)
    return (int32_t)(product % q);
}

// Função para centralizar os valores no intervalo [-q/2, q/2)
int32_t centerlift(int32_t x, int32_t q) {
    if(x > q / 2) {
        return x - q;
    } else if(x < -(q / 2)) {
        return x + q;
    }
    return x;
}

// Função para calcular (base^exp) mod mod
int32_t pow_mod(int32_t base, unsigned int exp, int32_t mod) {
    int64_t result = 1;
    int64_t b = base;
    while(exp > 0) {
        if(exp & 1) {
            result = (result * b) % mod;
        }
        b = (b * b) % mod;
        exp >>= 1;
    }
    return (int32_t)result;
}

// Função para calcular o inverso de a mod m usando o Algoritmo de Euclides Estendido
int32_t mod_inverse(int32_t a, int32_t m) {
    int32_t m0 = m, t, q;
    int32_t x0 = 0, x1 = 1;

    if (m == 1)
        return 0;

    while (a > 1) {
        // q é o quociente
        q = a / m;

        t = m;

        // m é o resto agora, processa como m = a % m
        m = a % m;
        a = t;

        t = x0;

        x0 = x1 - q * x0;
        x1 = t;
    }

    // Assegurar que o inverso seja positivo
    if (x1 < 0)
        x1 += m0;

    return x1;
}

// Função para calcular a ordem multiplicativa de z em Z_q
int32_t znorder(int32_t z, int32_t q) {
    int32_t order = 1;
    int32_t power = z;
    while(power != 1) {
        power = (int32_t)(((int64_t)power * z) % q);
        order++;
        if(order > 512) {  // Limite de busca para ordem
            return 0;
        }
    }
    return order;
}

// Função para pré-computar os fatores twiddle para radix-4
int32_t* precomp_radix4() {
    // Total de zetas para radix-4: 3 * (N /4) * log4(N) = 3 * 64 * 4 = 768
    static int32_t zetas_radix4[768];
    unsigned int i, stage;

    // Calcular mont = 2^32 mod Q
    const uint64_t R = -4186625; // 2^32
    int32_t mont = (int32_t)(R % Q);  // mont = 4294967296 mod 8380417 = 4093228

    // Calcular qinv = inverse of Q mod 2^32
    // Como 2^32 é uma potência de 2, o inverso de Q deve ser um número que satisfaça Q * qinv ≡ 1 mod 2^32
    // Este valor é pré-calculado para eficiência
    int32_t qinv = 58728449; // Q=8380417, m=4294967296

    // Precomputar os fatores twiddle para cada estágio e butterfly
    // Existem 4 estágios para radix-4 (log4(256)=4)
    for(stage = 0; stage < 4; stage++) { // Estágios 0 a 3
        unsigned int len = 1 << (stage * 2); // len = 1, 4, 16, 64
        unsigned int step = len << 2;       // step = 4, 16, 64, 256

        // Para cada butterfly dentro do estágio
        for(i = 0; i < (N / 4); i++) { // 64 butterflies por estágio
            unsigned int br = brv[i]; // Índice bit-reverso

            // Calcular as três potências necessárias para radix-4
            // zeta1 = ROOT_OF_UNITY^br mod Q
            int32_t zeta1 = pow_mod(ROOT_OF_UNITY, br, Q);
            // zeta2 = ROOT_OF_UNITY^(2 * br) mod Q
            int32_t zeta2 = pow_mod(ROOT_OF_UNITY, (2 * br) % (Q -1), Q); // Usar Q-1 para expoente
            // zeta3 = ROOT_OF_UNITY^(3 * br) mod Q
            int32_t zeta3 = pow_mod(ROOT_OF_UNITY, (3 * br) % (Q -1), Q);

            // Aplicar a multiplicação por mont e redução modular
            int32_t twiddle1 = (int32_t)(((int64_t)mont * zeta1) % Q);
            int32_t twiddle2 = (int32_t)(((int64_t)mont * zeta2) % Q);
            int32_t twiddle3 = (int32_t)(((int64_t)mont * zeta3) % Q);

            // Aplicar centerlift para centralizar os valores no intervalo [-Q/2, Q/2)
            zetas_radix4[stage * 192 + i * 3 + 0] = centerlift(twiddle1, Q);
            zetas_radix4[stage * 192 + i * 3 + 1] = centerlift(twiddle2, Q);
            zetas_radix4[stage * 192 + i * 3 + 2] = centerlift(twiddle3, Q);
        }
    }

    return zetas_radix4;
}

// Implementação da redução de Montgomery para NEON
int32x4_t montgomery_reduce_neon_4(int64x2x2_t prod, int32_t q) {
    // Precomputado: qinv = inverse of Q mod 2^32
    static int32_t qinv = 0;
    if(qinv == 0) {
        qinv = mod_inverse(Q, 4294967296); // Calcular apenas uma vez
    }

    // Multiplicação com qinv
    uint32_t t0 = (uint32_t)(prod.val[0] * (uint64_t)qinv);
    uint32_t t1 = (uint32_t)(prod.val[1] * (uint64_t)qinv);

    // Multiplicar t pelo Q
    uint64_t m0 = (uint64_t)t0 * Q;
    uint64_t m1 = (uint64_t)t1 * Q;

    // Subtrair de prod
    int64_t r0 = prod.val[0] - m0;
    int64_t r1 = prod.val[1] - m1;

    // Reduzir para 32 bits
    uint32_t res0 = (uint32_t)(r0 & 0xFFFFFFFF);
    uint32_t res1 = (uint32_t)(r1 & 0xFFFFFFFF);

    // Combinar os resultados de volta para um vetor de 4 elementos
    int32x2_t reduced_low  = vdup_lane_s32((int32_t)res0, vreinterpret_s32_u32(vmovn_u64(vcombine_u64(vmovn_u64(vreinterpret_u64_s64(vcreate_s64(res0))),
                                                                                                     vmovn_u64(vreinterpret_u64_s64(vcreate_s64(res1))))))), 0);
    int32x2_t reduced_high = vdup_n_s32(0); // Placeholder: Nenhum dado adicional

    int32x4_t result = vcombine_s32(reduced_low, reduced_high);

    return result;
}

// Implementação da NTT radix-4 utilizando NEON
void ntt_radix4_neon(int32_t a[N], const int32_t zetas_radix4[768]) {
    unsigned int stage, group, butterfly;
    int32x4_t a0_vec, a1_vec, a2_vec, a3_vec;
    int32x4_t t0_vec, t1_vec, t2_vec, t3_vec;
    int32x4_t zeta1_vec, zeta2_vec, zeta3_vec;
    int32x4_t t0_twiddled_vec, t1_twiddled_vec, t2_twiddled_vec;
    int64x2x2_t prod1, prod2, prod3;

    // Iterar sobre cada estágio da NTT radix-4
    for (stage = 0; stage < 4; stage++) {  // 4 estágios para radix-4
        unsigned int len = 1 << (stage * 2);     // len = 1, 4, 16, 64
        unsigned int step = len << 2;           // step = 4, 16, 64, 256

        // Iterar sobre cada grupo dentro do estágio
        for (group = 0; group < N; group += step) {
            // Iterar sobre cada butterfly dentro do grupo
            for (butterfly = 0; butterfly < len; butterfly += 4) {  // Processar 4 butterflies de cada vez
                unsigned int index = group + butterfly;

                // Carregar quatro elementos de cada sequência dentro do butterfly
                a0_vec = vld1q_s32(&a[index]);             // a0, a1, a2, a3
                a1_vec = vld1q_s32(&a[index + len]);       // a4, a5, a6, a7
                a2_vec = vld1q_s32(&a[index + 2 * len]);   // a8, a9, a10, a11
                a3_vec = vld1q_s32(&a[index + 3 * len]);   // a12, a13, a14, a15

                // Carregar os três fatores twiddle para os 4 butterflies
                // Cada butterfly requer três zetas
                unsigned int twiddle_base = stage * 192 + butterfly * 3;
                zeta1_vec = vld1q_s32(&zetas_radix4[twiddle_base + 0]);
                zeta2_vec = vld1q_s32(&zetas_radix4[twiddle_base + 1]);
                zeta3_vec = vld1q_s32(&zetas_radix4[twiddle_base + 2]);

                // Executar as operações de butterfly radix-4
                // t0 = a0 + a2
                t0_vec = vaddq_s32(a0_vec, a2_vec);
                // t1 = a1 + a3
                t1_vec = vaddq_s32(a1_vec, a3_vec);
                // t2 = a0 - a2
                t2_vec = vsubq_s32(a0_vec, a2_vec);
                // t3 = a1 - a3
                t3_vec = vsubq_s32(a1_vec, a3_vec);

                // Multiplicação de Montgomery para t3 * zeta1, zeta2, zeta3
                // Produto 1: t3 * zeta1
                prod1.val[0] = vmull_s32(vget_low_s32(t3_vec), vget_low_s32(zeta1_vec));
                prod1.val[1] = vmull_s32(vget_high_s32(t3_vec), vget_high_s32(zeta1_vec));
                t0_twiddled_vec = montgomery_reduce_neon_4(prod1, Q);

                // Produto 2: t3 * zeta2
                prod2.val[0] = vmull_s32(vget_low_s32(t3_vec), vget_low_s32(zeta2_vec));
                prod2.val[1] = vmull_s32(vget_high_s32(t3_vec), vget_high_s32(zeta2_vec));
                t1_twiddled_vec = montgomery_reduce_neon_4(prod2, Q);

                // Produto 3: t3 * zeta3
                prod3.val[0] = vmull_s32(vget_low_s32(t3_vec), vget_low_s32(zeta3_vec));
                prod3.val[1] = vmull_s32(vget_high_s32(t3_vec), vget_high_s32(zeta3_vec));
                t2_twiddled_vec = montgomery_reduce_neon_4(prod3, Q);

                // Combinações finais
                // a0' = t0 + t1
                int32x4_t a0_prime = vaddq_s32(t0_vec, t1_vec);
                // a1' = t2 + t0_twiddled
                int32x4_t a1_prime = vaddq_s32(t2_vec, t0_twiddled_vec);
                // a2' = t0 - t1
                int32x4_t a2_prime = vsubq_s32(t0_vec, t1_vec);
                // a3' = t2 - t0_twiddled
                int32x4_t a3_prime = vsubq_s32(t2_vec, t0_twiddled_vec);

                // Armazenar os resultados de volta no array 'a'
                vst1q_s32(&a[index], a0_prime);
                vst1q_s32(&a[index + len], a1_prime);
                vst1q_s32(&a[index + 2 * len], a2_prime);
                vst1q_s32(&a[index + 3 * len], a3_prime);
            }
        }
    }
}
