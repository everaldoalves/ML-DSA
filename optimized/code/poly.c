#include <stdint.h>
#include "params.h"
#include "poly.h"
#include "ntt.h"
#include "reduce.h"
#include "rounding.h"
#include "symmetric.h"
#include <arm_neon.h>
#include "fips202.h"
#include "fips202x2.h"
#include <string.h>

#ifdef DBENCH
#include "test/cpucycles.h"
extern const uint64_t timing_overhead;
extern uint64_t *tred, *tadd, *tmul, *tround, *tsample, *tpack;
#define DBENCH_START() uint64_t time = cpucycles()
#define DBENCH_STOP(t) t += cpucycles() - time - timing_overhead
#else
#define DBENCH_START()
#define DBENCH_STOP(t)
#endif

/*************************************************
* Name:        poly_reduce
*
* Description: Inplace reduction of all coefficients of polynomial to
*              representative in [-6283008,6283008].
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
// Esta versão processa 16 coeficientes por vez. Está funcionando corretamente.
void poly_reduce(poly *a) {
    int32x4x2_t q_vec = {vdupq_n_s32(Q), vdupq_n_s32(Q)};
    int32x4x2_t shift_vec = {vdupq_n_s32(1 << 22), vdupq_n_s32(1 << 22)};
    DBENCH_START();

    for (unsigned int i = 0; i < N; i += 16) {
        // Carregar 16 coeficientes de uma vez (8 + 8)
        int32x4x2_t a_vec1 = vld1q_s32_x2(&a->coeffs[i]);
        int32x4x2_t a_vec2 = vld1q_s32_x2(&a->coeffs[i + 8]);

        // Calcular t = (a + (1 << 22)) >> 23 para ambos os vetores
        int32x4_t t_vec1_1 = vshrq_n_s32(vaddq_s32(a_vec1.val[0], shift_vec.val[0]), 23);
        int32x4_t t_vec1_2 = vshrq_n_s32(vaddq_s32(a_vec1.val[1], shift_vec.val[1]), 23);
        int32x4_t t_vec2_1 = vshrq_n_s32(vaddq_s32(a_vec2.val[0], shift_vec.val[0]), 23);
        int32x4_t t_vec2_2 = vshrq_n_s32(vaddq_s32(a_vec2.val[1], shift_vec.val[1]), 23);

        // Calcular a - t * Q para ambos os vetores
        int32x4_t result1_1 = vsubq_s32(a_vec1.val[0], vmulq_s32(t_vec1_1, q_vec.val[0]));
        int32x4_t result1_2 = vsubq_s32(a_vec1.val[1], vmulq_s32(t_vec1_2, q_vec.val[1]));
        int32x4_t result2_1 = vsubq_s32(a_vec2.val[0], vmulq_s32(t_vec2_1, q_vec.val[0]));
        int32x4_t result2_2 = vsubq_s32(a_vec2.val[1], vmulq_s32(t_vec2_2, q_vec.val[1]));

        // Armazenar resultados (16 coeficientes)
        vst1q_s32(&a->coeffs[i], result1_1);
        vst1q_s32(&a->coeffs[i + 4], result1_2);
        vst1q_s32(&a->coeffs[i + 8], result2_1);
        vst1q_s32(&a->coeffs[i + 12], result2_2);
    }
    DBENCH_STOP(*tred);
}

/*************************************************
* Name:        poly_caddq
*
* Description: For all coefficients of in/out polynomial add Q if
*              coefficient is negative.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_caddq(poly *a) {
    unsigned int i;
    DBENCH_START();
    int32x4_t q_vec = vdupq_n_s32(Q);  // Vetor contendo o valor de Q

    for (i = 0; i < N; i += 4) {
        int32x4_t a_vec = vld1q_s32(&a->coeffs[i]);  // Carregar 4 coeficientes
        int32x4_t mask = vshrq_n_s32(a_vec, 31);     // Shift à direita para verificar negativos
        int32x4_t add_q = vandq_s32(mask, q_vec);    // Aplicar Q onde for negativo
        a_vec = vaddq_s32(a_vec, add_q);             // Somar Q aos negativos

        vst1q_s32(&a->coeffs[i], a_vec);             // Armazenar os resultados de volta
    }
    DBENCH_STOP(*tred);
}

/*************************************************
* Name:        poly_add
*
* Description: Add polynomials. No modular reduction is performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first summand
*              - const poly *b: pointer to second summand
**************************************************/
void poly_add(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    DBENCH_START();

    // Desenrolando o loop para processar 8 elementos por iteração
    for (i = 0; i < N; i += 8) {
        // Primeiros 4 coeficientes
        int32x4_t a_vec1 = vld1q_s32(&a->coeffs[i]);
        int32x4_t b_vec1 = vld1q_s32(&b->coeffs[i]);
        int32x4_t c_vec1 = vaddq_s32(a_vec1, b_vec1);
        vst1q_s32(&c->coeffs[i], c_vec1);

        // Próximos 4 coeficientes
        int32x4_t a_vec2 = vld1q_s32(&a->coeffs[i+4]);
        int32x4_t b_vec2 = vld1q_s32(&b->coeffs[i+4]);
        int32x4_t c_vec2 = vaddq_s32(a_vec2, b_vec2);
        vst1q_s32(&c->coeffs[i+4], c_vec2);
    }

    DBENCH_STOP(*tadd);
}
/*************************************************
* Name:        poly_sub
*
* Description: Subtract polynomials. No modular reduction is
*              performed.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial to be
*                               subtraced from first input polynomial
**************************************************/
void poly_sub(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    DBENCH_START();
    for (i = 0; i < N; i += 4) {
        int32x4_t a_vec = vld1q_s32(&a->coeffs[i]);  // Carrega 4 coeficientes de a
        int32x4_t b_vec = vld1q_s32(&b->coeffs[i]);  // Carrega 4 coeficientes de b

        int32x4_t c_vec = vsubq_s32(a_vec, b_vec);   // Subtração vetorial dos coeficientes

        vst1q_s32(&c->coeffs[i], c_vec);             // Armazena o resultado em c
    }
    DBENCH_STOP(*tadd);
}

/*************************************************
* Name:        poly_shiftl
*
* Description: Multiply polynomial by 2^D without modular reduction. Assumes
*              input coefficients to be less than 2^{31-D} in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_shiftl(poly *a) {
    unsigned int i;
    DBENCH_START();
    
    // Processar 4 coeficientes por vez
    for (i = 0; i < N; i += 4) {
        // Carregar 4 coeficientes em um registrador NEON
        int32x4_t vec_a = vld1q_s32(&a->coeffs[i]);

        // Aplicar o deslocamento lógico à esquerda de D posições
        vec_a = vshlq_n_s32(vec_a, D);

        // Armazenar o resultado de volta
        vst1q_s32(&a->coeffs[i], vec_a);
    }
    DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_ntt
*
* Description: Inplace forward NTT. Coefficients can grow by
*              8*Q in absolute value.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_ntt(poly *a) {
  DBENCH_START();

  ntt(a->coeffs);

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_invntt_tomont
*
* Description: Inplace inverse NTT and multiplication by 2^{32}.
*              Input coefficients need to be less than Q in absolute
*              value and output coefficients are again bounded by Q.
*
* Arguments:   - poly *a: pointer to input/output polynomial
**************************************************/
void poly_invntt_tomont(poly *a) {
  DBENCH_START();

  invntt_tomont(a->coeffs);

  DBENCH_STOP(*tmul);
}

/*************************************************
* Name:        poly_pointwise_montgomery
*
* Description: Pointwise multiplication of polynomials in NTT domain
*              representation and multiplication of resulting polynomial
*              by 2^{-32}.
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const poly *a: pointer to first input polynomial
*              - const poly *b: pointer to second input polynomial
**************************************************/
// Função otimizada para multiplicação de polinômios utilizando NEON para processar 8 coeficientes de uma vez
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    DBENCH_START();
    for(i = 0; i < N; i += 8) {
        // Carregar 8 coeficientes de cada polinômio a e b
        int32x4x2_t a_vec = vld1q_s32_x2(&a->coeffs[i]);  // Carrega 8 coeficientes de 'a'
        int32x4x2_t b_vec = vld1q_s32_x2(&b->coeffs[i]);  // Carrega 8 coeficientes de 'b'

        // Multiplicar os 8 coeficientes
        int64x2x2_t product_vec1, product_vec2;
        product_vec1.val[0] = vmull_s32(vget_low_s32(a_vec.val[0]), vget_low_s32(b_vec.val[0]));
        product_vec1.val[1] = vmull_s32(vget_high_s32(a_vec.val[0]), vget_high_s32(b_vec.val[0]));
        
        product_vec2.val[0] = vmull_s32(vget_low_s32(a_vec.val[1]), vget_low_s32(b_vec.val[1]));
        product_vec2.val[1] = vmull_s32(vget_high_s32(a_vec.val[1]), vget_high_s32(b_vec.val[1]));

        // Reduzir os 8 coeficientes usando Montgomery reduction
        int32x4_t result1 = montgomery_reduce_neon_4(product_vec1);
        int32x4_t result2 = montgomery_reduce_neon_4(product_vec2);

        // Armazenar os resultados de volta em c->coeffs
        vst1q_s32(&c->coeffs[i], result1);     // Armazena os primeiros 4 coeficientes
        vst1q_s32(&c->coeffs[i + 4], result2); // Armazena os próximos 4 coeficientes
    }
     DBENCH_STOP(*tmul);
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
  DBENCH_START();

  for(i = 0; i < N; ++i)
    a1->coeffs[i] = power2round(&a0->coeffs[i], a->coeffs[i]);

  DBENCH_STOP(*tround);
}

/*************************************************
* Name:        poly_decompose
*
* Description: For all coefficients c of the input polynomial,
*              compute high and low bits c0, c1 such c mod Q = c1*ALPHA + c0
*              with -ALPHA/2 < c0 <= ALPHA/2 except c1 = (Q-1)/ALPHA where we
*              set c1 = 0 and -ALPHA/2 <= c0 = c mod Q - Q < 0.
*              Assumes coefficients to be standard representatives.
*
* Arguments:   - poly *a1: pointer to output polynomial with coefficients c1
*              - poly *a0: pointer to output polynomial with coefficients c0
*              - const poly *a: pointer to input polynomial
**************************************************/
void poly_decompose(poly *a1, poly *a0, const poly *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N; ++i)
    a1->coeffs[i] = decompose(&a0->coeffs[i], a->coeffs[i]);

  DBENCH_STOP(*tround);
}

/*************************************************
* Name:        poly_make_hint
*
* Description: Compute hint polynomial. The coefficients of which indicate
*              whether the low bits of the corresponding coefficient of
*              the input polynomial overflow into the high bits.
*
* Arguments:   - poly *h: pointer to output hint polynomial
*              - const poly *a0: pointer to low part of input polynomial
*              - const poly *a1: pointer to high part of input polynomial
*
* Returns number of 1 bits.
**************************************************/
unsigned int poly_make_hint(poly *h, const poly *a0, const poly *a1) {
  unsigned int i, s = 0;
  DBENCH_START();

  for(i = 0; i < N; ++i) {
    h->coeffs[i] = make_hint(a0->coeffs[i], a1->coeffs[i]);
    s += h->coeffs[i];
  }

  DBENCH_STOP(*tround);
  return s;
}

/*************************************************
* Name:        poly_use_hint
*
* Description: Use hint polynomial to correct the high bits of a polynomial.
*
* Arguments:   - poly *b: pointer to output polynomial with corrected high bits
*              - const poly *a: pointer to input polynomial
*              - const poly *h: pointer to input hint polynomial
**************************************************/
void poly_use_hint(poly *b, const poly *a, const poly *h) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N; ++i)
    b->coeffs[i] = use_hint(a->coeffs[i], h->coeffs[i]);

  DBENCH_STOP(*tround);
}

/*************************************************
* Name:        poly_chknorm
*
* Description: Check infinity norm of polynomial against given bound.
*              Assumes input coefficients were reduced by reduce32().
*
* Arguments:   - const poly *a: pointer to polynomial
*              - int32_t B: norm bound
*
* Returns 0 if norm is strictly smaller than B <= (Q-1)/8 and 1 otherwise.
**************************************************/
int poly_chknorm(const poly *a, int32_t B) {
    unsigned int i;
    int32x4_t B_vec = vdupq_n_s32(B); // Vetor com valor B replicado
    int32x4_t abs_vec, mask_vec;

    if (B > (Q - 1) / 8)
        return 1;

    for (i = 0; i < N; i += 4) {
        // Carrega 4 coeficientes
        int32x4_t coeff_vec = vld1q_s32(&a->coeffs[i]);

        // Calcula o valor absoluto
        mask_vec = vshrq_n_s32(coeff_vec, 31);      // t = a[i] >> 31
        abs_vec = vsubq_s32(coeff_vec, vandq_s32(mask_vec, vaddq_s32(coeff_vec, coeff_vec)));  // t = a - (t & 2*a)

        // Verifica se algum coeficiente é maior ou igual a B
        uint32x4_t cmp_result = vcgeq_s32(abs_vec, B_vec);  // Compara abs_vec >= B
        if (vmaxvq_u32(cmp_result) != 0) { // vmaxvq_u32 retorna o valor máximo no vetor
            return 1;
        }
    }

    return 0;
}

/*************************************************
* Name:        rej_uniform
*
* Description: Sample uniformly random coefficients in [0, Q-1] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
// Versão desenvolvida para funcionar em consonância à poly_uniform_2x
static unsigned int rej_uniform(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0, pos = 0;
    const uint32_t mask = 0x7FFFFF; // Máscara para 23 bits

    uint32x4_t q_vec = vdupq_n_u32(Q);

    while (ctr < len && pos + 12 <= buflen) {
        // Carregar 12 bytes do buffer
        uint8x8x3_t bytes;
        bytes.val[0] = vld1_u8(&buf[pos]);       // Carrega bytes 0-7
        bytes.val[1] = vld1_u8(&buf[pos + 4]);   // Carrega bytes 4-11
        bytes.val[2] = vld1_u8(&buf[pos + 8]);   // Carrega bytes 8-15
        pos += 12;

        // Montar 4 valores de 24 bits
        uint32x4_t t_vec;
        t_vec = vmovl_u16(vget_low_u16(vmovl_u8(bytes.val[0]))); // bytes 0-3
        t_vec = vorrq_u32(t_vec, vshlq_n_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes.val[1]))), 8));
        t_vec = vorrq_u32(t_vec, vshlq_n_u32(vmovl_u16(vget_low_u16(vmovl_u8(bytes.val[2]))), 16));
        t_vec = vandq_u32(t_vec, vdupq_n_u32(mask));

        // Comparar com Q
        uint32x4_t cmp = vcltq_u32(t_vec, q_vec);

        // Desloca cada elemento 31 bits para a direita para obter 0x1 ou 0x0
        uint32x4_t cmp_shr = vshrq_n_u32(cmp, 31);

        // Extrai os bits e constrói a máscara
        uint16_t mask_bits = (vgetq_lane_u32(cmp_shr, 0) & 1) << 0;
        mask_bits |= (vgetq_lane_u32(cmp_shr, 1) & 1) << 1;
        mask_bits |= (vgetq_lane_u32(cmp_shr, 2) & 1) << 2;
        mask_bits |= (vgetq_lane_u32(cmp_shr, 3) & 1) << 3;

        // Extrair os valores válidos
        uint32_t temp_vals[4];
        vst1q_u32(temp_vals, t_vec);

        for (int i = 0; i < 4 && ctr < len; ++i) {
            if (mask_bits & (1 << i)) {
                a[ctr++] = temp_vals[i];
            }
        }
    }

    // Processamento escalar dos bytes restantes
    uint32_t t;
    while (ctr < len && pos + 3 <= buflen) {
        t  = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= mask;

        if (t < Q)
            a[ctr++] = t;
    }

    return ctr;
}

/*************************************************
* Name:        poly_uniform
*
* Description: Sample polynomial with uniformly random coefficients
*              in [0,Q-1] by performing rejection sampling on the
*              output stream of SHAKE128(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length SEEDBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)

// Esta versão é otimizada para processar múltiplos polinômios de uma vez usando batch_size
void poly_uniform(poly *a[], const uint8_t seed[SEEDBYTES], uint16_t nonce[], int batch_size) {
    // 'a' é um array de ponteiros para polinômios
    // 'seed' é a semente
    // 'nonce' é um array de nonces
    // 'batch_size' é o número de polinômios a serem processados

    // Inicializa estados de stream para cada polinômio
    stream128_state state[batch_size];

    for (int idx = 0; idx < batch_size; ++idx) {
        stream128_init(&state[idx], seed, nonce[idx]);
    }

    // Define buffers para cada polinômio
    uint8_t buf[batch_size][POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;

    // Squeeze blocos para cada stream
    for (int idx = 0; idx < batch_size; ++idx) {
        stream128_squeezeblocks(buf[idx], POLY_UNIFORM_NBLOCKS, &state[idx]);
    }

    // Inicializa contadores para cada polinômio
    unsigned int ctr[batch_size];
    for (int idx = 0; idx < batch_size; ++idx) {
        ctr[idx] = 0;
    }

    // Processa a rejeição de amostras
    int completed;
    do {
        completed = 1;
        for (int idx = 0; idx < batch_size; ++idx) {
            if (ctr[idx] < N) {
                completed = 0;

                unsigned int buflen_idx = buflen;

                // Executa a rejeição utilizando a versão otimizada
                unsigned int ncoeffs = N - ctr[idx];
                unsigned int n = rej_uniform(a[idx]->coeffs + ctr[idx], ncoeffs, buf[idx], buflen_idx);
                ctr[idx] += n;

                // Se não preencheu todos os coeficientes, squeeze mais blocos
                if (ctr[idx] < N) {
                    unsigned int off = buflen_idx % 3;
                    // Ajusta o buffer
                    for (unsigned int i = 0; i < off; ++i) {
                        buf[idx][i] = buf[idx][buflen_idx - off + i];
                    }

                    // Pré-carregamento do buffer
                    __builtin_prefetch(&buf[idx], 0, 3);

                    // Squeeze mais blocos do stream
                    stream128_squeezeblocks(buf[idx] + off, 1, &state[idx]);
                    buflen_idx = STREAM128_BLOCKBYTES + off;
                    buflen = buflen_idx; // Atualiza o buflen para o próximo loop
                }
            }
        }
    } while (!completed);
}



/******************************************************************************
 * Name:        poly_uniform_2x
 *  
 * Description: Sample two polynomials with uniformly random coefficients
 *             in [0,Q-1] by performing rejection sampling on the
 *            output stream of SHAKE128(seed|nonce0, nonce1)
 *      
 * Arguments:   - poly *a0: pointer to output polynomial
 *             - poly *a1: pointer to output polynomial
 *            - const uint8_t seed[]: byte array with seed of length SEEDBYTES
 *           - uint16_t nonce0: 2-byte nonce
 *         - uint16_t nonce1: 2-byte nonce
 * *******************************************************************************/

void poly_uniform_2x(poly *a0, poly *a1, const uint8_t seed[SEEDBYTES], uint16_t nonce0, uint16_t nonce1) {
    unsigned int ctr0 = 0, ctr1 = 0;
    uint8_t buf0[SEEDBYTES + 2];
    uint8_t buf1[SEEDBYTES + 2];
    keccakx2_state state;

    // Preparação dos Buffers
    memcpy(buf0, seed, SEEDBYTES);
    memcpy(buf1, seed, SEEDBYTES);

    buf0[SEEDBYTES + 0] = (uint8_t)(nonce0 & 0xFF);
    buf0[SEEDBYTES + 1] = (uint8_t)(nonce0 >> 8);
    buf1[SEEDBYTES + 0] = (uint8_t)(nonce1 & 0xFF);
    buf1[SEEDBYTES + 1] = (uint8_t)(nonce1 >> 8);

    // Absorção com SHAKE128x2
    FIPS202X2_NAMESPACE(shake128x2_absorb_once)(&state, buf0, buf1, SEEDBYTES + 2);

    // Squeeze Inicial
    static uint8_t global_outbuf0[REJ_UNIFORM_BUFLEN];
    static uint8_t global_outbuf1[REJ_UNIFORM_BUFLEN];
    FIPS202X2_NAMESPACE(shake128x2_squeezeblocks)(global_outbuf0, global_outbuf1, REJ_UNIFORM_NBLOCKS, &state);

    // Rejeição Uniforme Otimizada
    ctr0 = rej_uniform(a0->coeffs, N, global_outbuf0, REJ_UNIFORM_BUFLEN);
    ctr1 = rej_uniform(a1->coeffs, N, global_outbuf1, REJ_UNIFORM_BUFLEN);

    // Loop de Rejeição Adicional
    while (ctr0 < N || ctr1 < N) {
        FIPS202X2_NAMESPACE(shake128x2_squeezeblocks)(global_outbuf0, global_outbuf1, 1, &state);

        ctr0 += rej_uniform(a0->coeffs + ctr0, N - ctr0, global_outbuf0, SHAKE128_RATE);
        ctr1 += rej_uniform(a1->coeffs + ctr1, N - ctr1, global_outbuf1, SHAKE128_RATE);
    }
}



/*************************************************
* Name:        rej_eta
*
* Description: Sample uniformly random coefficients in [-ETA, ETA] by
*              performing rejection sampling on array of random bytes.
*
* Arguments:   - int32_t *a: pointer to output array (allocated)
*              - unsigned int len: number of coefficients to be sampled
*              - const uint8_t *buf: array of random bytes
*              - unsigned int buflen: length of array of random bytes
*
* Returns number of sampled coefficients. Can be smaller than len if not enough
* random bytes were given.
**************************************************/
// Tabela de lookup somente para ETA = 4
#if ETA == 4
static const int8_t eta4_lookup[16] = {4, 3, 2, 1, 0, -1, -2, -3, -4, -1, -1, -1, -1, -1, -1, -1};
#endif

static unsigned int rej_eta(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
  unsigned int ctr = 0, pos = 0;

#if ETA == 2
  // Para ETA = 2, o ajuste precisa ser feito em todos os valores
  while (ctr < len && pos < buflen) {
    uint32_t t0 = buf[pos] & 0x0F;
    uint32_t t1 = buf[pos++] >> 4;

    if (t0 < 15) {
      t0 = t0 - (205 * t0 >> 10) * 5;
      a[ctr++] = 2 - t0;
    }

    if (t1 < 15 && ctr < len) {
      t1 = t1 - (205 * t1 >> 10) * 5;
      a[ctr++] = 2 - t1;
    }
  }
#elif ETA == 4
  // Para ETA = 4, podemos utilizar a tabela de lookup e processar em blocos
  uint8x16_t vec, low_mask = vdupq_n_u8(0x0F); // Mask para os nibbles baixos

  while (ctr < len && pos + 16 <= buflen) {
    // Carregar 16 bytes do buffer
    vec = vld1q_u8(&buf[pos]);
    pos += 16;

    // Extrair os nibbles baixos e altos
    uint8x16_t t0 = vandq_u8(vec, low_mask);  // Nibbles baixos
    uint8x16_t t1 = vshrq_n_u8(vec, 4);       // Nibbles altos

    // Aplicar a tabela de lookup para os nibbles
    int8x16_t t0_mapped = vqtbl1q_s8(vld1q_s8(eta4_lookup), t0);
    int8x16_t t1_mapped = vqtbl1q_s8(vld1q_s8(eta4_lookup), t1);

    // Armazenar os resultados
    vst1q_s32(&a[ctr], vreinterpretq_s32_s8(t0_mapped));
    ctr += 8;  // Cada bloco t0_mapped armazena 8 valores
    vst1q_s32(&a[ctr], vreinterpretq_s32_s8(t1_mapped));
    ctr += 8;  // Cada bloco t1_mapped armazena 8 valores
  }

  // Processar quaisquer bytes restantes
  while (ctr < len && pos < buflen) {
    uint32_t t0 = buf[pos] & 0x0F;
    uint32_t t1 = buf[pos++] >> 4;

    if (t0 < 9)
      a[ctr++] = eta4_lookup[t0];
    if (t1 < 9 && ctr < len)
      a[ctr++] = eta4_lookup[t1];
  }
#endif

  return ctr;
}

/*************************************************
* Name:        poly_uniform_eta
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-ETA,ETA] by performing rejection sampling on the
*              output stream from SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 2-byte nonce
**************************************************/
#if ETA == 2
#define POLY_UNIFORM_ETA_NBLOCKS ((136 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#elif ETA == 4
#define POLY_UNIFORM_ETA_NBLOCKS ((227 + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
#endif
void poly_uniform_eta(poly *a,
                      const uint8_t seed[CRHBYTES],
                      uint16_t nonce)
{
  unsigned int ctr;
  unsigned int buflen = POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_ETA_NBLOCKS*STREAM256_BLOCKBYTES];
  stream256_state state;

  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_ETA_NBLOCKS, &state);

  ctr = rej_eta(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    stream256_squeezeblocks(buf, 1, &state);
    ctr += rej_eta(a->coeffs + ctr, N - ctr, buf, STREAM256_BLOCKBYTES);
  }
}

/*************************************************
* Name:        poly_uniform_gamma1m1
*
* Description: Sample polynomial with uniformly random coefficients
*              in [-(GAMMA1 - 1), GAMMA1] by unpacking output stream
*              of SHAKE256(seed|nonce)
*
* Arguments:   - poly *a: pointer to output polynomial
*              - const uint8_t seed[]: byte array with seed of length CRHBYTES
*              - uint16_t nonce: 16-bit nonce
**************************************************/
#define POLY_UNIFORM_GAMMA1_NBLOCKS ((POLYZ_PACKEDBYTES + STREAM256_BLOCKBYTES - 1)/STREAM256_BLOCKBYTES)
void poly_uniform_gamma1(poly *a,
                         const uint8_t seed[CRHBYTES],
                         uint16_t nonce)
{
  uint8_t buf[POLY_UNIFORM_GAMMA1_NBLOCKS*STREAM256_BLOCKBYTES];
  stream256_state state;

  stream256_init(&state, seed, nonce);
  stream256_squeezeblocks(buf, POLY_UNIFORM_GAMMA1_NBLOCKS, &state);
  polyz_unpack(a, buf);
}

void poly_uniform_gamma1_2x(poly *a0, poly *a1, const uint8_t seed[64], 
                            uint16_t nonce0, uint16_t nonce1) {
  uint8_t buf[2][POLY_UNIFORM_GAMMA1_NBLOCKS * STREAM256_BLOCKBYTES + 14];
  uint64x2_t f0, f1;
  keccakx2_state state;

  // Carregar os primeiros 32 bytes do seed em registradores NEON
  f0 = vld1q_u64((const uint64_t *)&seed[0]);
  vst1q_u64((uint64_t *)&buf[0][0], f0);
  vst1q_u64((uint64_t *)&buf[1][0], f0);

  // Carregar os segundos 32 bytes do seed em registradores NEON
  f1 = vld1q_u64((const uint64_t *)&seed[32]);
  vst1q_u64((uint64_t *)&buf[0][16], f1);
  vst1q_u64((uint64_t *)&buf[1][16], f1);

  // Definir os nonces nos buffers
  buf[0][64] = nonce0 & 0xFF;
  buf[0][65] = (nonce0 >> 8) & 0xFF;
  buf[1][64] = nonce1 & 0xFF;
  buf[1][65] = (nonce1 >> 8) & 0xFF;

  // Absorver os dados para os 2 polinômios simultaneamente
  FIPS202X2_NAMESPACE(shake256x2_absorb)(&state, buf[0], buf[1], 66);

  // Realizar squeezeblocks para obter os coeficientes
  FIPS202X2_NAMESPACE(shake256x2_squeezeblocks)(buf[0], buf[1], POLY_UNIFORM_GAMMA1_NBLOCKS, &state);

  // Descompactar os coeficientes em polinômios
  polyz_unpack(a0, buf[0]);
  polyz_unpack(a1, buf[1]);
}

/*************************************************
* Name:        challenge
*
* Description: Implementation of H. Samples polynomial with TAU nonzero
*              coefficients in {-1,1} using the output stream of
*              SHAKE256(seed).
*
* Arguments:   - poly *c: pointer to output polynomial
*              - const uint8_t mu[]: byte array containing seed of length CTILDEBYTES
**************************************************/
void poly_challenge(poly *c, const uint8_t seed[CTILDEBYTES]) {
  unsigned int i, b, pos;
  uint64_t signs;
  uint8_t buf[SHAKE256_RATE];
  keccak_state state;

  shake256_init(&state);
  shake256_absorb(&state, seed, CTILDEBYTES);
  shake256_finalize(&state);
  shake256_squeezeblocks(buf, 1, &state);

  signs = 0;
  for(i = 0; i < 8; ++i)
    signs |= (uint64_t)buf[i] << 8*i;
  pos = 8;

  for(i = 0; i < N; ++i)
    c->coeffs[i] = 0;
  for(i = N-TAU; i < N; ++i) {
    do {
      if(pos >= SHAKE256_RATE) {
        shake256_squeezeblocks(buf, 1, &state);
        pos = 0;
      }

      b = buf[pos++];
    } while(b > i);

    c->coeffs[i] = c->coeffs[b];
    c->coeffs[b] = 1 - 2*(signs & 1);
    signs >>= 1;
  }
}

/*************************************************
* Name:        polyeta_pack
*
* Description: Bit-pack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYETA_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyeta_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint8_t t[8];
  DBENCH_START();

#if ETA == 2
  for(i = 0; i < N/8; ++i) {
    t[0] = ETA - a->coeffs[8*i+0];
    t[1] = ETA - a->coeffs[8*i+1];
    t[2] = ETA - a->coeffs[8*i+2];
    t[3] = ETA - a->coeffs[8*i+3];
    t[4] = ETA - a->coeffs[8*i+4];
    t[5] = ETA - a->coeffs[8*i+5];
    t[6] = ETA - a->coeffs[8*i+6];
    t[7] = ETA - a->coeffs[8*i+7];

    r[3*i+0]  = (t[0] >> 0) | (t[1] << 3) | (t[2] << 6);
    r[3*i+1]  = (t[2] >> 2) | (t[3] << 1) | (t[4] << 4) | (t[5] << 7);
    r[3*i+2]  = (t[5] >> 1) | (t[6] << 2) | (t[7] << 5);
  }
#elif ETA == 4
  for(i = 0; i < N/2; ++i) {
    t[0] = ETA - a->coeffs[2*i+0];
    t[1] = ETA - a->coeffs[2*i+1];
    r[i] = t[0] | (t[1] << 4);
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyeta_unpack
*
* Description: Unpack polynomial with coefficients in [-ETA,ETA].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyeta_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

#if ETA == 2
  for(i = 0; i < N/8; ++i) {
    r->coeffs[8*i+0] =  (a[3*i+0] >> 0) & 7;
    r->coeffs[8*i+1] =  (a[3*i+0] >> 3) & 7;
    r->coeffs[8*i+2] = ((a[3*i+0] >> 6) | (a[3*i+1] << 2)) & 7;
    r->coeffs[8*i+3] =  (a[3*i+1] >> 1) & 7;
    r->coeffs[8*i+4] =  (a[3*i+1] >> 4) & 7;
    r->coeffs[8*i+5] = ((a[3*i+1] >> 7) | (a[3*i+2] << 1)) & 7;
    r->coeffs[8*i+6] =  (a[3*i+2] >> 2) & 7;
    r->coeffs[8*i+7] =  (a[3*i+2] >> 5) & 7;

    r->coeffs[8*i+0] = ETA - r->coeffs[8*i+0];
    r->coeffs[8*i+1] = ETA - r->coeffs[8*i+1];
    r->coeffs[8*i+2] = ETA - r->coeffs[8*i+2];
    r->coeffs[8*i+3] = ETA - r->coeffs[8*i+3];
    r->coeffs[8*i+4] = ETA - r->coeffs[8*i+4];
    r->coeffs[8*i+5] = ETA - r->coeffs[8*i+5];
    r->coeffs[8*i+6] = ETA - r->coeffs[8*i+6];
    r->coeffs[8*i+7] = ETA - r->coeffs[8*i+7];
  }
#elif ETA == 4
  for(i = 0; i < N/2; ++i) {
    r->coeffs[2*i+0] = a[i] & 0x0F;
    r->coeffs[2*i+1] = a[i] >> 4;
    r->coeffs[2*i+0] = ETA - r->coeffs[2*i+0];
    r->coeffs[2*i+1] = ETA - r->coeffs[2*i+1];
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt1_pack
*
* Description: Bit-pack polynomial t1 with coefficients fitting in 10 bits.
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt1_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/4; ++i) {
    r[5*i+0] = (a->coeffs[4*i+0] >> 0);
    r[5*i+1] = (a->coeffs[4*i+0] >> 8) | (a->coeffs[4*i+1] << 2);
    r[5*i+2] = (a->coeffs[4*i+1] >> 6) | (a->coeffs[4*i+2] << 4);
    r[5*i+3] = (a->coeffs[4*i+2] >> 4) | (a->coeffs[4*i+3] << 6);
    r[5*i+4] = (a->coeffs[4*i+3] >> 2);
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt1_unpack
*
* Description: Unpack polynomial t1 with 10-bit coefficients.
*              Output coefficients are standard representatives.
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyt1_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/4; ++i) {
    r->coeffs[4*i+0] = ((a[5*i+0] >> 0) | ((uint32_t)a[5*i+1] << 8)) & 0x3FF;
    r->coeffs[4*i+1] = ((a[5*i+1] >> 2) | ((uint32_t)a[5*i+2] << 6)) & 0x3FF;
    r->coeffs[4*i+2] = ((a[5*i+2] >> 4) | ((uint32_t)a[5*i+3] << 4)) & 0x3FF;
    r->coeffs[4*i+3] = ((a[5*i+3] >> 6) | ((uint32_t)a[5*i+4] << 2)) & 0x3FF;
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt0_pack
*
* Description: Bit-pack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYT0_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyt0_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint32_t t[8];
  DBENCH_START();

  for(i = 0; i < N/8; ++i) {
    t[0] = (1 << (D-1)) - a->coeffs[8*i+0];
    t[1] = (1 << (D-1)) - a->coeffs[8*i+1];
    t[2] = (1 << (D-1)) - a->coeffs[8*i+2];
    t[3] = (1 << (D-1)) - a->coeffs[8*i+3];
    t[4] = (1 << (D-1)) - a->coeffs[8*i+4];
    t[5] = (1 << (D-1)) - a->coeffs[8*i+5];
    t[6] = (1 << (D-1)) - a->coeffs[8*i+6];
    t[7] = (1 << (D-1)) - a->coeffs[8*i+7];

    r[13*i+ 0]  =  t[0];
    r[13*i+ 1]  =  t[0] >>  8;
    r[13*i+ 1] |=  t[1] <<  5;
    r[13*i+ 2]  =  t[1] >>  3;
    r[13*i+ 3]  =  t[1] >> 11;
    r[13*i+ 3] |=  t[2] <<  2;
    r[13*i+ 4]  =  t[2] >>  6;
    r[13*i+ 4] |=  t[3] <<  7;
    r[13*i+ 5]  =  t[3] >>  1;
    r[13*i+ 6]  =  t[3] >>  9;
    r[13*i+ 6] |=  t[4] <<  4;
    r[13*i+ 7]  =  t[4] >>  4;
    r[13*i+ 8]  =  t[4] >> 12;
    r[13*i+ 8] |=  t[5] <<  1;
    r[13*i+ 9]  =  t[5] >>  7;
    r[13*i+ 9] |=  t[6] <<  6;
    r[13*i+10]  =  t[6] >>  2;
    r[13*i+11]  =  t[6] >> 10;
    r[13*i+11] |=  t[7] <<  3;
    r[13*i+12]  =  t[7] >>  5;
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyt0_unpack
*
* Description: Unpack polynomial t0 with coefficients in ]-2^{D-1}, 2^{D-1}].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyt0_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

  for(i = 0; i < N/8; ++i) {
    r->coeffs[8*i+0]  = a[13*i+0];
    r->coeffs[8*i+0] |= (uint32_t)a[13*i+1] << 8;
    r->coeffs[8*i+0] &= 0x1FFF;

    r->coeffs[8*i+1]  = a[13*i+1] >> 5;
    r->coeffs[8*i+1] |= (uint32_t)a[13*i+2] << 3;
    r->coeffs[8*i+1] |= (uint32_t)a[13*i+3] << 11;
    r->coeffs[8*i+1] &= 0x1FFF;

    r->coeffs[8*i+2]  = a[13*i+3] >> 2;
    r->coeffs[8*i+2] |= (uint32_t)a[13*i+4] << 6;
    r->coeffs[8*i+2] &= 0x1FFF;

    r->coeffs[8*i+3]  = a[13*i+4] >> 7;
    r->coeffs[8*i+3] |= (uint32_t)a[13*i+5] << 1;
    r->coeffs[8*i+3] |= (uint32_t)a[13*i+6] << 9;
    r->coeffs[8*i+3] &= 0x1FFF;

    r->coeffs[8*i+4]  = a[13*i+6] >> 4;
    r->coeffs[8*i+4] |= (uint32_t)a[13*i+7] << 4;
    r->coeffs[8*i+4] |= (uint32_t)a[13*i+8] << 12;
    r->coeffs[8*i+4] &= 0x1FFF;

    r->coeffs[8*i+5]  = a[13*i+8] >> 1;
    r->coeffs[8*i+5] |= (uint32_t)a[13*i+9] << 7;
    r->coeffs[8*i+5] &= 0x1FFF;

    r->coeffs[8*i+6]  = a[13*i+9] >> 6;
    r->coeffs[8*i+6] |= (uint32_t)a[13*i+10] << 2;
    r->coeffs[8*i+6] |= (uint32_t)a[13*i+11] << 10;
    r->coeffs[8*i+6] &= 0x1FFF;

    r->coeffs[8*i+7]  = a[13*i+11] >> 3;
    r->coeffs[8*i+7] |= (uint32_t)a[13*i+12] << 5;
    r->coeffs[8*i+7] &= 0x1FFF;

    r->coeffs[8*i+0] = (1 << (D-1)) - r->coeffs[8*i+0];
    r->coeffs[8*i+1] = (1 << (D-1)) - r->coeffs[8*i+1];
    r->coeffs[8*i+2] = (1 << (D-1)) - r->coeffs[8*i+2];
    r->coeffs[8*i+3] = (1 << (D-1)) - r->coeffs[8*i+3];
    r->coeffs[8*i+4] = (1 << (D-1)) - r->coeffs[8*i+4];
    r->coeffs[8*i+5] = (1 << (D-1)) - r->coeffs[8*i+5];
    r->coeffs[8*i+6] = (1 << (D-1)) - r->coeffs[8*i+6];
    r->coeffs[8*i+7] = (1 << (D-1)) - r->coeffs[8*i+7];
  }

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyz_pack
*
* Description: Bit-pack polynomial with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYZ_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyz_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  uint32_t t[4];
  DBENCH_START();

#if GAMMA1 == (1 << 17)
  for(i = 0; i < N/4; ++i) {
    t[0] = GAMMA1 - a->coeffs[4*i+0];
    t[1] = GAMMA1 - a->coeffs[4*i+1];
    t[2] = GAMMA1 - a->coeffs[4*i+2];
    t[3] = GAMMA1 - a->coeffs[4*i+3];

    r[9*i+0]  = t[0];
    r[9*i+1]  = t[0] >> 8;
    r[9*i+2]  = t[0] >> 16;
    r[9*i+2] |= t[1] << 2;
    r[9*i+3]  = t[1] >> 6;
    r[9*i+4]  = t[1] >> 14;
    r[9*i+4] |= t[2] << 4;
    r[9*i+5]  = t[2] >> 4;
    r[9*i+6]  = t[2] >> 12;
    r[9*i+6] |= t[3] << 6;
    r[9*i+7]  = t[3] >> 2;
    r[9*i+8]  = t[3] >> 10;
  }
#elif GAMMA1 == (1 << 19)
  for(i = 0; i < N/2; ++i) {
    t[0] = GAMMA1 - a->coeffs[2*i+0];
    t[1] = GAMMA1 - a->coeffs[2*i+1];

    r[5*i+0]  = t[0];
    r[5*i+1]  = t[0] >> 8;
    r[5*i+2]  = t[0] >> 16;
    r[5*i+2] |= t[1] << 4;
    r[5*i+3]  = t[1] >> 4;
    r[5*i+4]  = t[1] >> 12;
  }
#endif

  DBENCH_STOP(*tpack);
}

/*************************************************
* Name:        polyz_unpack
*
* Description: Unpack polynomial z with coefficients
*              in [-(GAMMA1 - 1), GAMMA1].
*
* Arguments:   - poly *r: pointer to output polynomial
*              - const uint8_t *a: byte array with bit-packed polynomial
**************************************************/
void polyz_unpack(poly *r, const uint8_t *a) {
  unsigned int i;
  DBENCH_START();

#if GAMMA1 == (1 << 17) // Se gamma1 = 2^17
  for(i = 0; i < N/4; ++i) {
    r->coeffs[4*i+0]  = a[9*i+0];
    r->coeffs[4*i+0] |= (uint32_t)a[9*i+1] << 8;
    r->coeffs[4*i+0] |= (uint32_t)a[9*i+2] << 16;
    r->coeffs[4*i+0] &= 0x3FFFF;

    r->coeffs[4*i+1]  = a[9*i+2] >> 2;
    r->coeffs[4*i+1] |= (uint32_t)a[9*i+3] << 6;
    r->coeffs[4*i+1] |= (uint32_t)a[9*i+4] << 14;
    r->coeffs[4*i+1] &= 0x3FFFF;

    r->coeffs[4*i+2]  = a[9*i+4] >> 4;
    r->coeffs[4*i+2] |= (uint32_t)a[9*i+5] << 4;
    r->coeffs[4*i+2] |= (uint32_t)a[9*i+6] << 12;
    r->coeffs[4*i+2] &= 0x3FFFF;

    r->coeffs[4*i+3]  = a[9*i+6] >> 6;
    r->coeffs[4*i+3] |= (uint32_t)a[9*i+7] << 2;
    r->coeffs[4*i+3] |= (uint32_t)a[9*i+8] << 10;
    r->coeffs[4*i+3] &= 0x3FFFF;

    r->coeffs[4*i+0] = GAMMA1 - r->coeffs[4*i+0];
    r->coeffs[4*i+1] = GAMMA1 - r->coeffs[4*i+1];
    r->coeffs[4*i+2] = GAMMA1 - r->coeffs[4*i+2];
    r->coeffs[4*i+3] = GAMMA1 - r->coeffs[4*i+3];
  }
#elif GAMMA1 == (1 << 19)
  for(i = 0; i < N/2; ++i) {
    r->coeffs[2*i+0]  = a[5*i+0];
    r->coeffs[2*i+0] |= (uint32_t)a[5*i+1] << 8;
    r->coeffs[2*i+0] |= (uint32_t)a[5*i+2] << 16;
    r->coeffs[2*i+0] &= 0xFFFFF;

    r->coeffs[2*i+1]  = a[5*i+2] >> 4;
    r->coeffs[2*i+1] |= (uint32_t)a[5*i+3] << 4;
    r->coeffs[2*i+1] |= (uint32_t)a[5*i+4] << 12;
    /* r->coeffs[2*i+1] &= 0xFFFFF; */ /* No effect, since we're anyway at 20 bits */

    r->coeffs[2*i+0] = GAMMA1 - r->coeffs[2*i+0];
    r->coeffs[2*i+1] = GAMMA1 - r->coeffs[2*i+1];
  }
#endif

  DBENCH_STOP(*tpack);
}



/*************************************************
* Name:        polyw1_pack
*
* Description: Bit-pack polynomial w1 with coefficients in [0,15] or [0,43].
*              Input coefficients are assumed to be standard representatives.
*
* Arguments:   - uint8_t *r: pointer to output byte array with at least
*                            POLYW1_PACKEDBYTES bytes
*              - const poly *a: pointer to input polynomial
**************************************************/
void polyw1_pack(uint8_t *r, const poly *a) {
  unsigned int i;
  DBENCH_START();

#if GAMMA2 == (Q-1)/88
  for(i = 0; i < N/4; ++i) {
    r[3*i+0]  = a->coeffs[4*i+0];
    r[3*i+0] |= a->coeffs[4*i+1] << 6;
    r[3*i+1]  = a->coeffs[4*i+1] >> 2;
    r[3*i+1] |= a->coeffs[4*i+2] << 4;
    r[3*i+2]  = a->coeffs[4*i+2] >> 4;
    r[3*i+2] |= a->coeffs[4*i+3] << 2;
  }
#elif GAMMA2 == (Q-1)/32
  for(i = 0; i < N/2; ++i)
    r[i] = a->coeffs[2*i+0] | (a->coeffs[2*i+1] << 4);
#endif

  DBENCH_STOP(*tpack);
}
