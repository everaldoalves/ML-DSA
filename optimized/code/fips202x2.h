#ifndef FIPS202x2_H
#define FIPS202x2_H

#ifndef FIPS202X2_NAMESPACE
#define FIPS202X2_NAMESPACE(s) dilithium_fips202x2_##s
#endif

#ifdef __ASSEMBLER__
/* O ABI C no MacOS exporta todos os símbolos com um sublinhado
 * no início. Isso significa que quaisquer símbolos que referimos
 * a partir de arquivos C (funções) não podem ser encontrados, e
 * todos os símbolos que referimos a partir do ASM também não podem
 * ser encontrados.
 *
 * Esta definição nos ajuda a contornar isso.
 */
#if defined(__WIN32__) || defined(__APPLE__)
#define decorate(s) _##s
#define _cdecl(s) decorate(s)
#define cdecl(s) _cdecl(FIPS202X2_NAMESPACE(##s))
#else
#define cdecl(s) FIPS202X2_NAMESPACE(##s)
#endif

#else

#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_512_RATE 72

#define REJ_UNIFORM_NBLOCKS ((768+STREAM128_BLOCKBYTES-1)/STREAM128_BLOCKBYTES)
#define REJ_UNIFORM_BUFLEN (REJ_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES)

typedef uint64x2_t v128;

typedef struct {
    v128 s[25];
} keccakx2_state;


void FIPS202X2_NAMESPACE(shake128x2_absorb_once)(keccakx2_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            size_t inlen);


void FIPS202X2_NAMESPACE(shake128x2_squeezeblocks)(uint8_t *out0,
                              uint8_t *out1,
                              size_t nblocks,
                              keccakx2_state *state);


void FIPS202X2_NAMESPACE(shake256x2_absorb_once)(keccakx2_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            size_t inlen);

void FIPS202X2_NAMESPACE(shake256x2_absorb)(keccakx2_state *state,
                            const uint8_t *in0,
                            const uint8_t *in1,
                            size_t inlen);

void FIPS202X2_NAMESPACE(shake256x2_squeezeblocks)(uint8_t *out0,
                              uint8_t *out1,
                              size_t nblocks,
                              keccakx2_state *state);


void FIPS202X2_NAMESPACE(shake128x2)(uint8_t *out0,
                uint8_t *out1,
                size_t outlen,
                const uint8_t *in0,
                const uint8_t *in1,
                size_t inlen);


void FIPS202X2_NAMESPACE(shake256x2)(uint8_t *out0,
                uint8_t *out1,
                size_t outlen,
                const uint8_t *in0,
                const uint8_t *in1,
                size_t inlen);

#endif
#endif

