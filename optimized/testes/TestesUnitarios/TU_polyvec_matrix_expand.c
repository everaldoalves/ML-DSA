//************************************************************************************************
// Autor: Everaldo Alves
// Data: 11 de Setembro/2024
// Função: polyvec_matrix_expand
// Descrição: Esta função realiza a expansão de uma matriz de polinômios
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜30% nos ciclos de CPU
//************************************************************************************************


#include <arm_neon.h>
#include "cpucycles.h"
#include "symmetric.h"
#include "fips202.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>

#define SEEDBYTES 32

#define N 256
#define Q 8380417
#define K 4  // {4,6,8}
#define NTESTS 10000
#define POLY_UNIFORM_NBLOCKS ((768 + STREAM128_BLOCKBYTES - 1)/STREAM128_BLOCKBYTES)
#define L 4   // {4,5,7}

// Estrutura de um polinômio
typedef struct {
    int32_t coeffs[N];
} poly;

/* Vectors of polynomials of length L */
typedef struct {
  poly vec[L];
} polyvecl;



// Função para inicializar o polinômio com valores aleatórios
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
static unsigned int rej_uniform(int32_t *a,
                                unsigned int len,
                                const uint8_t *buf,
                                unsigned int buflen)
{
  unsigned int ctr, pos;
  uint32_t t;
  //DBENCH_START();

  ctr = pos = 0;
  while(ctr < len && pos + 3 <= buflen) {
    t  = buf[pos++];
    t |= (uint32_t)buf[pos++] << 8; //
    t |= (uint32_t)buf[pos++] << 16; // Esta linha calcula o valor de t a partir de 3 bytes do array buf fazendo um deslocamento de 8 bits para a esquerda em cada iteração
    t &= 0x7FFFFF; // Esta linha representa uma operação de bitwise AND para garantir que o valor de t esteja entre 0 e 2^23 - 1 (ou seja, 8380417 - 1)

    if(t < Q)
      a[ctr++] = t;
  }

  //DBENCH_STOP(*tsample);
  return ctr;
}

// Função otiimizada com NEON. Esta funcionando corretamente
static unsigned int rej_uniform_neon(int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    unsigned int ctr = 0, pos = 0;
    uint32x4_t mask = vdupq_n_u32(0x7FFFFF);  // Máscara para extrair 24 bits

    while (ctr < len && pos + 12 <= buflen) {
        uint32x4_t t_vec;
        t_vec = vsetq_lane_u32(buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16), t_vec, 0);
        t_vec = vsetq_lane_u32(buf[pos + 3] | (buf[pos + 4] << 8) | (buf[pos + 5] << 16), t_vec, 1);
        t_vec = vsetq_lane_u32(buf[pos + 6] | (buf[pos + 7] << 8) | (buf[pos + 8] << 16), t_vec, 2);
        t_vec = vsetq_lane_u32(buf[pos + 9] | (buf[pos + 10] << 8) | (buf[pos + 11] << 16), t_vec, 3);

        t_vec = vandq_u32(t_vec, mask);

        // Armazenar o vetor NEON em um array
        uint32_t t[4];
        vst1q_u32(t, t_vec);  // Descarregar o vetor para uma array normal

        for (int i = 0; i < 4 && ctr < len; i++) {
            if (t[i] < Q) {
                a[ctr++] = t[i];
            }
        }

        pos += 12;  // Avançar 12 bytes
    }

    while (ctr < len && pos + 3 <= buflen) {
        uint32_t t = buf[pos++];
        t |= (uint32_t)buf[pos++] << 8;
        t |= (uint32_t)buf[pos++] << 16;
        t &= 0x7FFFFF;

        if (t < Q) {
            a[ctr++] = t;
        }
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
void poly_uniform(poly *a,
                  const uint8_t seed[SEEDBYTES],
                  uint16_t nonce)
{
  unsigned int i, ctr, off;
  unsigned int buflen = POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES;
  uint8_t buf[POLY_UNIFORM_NBLOCKS*STREAM128_BLOCKBYTES + 2];
  stream128_state state;

  stream128_init(&state, seed, nonce);
  stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

  ctr = rej_uniform(a->coeffs, N, buf, buflen);

  while(ctr < N) {
    off = buflen % 3;
    for(i = 0; i < off; ++i)
      buf[i] = buf[buflen - off + i];

    stream128_squeezeblocks(buf + off, 1, &state);
    buflen = STREAM128_BLOCKBYTES + off;
    ctr += rej_uniform(a->coeffs + ctr, N - ctr, buf, buflen);
  }
}

void poly_uniform_neon(poly *a,
                                    const uint8_t seed[SEEDBYTES],
                                    uint16_t nonce) {
    unsigned int ctr, off;
    unsigned int buflen = POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES;
    uint8_t buf[POLY_UNIFORM_NBLOCKS * STREAM128_BLOCKBYTES + 2];
    stream128_state state;

    // Inicializar o stream de SHAKE128
    stream128_init(&state, seed, nonce);
    stream128_squeezeblocks(buf, POLY_UNIFORM_NBLOCKS, &state);

    // Usar a função rej_uniform_neon para gerar os coeficientes
    ctr = rej_uniform_neon(a->coeffs, N, buf, buflen);

    // Processar blocos pequenos adicionais apenas se necessário
    while (ctr < N) {
        off = buflen % 3;
        if (off > 0) {
            for (unsigned int i = 0; i < off; ++i) {
                buf[i] = buf[buflen - off + i];
            }
        }

        stream128_squeezeblocks(buf + off, 1, &state);
        buflen = STREAM128_BLOCKBYTES + off;

        ctr += rej_uniform_neon(a->coeffs + ctr, N - ctr, buf, buflen);
    }
}



/*************************************************
* Name:        expand_mat
*
* Description: Implementation of ExpandA. Generates matrix A with uniformly
*              random coefficients a_{i,j} by performing rejection
*              sampling on the output stream of SHAKE128(rho|j|i)
*
* Arguments:   - polyvecl mat[K]: output matrix
*              - const uint8_t rho[]: byte array containing seed rho
**************************************************/
void polyvec_matrix_expand(polyvecl mat[K], const uint8_t rho[SEEDBYTES]) {
  unsigned int i, j;

  for(i = 0; i < K; ++i)
    for(j = 0; j < L; ++j)
      poly_uniform(&mat[i].vec[j], rho, (i << 8) + j);
}


// Função para medir tempo em ciclos de clock
uint64_t measure_cycles(void (*func)(int32_t*, unsigned int, const uint8_t*, unsigned int), int32_t *a, unsigned int len, const uint8_t *buf, unsigned int buflen) {
    uint64_t start, end;
    start = clock();
    func(a, len, buf, buflen);
    end = clock();
    return end - start;
}

uint64_t measure_cycles_poly_uniform(void (*func)(poly *, const uint8_t *, uint16_t), poly *a, const uint8_t *seed, uint16_t nonce) {
    uint64_t start, end;
    start = clock();  // Iniciar a medição do tempo
    func(a, seed, nonce);  // Chamada da função a ser medida
    end = clock();  // Fim da medição do tempo
    return end - start;  // Retornar a diferença de ciclos
}


int main() {
    int32_t a_ref[N], a_neon[N];
    uint8_t buf[3 * N];  // Exemplo de buffer de bytes aleatórios
    unsigned int len = N, buflen = 3 * N;
   
    // Preenchendo o buffer com valores aleatórios
    for (unsigned int i = 0; i < buflen; i++) {
        buf[i] = rand() % 256;
    }

    // Chamando a função de referência
    uint64_t cycles_ref = measure_cycles((void (*)(int32_t*, unsigned int, const uint8_t*, unsigned int))rej_uniform, a_ref, len, buf, buflen);

    // Chamando a função otimizada com NEON
    uint64_t cycles_neon = measure_cycles((void (*)(int32_t*, unsigned int, const uint8_t*, unsigned int))rej_uniform_neon, a_neon, len, buf, buflen);

    // Comparando as saídas
    int correct = 1;
    for (unsigned int i = 0; i < len; i++) {
        if (a_ref[i] != a_neon[i]) {
            correct = 0;
            printf("Erro: a_ref[%d] = %d, a_neon[%d] = %d\n", i, a_ref[i], i, a_neon[i]);
        }
    }

    // Resultados
    if (correct) {
        printf("Saídas são iguais!\n");
    } else {
        printf("Saídas são diferentes!\n");
    }

    // Exibir ciclos e speed-up
    printf("Ciclos de referência: %llu\n", cycles_ref);
    printf("Ciclos da função otimizada com NEON: %llu\n", cycles_neon);
    printf("Speed-up: %.2f X\n", (double)cycles_ref / cycles_neon);
    

    // ************************************************************************************************************
    // Testar a função poly_uniform
    // ************************************************************************************************************
    poly pa_ref, pa_neon;
   uint8_t seed[SEEDBYTES] = {
    0xf7, 0xe8, 0x05, 0xae, 0x1b, 0xbc, 0xf3, 0x3f,
    0x4b, 0x1d, 0x4e, 0xed, 0x05, 0x4a, 0xa3, 0x40,
    0x1e, 0x89, 0xbe, 0xa6, 0x73, 0xa1, 0xd5, 0x51,
    0x84, 0xc2, 0xea, 0xfb, 0xd6, 0x54, 0xff, 0xd5
}; // Seed de exemplo
    uint16_t nonce = 0;

    // Testar a função de referência
    uint64_t cycles_ref2 = measure_cycles_poly_uniform((void (*)(poly*, const uint8_t*, uint16_t))poly_uniform, &pa_ref, seed, nonce);

    // Testar a função otimizada com NEON
    uint64_t cycles_neon2 = measure_cycles_poly_uniform((void (*)(poly*, const uint8_t*, uint16_t))poly_uniform_neon, &pa_neon, seed, nonce);

    // Comparar saídas
    correct = 1;
    for (int i = 0; i < N; i++) {
        if (pa_ref.coeffs[i] != pa_neon.coeffs[i]) {
            correct = 0;
            printf("Erro: a_ref.coeffs[%d] = %d, a_neon.coeffs[%d] = %d\n", i, pa_ref.coeffs[i], i, pa_neon.coeffs[i]);
        }
    }

    if (correct) {
        printf("Saídas são iguais!\n");
    } else {
        printf("Saídas são diferentes!\n");
    }

    // Exibir ciclos e speed-up
    printf("Ciclos de referência: %llu\n", cycles_ref2);
    printf("Ciclos da função otimizada com NEON: %llu\n", cycles_neon2);
    printf("Speed-up: %.2f X\n", (double)cycles_ref2 / cycles_neon2);

    return 0;
}