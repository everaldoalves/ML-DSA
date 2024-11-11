#include <arm_neon.h>
#include <stdint.h>
#include <stddef.h>


static void store64(uint8_t x[8], uint64_t u) {
  unsigned int i;

  for(i=0;i<8;i++)
    x[i] = u >> 8*i;
}

/*************************************************
* Name:        keccak_squeezeblocks
*
* Description: Squeeze step of Keccak. Squeezes full blocks of r bytes each.
*              Modifies the state. Can be called multiple times to keep
*              squeezing, i.e., is incremental. Assumes zero bytes of current
*              block have already been squeezed.
*
* Arguments:   - uint8_t *out:   pointer to output blocks
*              - size_t nblocks: number of blocks to be squeezed (written to out)
*              - uint64_t *s:    pointer to input/output Keccak state
*              - unsigned int r: rate in bytes (e.g., 168 for SHAKE128)
**************************************************/
static void keccak_squeezeblocks(uint8_t *out,
                                 size_t nblocks,
                                 uint64_t s[25],
                                 unsigned int r)
{
  unsigned int i;

  while(nblocks > 0) {
    KeccakF1600_StatePermute(s);
    for(i=0;i<r/8;i++)
      store64(out + 8*i, s[i]);
    out += r;
    nblocks--;
  }
}

static void keccak_squeezeblocks_neon(uint8_t *out, size_t nblocks, uint64_t s[25], unsigned int r) {
    unsigned int i;

    while(nblocks > 0) {
        // Realizar a permutação no estado Keccak
        KeccakF1600_StatePermute(s);

        // Otimizar a operação de store64 usando NEON para processar 4 blocos simultaneamente
        for(i = 0; i < r / 8; i += 2) {
            uint64x2_t vec1 = vld1q_u64(&s[i]);   // Carregar dois blocos de 64 bits
            vst1q_u8(out + 16 * i, vreinterpretq_u8_u64(vec1));  // Armazenar 128 bits de saída
        }

        out += r;
        nblocks--;
    }
}
