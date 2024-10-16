#include <stdint.h>
#include "params.h"
#include "reduce.h"
#include "arm_neon.h"
#include <stdio.h>


/*************************************************
* Name:        montgomery_reduce
*
* Description: For finite field element a with -2^{31}Q <= a <= Q*2^31,
*              compute r \equiv a*2^{-32} (mod Q) such that -Q < r < Q.
*
* Arguments:   - int64_t: finite field element a
*
* Returns r.
**************************************************/
int32_t montgomery_reduce(int64_t a) {
  int32_t t;

  t = (int64_t)(int32_t)a*QINV;
  t = (a - (int64_t)t*Q) >> 32;
  return t;
}


/*************************************************
* Name:        montgomery_reduce NEON
*
* Description: For finite field element a with -2^{31}Q <= a <= Q*2^31,
*              compute r \equiv a*2^{-32} (mod Q) such that -Q < r < Q.
*
* Arguments:   - int64_t: finite field element a
*
* Returns r.
**************************************************/

int32x4_t montgomery_reduce1(int64x2x2_t a_vec) {
  // Dividimos o vetor de 64 bits em dois vetores de 32 bits
  int32x4_t t_low, t_high, t;

  // Convertendo cada parte do vetor de 64 bits para dois vetores de 32 bits
  int32x2_t a_low = vmovn_s64(a_vec.val[0]);
  int32x2_t a_high = vmovn_s64(a_vec.val[1]);

  // Combina as partes baixas e altas para criar um vetor de 4 elementos
  int32x4_t a_32 = vcombine_s32(a_low, a_high);

  // t = a_32 * QINV mod 2^32
  t = vmulq_n_s32(a_32, QINV);

  // t = (a_32 - t * Q) >> 32
  t = vshrq_n_s32(vsubq_s32(a_32, vmulq_n_s32(t, Q)), 32);
  
  return t;
}

int32x4_t montgomery_reduce_neon1(int32x4_t a_vec) {
    // Carregar a constante QINV e Q como vetores de 32 bits
    int32x4_t QINV_vec = vdupq_n_s32(QINV);
    int32x4_t Q_vec = vdupq_n_s32(Q);

    // Multiplicar 'a' por QINV para cada elemento de a_vec
    int32x4_t t_vec = vmulq_s32(a_vec, QINV_vec);

    // Multiplicar t por Q e subtrair de a
    int32x4_t res_vec = vmlsq_s32(a_vec, t_vec, Q_vec);

    // Deslocar para ajustar o valor final (>> 32)
    int32x4_t final_res = vshrq_n_s32(res_vec, 32);

    return final_res;
}





/*************************************************
* Name:        reduce32
*
* Description: For finite field element a with a <= 2^{31} - 2^{22} - 1,
*              compute r \equiv a (mod Q) such that -6283009 <= r <= 6283007.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t reduce32(int32_t a) {
  int32_t t;

  t = (a + (1 << 22)) >> 23;
  t = a - t*Q;
  return t;
}

/*************************************************
* Name:        caddq
*
* Description: Add Q if input coefficient is negative.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t caddq(int32_t a) {
  a += (a >> 31) & Q;
  return a;
}

/*************************************************
* Name:        freeze
*
* Description: For finite field element a, compute standard
*              representative r = a mod^+ Q.
*
* Arguments:   - int32_t: finite field element a
*
* Returns r.
**************************************************/
int32_t freeze(int32_t a) {
  a = reduce32(a);
  a = caddq(a);
  return a;
}
