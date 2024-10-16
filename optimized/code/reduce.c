#include <stdint.h>
#include "params.h"
#include "reduce.h"

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
int32x4x2_t montgomery_reduce_neon_8(int64x2x2_t a1, int64x2x2_t a2) {
    // Extrair 32 bits mais baixos dos 64 bits resultantes
    int32x2_t a_low1_1 = vmovn_s64(a1.val[0]);
    int32x2_t a_low1_2 = vmovn_s64(a1.val[1]);
    int32x2_t a_low2_1 = vmovn_s64(a2.val[0]);
    int32x2_t a_low2_2 = vmovn_s64(a2.val[1]);

    // Multiplicar com QINV para obter o valor 't'
    int32x2_t qinv = vdup_n_s32(QINV);
    int32x2_t t1_1 = vmul_s32(a_low1_1, qinv);
    int32x2_t t1_2 = vmul_s32(a_low1_2, qinv);
    int32x2_t t2_1 = vmul_s32(a_low2_1, qinv);
    int32x2_t t2_2 = vmul_s32(a_low2_2, qinv);

    // Multiplicar por Q
    int64x2_t t_full1_1 = vmull_s32(t1_1, vdup_n_s32(Q));
    int64x2_t t_full1_2 = vmull_s32(t1_2, vdup_n_s32(Q));
    int64x2_t t_full2_1 = vmull_s32(t2_1, vdup_n_s32(Q));
    int64x2_t t_full2_2 = vmull_s32(t2_2, vdup_n_s32(Q));

    // Subtrair
    int64x2_t res1_1 = vsubq_s64(a1.val[0], t_full1_1);
    int64x2_t res1_2 = vsubq_s64(a1.val[1], t_full1_2);
    int64x2_t res2_1 = vsubq_s64(a2.val[0], t_full2_1);
    int64x2_t res2_2 = vsubq_s64(a2.val[1], t_full2_2);

    // Shift right 32 bits
    int32x2_t reduced1_1 = vshrn_n_s64(res1_1, 32);
    int32x2_t reduced1_2 = vshrn_n_s64(res1_2, 32);
    int32x2_t reduced2_1 = vshrn_n_s64(res2_1, 32);
    int32x2_t reduced2_2 = vshrn_n_s64(res2_2, 32);

    // Combinar resultados
    int32x4_t result1 = vcombine_s32(reduced1_1, reduced1_2);
    int32x4_t result2 = vcombine_s32(reduced2_1, reduced2_2);

    return (int32x4x2_t) { result1, result2 };
}

/*************************************************
* Name:        reduce32
*
* Description: For finite field element a with a <= 2^{31} - 2^{22} - 1,
*              compute r \equiv a (mod Q) such that -6283008 <= r <= 6283008.
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
