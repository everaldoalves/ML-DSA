#ifndef REDUCE_H
#define REDUCE_H

#include <stdint.h>
#include "params.h"
#include <arm_neon.h>

#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32

#define montgomery_reduce DILITHIUM_NAMESPACE(montgomery_reduce)
int32_t montgomery_reduce(int64_t a);
int32x4_t montgomery_reduce1(int64x2x2_t a_vec);

#define reduce32 DILITHIUM_NAMESPACE(reduce32)
int32_t reduce32(int32_t a);

#define caddq DILITHIUM_NAMESPACE(caddq)
int32_t caddq(int32_t a);

#define freeze DILITHIUM_NAMESPACE(freeze)
int32_t freeze(int32_t a);

int32_t montgomery_reduce_neon(int32_t a);

#endif
