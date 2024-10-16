#ifndef NTT_H
#define NTT_H

#include <stdint.h>
#include "params.h"
#include <arm_neon.h>

//#define ntt DILITHIUM_NAMESPACE(ntt)
void ntt(int32_t a[N]);

//#define invntt_tomont DILITHIUM_NAMESPACE(invntt_tomont)
void invntt_tomont(int32_t a[N]);

void ntt_neon(int32_t *a);
 
void invntt_tomont_neon(int32_t a[N]);
int32_t montgomery_reducev2(int64_t t);

int32x4_t montgomery_reduceX(int32x4_t t, int32x4_t q_vec); 
int32x4_t barrett_reduce(int32x4_t t, int32x4_t q_vec); 

#endif

