#ifndef CONFIG_H
#define CONFIG_H

//#define DILITHIUM_MODE 2
#define DILITHIUM_RANDOMIZED_SIGNING
//#define USE_RDPMC
//#define DBENCH

#ifndef DILITHIUM_MODE
#define DILITHIUM_MODE 2
#endif

#if DILITHIUM_MODE == 2
#define CRYPTO_ALGNAME "Dilithium2"
#define DILITHIUM_NAMESPACETOP dilithium2_everaldo
#define DILITHIUM_NAMESPACE(s) dilithium2_everaldo_##s
#elif DILITHIUM_MODE == 3
#define CRYPTO_ALGNAME "Dilithium3"
#define DILITHIUM_NAMESPACETOP dilithium3_everaldo
#define DILITHIUM_NAMESPACE(s) dilithium3_everaldo_##s
#elif DILITHIUM_MODE == 5
#define CRYPTO_ALGNAME "Dilithium5"
#define DILITHIUM_NAMESPACETOP dilithium5_everaldo
#define DILITHIUM_NAMESPACE(s) dilithium5_everaldo_##s
#endif

#endif
