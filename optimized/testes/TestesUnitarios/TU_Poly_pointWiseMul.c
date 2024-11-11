//************************************************************************************************
// Autor: Everaldo Alves
// Data: 08 de Setembro/2024
// Função: poly_pointwise_montgomery
// Descrição: Esta função realiza a multiplicação ponto a ponto de dois polinômios no domínio NTT
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: Os resultados apontam para uma redução de ˜25% no tempo de execução
//************************************************************************************************

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <arm_neon.h>
#include <stdio.h>

// Anel do Dilithium Zq[X]/(X^N + 1)
#define N 256
#define Q 8380417
#define MONT -4186625 // 2^32 % Q
#define QINV 58728449 // q^(-1) mod 2^32
#define REPEAT 10000

// Tamanho do alinhamento ideal (NEON usa 16 bytes para vetores de 128 bits)
#define ALIGNMENT 16 


typedef struct {
    int32_t coeffs[N];
} poly;

#define DBENCH_START(time_var) time_var = cpucycles()
#define DBENCH_STOP(t, time_var) t += cpucycles() - time_var - timing_overhead

// Função para medir ciclos de CPU
uint64_t cpucycles(void) {
    uint64_t result;
    asm volatile("mrs %0, cntvct_el0" : "=r" (result));
    return result;
}

// Função para calcular a sobrecarga de medição de tempo
uint64_t timing_overhead = 0;
void measure_timing_overhead(void) {
    uint64_t start = cpucycles();
    uint64_t end = cpucycles();
    timing_overhead = end - start;
}


// Função para a Lei de Amdahl
double amdahl_law(double P, int n) {
    return 1.0 / ((1 - P) + (P / n));
}

// Função para medir eficiência
double calculate_efficiency(double speedup, int num_cores) {
    return speedup / num_cores;
}



// Função montgomery_reduce - código de referência
int32_t montgomery_reduce(int64_t a) {
    int32_t t;

    t = (int64_t)(int32_t)a * QINV;
    t = (a - (int64_t)t * Q) >> 32;
    return t;
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
void poly_pointwise_montgomery(poly *c, const poly *a, const poly *b) {
  unsigned int i;
  for(i = 0; i < N; ++i)
    c->coeffs[i] = montgomery_reduce((int64_t)a->coeffs[i] * b->coeffs[i]);  
}


// Esta implementação otimizada de montgomery_reduce está gerando saídas CORRETAS e opera em conjunto com a função NTT_3
int32x2_t montgomery_reduce_neon_3(int64x2_t a) {
    int32x2_t a_low = vmovn_s64(a);  // Extrai os 32 bits inferiores
    int32x2_t qinv = vdup_n_s32(QINV);
    
    // t = a * QINV (parte baixa)
    int32x2_t t = vmul_s32(a_low, qinv);

    // t * Q (resultando em 64 bits)
    int64x2_t t_full = vmull_s32(t, vdup_n_s32(Q));

    // a - t * Q e realiza o shift de 32 bits
    int64x2_t res = vsubq_s64(a, t_full);
    
    // Shift à direita de 32 bits para fazer a redução e retorna dois valores de 32 bits
    return vshrn_n_s64(res, 32);
}

//PROPOSTA 1
//Função otimizada
void poly_pointwise_montgomery_neon2(poly *c, const poly *a, const poly *b) {
    unsigned int i;

    for (i = 0; i < N; i += 2) {
        // Carregar dois coeficientes de 'a' e 'b' em registradores NEON
        int32x2_t a_vec = vld1_s32(&a->coeffs[i]);
        int32x2_t b_vec = vld1_s32(&b->coeffs[i]);

        // Multiplicar os coeficientes de a e b, resultando em 64 bits
        int64x2_t product_vec = vmull_s32(a_vec, b_vec);

        // Aplicar montgomery_reduce usando NEON para os dois produtos
        int32x2_t reduced_vec = montgomery_reduce_neon_3(product_vec);

        // Armazenar os resultados no polinômio de saída
        vst1_s32(&c->coeffs[i], reduced_vec);
    }
}


// PROPOSTA 2
int32x4_t multiplica_poly4(const int32_t a[4], const int32_t b[4]) {
    // Carregar 4 coeficientes de 'a' e 'b' em registradores NEON
    int32x4_t a_vec = vld1q_s32(a);
    int32x4_t b_vec = vld1q_s32(b);

    // Multiplicar os coeficientes de 'a' e 'b', resultando em dois vetores de 64 bits
    int64x2_t product_low = vmull_s32(vget_low_s32(a_vec), vget_low_s32(b_vec));
    int64x2_t product_high = vmull_s32(vget_high_s32(a_vec), vget_high_s32(b_vec));

    // Aplicar montgomery_reduce usando NEON para os dois primeiros produtos
    int32x2_t reduced_low = montgomery_reduce_neon_3(product_low);
    int32x2_t reduced_high = montgomery_reduce_neon_3(product_high);

    // Combinar os dois resultados em um vetor de 4 inteiros
    return vcombine_s32(reduced_low, reduced_high);
}
void poly_pointwise_montgomery_neon(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    for (i = 0; i < N; i += 16) {
        // Processar 16 coeficientes em 4 blocos de 4
        int32x4_t c_vec1 = multiplica_poly4(&a->coeffs[i], &b->coeffs[i]);
        int32x4_t c_vec2 = multiplica_poly4(&a->coeffs[i + 4], &b->coeffs[i + 4]);
        int32x4_t c_vec3 = multiplica_poly4(&a->coeffs[i + 8], &b->coeffs[i + 8]);
        int32x4_t c_vec4 = multiplica_poly4(&a->coeffs[i + 12], &b->coeffs[i + 12]);

        // Armazenar os resultados no polinômio de saída
        vst1q_s32(&c->coeffs[i], c_vec1);
        vst1q_s32(&c->coeffs[i + 4], c_vec2);
        vst1q_s32(&c->coeffs[i + 8], c_vec3);
        vst1q_s32(&c->coeffs[i + 12], c_vec4);
    }
}

// Proposta 3
// Função de redução Montgomery otimizada para 4 coeficientes
int32x4_t montgomery_reduce_neon4(int64x2x2_t a) {
    // Extrair os 32 bits inferiores de cada bloco
    int32x2_t a_low1 = vmovn_s64(a.val[0]);
    int32x2_t a_low2 = vmovn_s64(a.val[1]);

    int32x2_t qinv = vdup_n_s32(QINV);

    // t = a * QINV (parte baixa)
    int32x2_t t1 = vmul_s32(a_low1, qinv);
    int32x2_t t2 = vmul_s32(a_low2, qinv);

    // t * Q (resultando em 64 bits)
    int64x2_t t_full1 = vmull_s32(t1, vdup_n_s32(Q));
    int64x2_t t_full2 = vmull_s32(t2, vdup_n_s32(Q));

    // a - t * Q e realizar o shift de 32 bits
    int64x2_t res1 = vsubq_s64(a.val[0], t_full1);
    int64x2_t res2 = vsubq_s64(a.val[1], t_full2);

    // Shift à direita de 32 bits para fazer a redução e retorna dois blocos de 4 valores de 32 bits
    int32x2_t reduced1 = vshrn_n_s64(res1, 32);
    int32x2_t reduced2 = vshrn_n_s64(res2, 32);

    // Combinar os resultados em um vetor de 4 valores
    int32x4_t result;
    result = vcombine_s32(reduced1, reduced2);

    return result;
}


//****************************************************************************************************************************************************************
// Função selecionada é essa. Aceleração de 3.00x de 6 ciclos para 2 ciclos
//****************************************************************************************************************************************************************
// Função otimizada para multiplicação de polinômios utilizando NEON para processar 8 coeficientes de uma vez
void poly_pointwise_montgomery8(poly *c, const poly *a, const poly *b) {
    unsigned int i;
    
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
        int32x4_t result1 = montgomery_reduce_neon4(product_vec1);
        int32x4_t result2 = montgomery_reduce_neon4(product_vec2);

        // Armazenar os resultados de volta em c->coeffs
        vst1q_s32(&c->coeffs[i], result1);     // Armazena os primeiros 4 coeficientes
        vst1q_s32(&c->coeffs[i + 4], result2); // Armazena os próximos 4 coeficientes
    }
}

// Função para inicializar o polinômio com valores aleatórios
void initialize_poly(poly *p) {
    for (int i = 0; i < N; ++i) {
        p->coeffs[i] = rand() % (2 * Q) - Q;  // Valores aleatórios em [-Q, Q]
    }
}


// Função para comparar dois polinômios
int compare_polys(poly *a, poly *b) {
    for (int i = 0; i < N; ++i) {
        if (a->coeffs[i] != b->coeffs[i]) {
            return 0; // Diferença encontrada
        }
    }
    return 1; // Polinômios iguais
}

// Função para comparar dois polinômios
void exibe_poly(poly *a, char* label) { 
    printf("Polinômio %s : ", label);
    for (int i = 0; i < N; ++i) {
        printf("%d, ", a->coeffs[i]);                    
    }    
    printf("\n\n");
}

int main() {

    // Medir a sobrecarga de medição de tempo
    measure_timing_overhead();

    // Inicializar dois polinômios com memória alinhada
    poly poly_refA, poly_refB, poly_refC, poly_optA, poly_optB, poly_optC;
    /*
    // Alocar os polinômios com alinhamento de 16 bytes
    if (posix_memalign((void**)&poly_refA, ALIGNMENT, sizeof(poly)) != 0 ||
        posix_memalign((void**)&poly_refB, ALIGNMENT, sizeof(poly)) != 0 ||
        posix_memalign((void**)&poly_optA, ALIGNMENT, sizeof(poly)) != 0 ||
        posix_memalign((void**)&poly_optB, ALIGNMENT, sizeof(poly)) != 0 ||
        posix_memalign((void**)&poly_refC, ALIGNMENT, sizeof(poly)) != 0 ||
        posix_memalign((void**)&poly_optC, ALIGNMENT, sizeof(poly)) != 0) {
        perror("Erro ao alocar memória alinhada");
        return 1; // Saia do programa se a alocação falhar
    }
*/
   
    // Variáveis para medir o tempo
    uint64_t ref_time = 0, opt_time = 0, dbench_time;

    // Repetir o teste 1000 vezes
    for (int i = 0; i < REPEAT; ++i) {
        // Inicializar novamente os polinômios para cada iteração
        initialize_poly(&poly_refA);
        initialize_poly(&poly_refB);    
        // Copiar os polinômios
        poly_optA = poly_refA;  // Copiar valores para o polinômio otimizado        
        poly_optB = poly_refB;  // Copiar valores para o polinômio otimizado  
        compare_polys(&poly_refA, &poly_optA);      
        compare_polys(&poly_refB, &poly_optB);
        //printf("\nSucesso na comparação dos polinômios na iteração %d \n", i);

        // Medir o tempo da função de referência
        DBENCH_START(dbench_time);
        poly_pointwise_montgomery(&poly_refC, &poly_refA, &poly_refB);
        DBENCH_STOP(ref_time, dbench_time);        

        // Medir o tempo da função OTIMIZADA
        DBENCH_START(dbench_time);
        poly_pointwise_montgomery8(&poly_optC, &poly_optA, &poly_optB);
        DBENCH_STOP(opt_time, dbench_time);        

        //exibe_poly(poly_refC.coeffs, "C - Referência");
        //exibe_poly(poly_optC.coeffs, "C - Otimizado");

        // Verificar se os resultados são iguais
        if (!compare_polys(&poly_refC, &poly_optC)) {
            printf("\nERRO: Os resultados são diferentes na iteração %d\n", i);
            return 1;
        }
    }

    // Calcular a média dos tempos
    uint64_t avg_ref_time = ref_time / REPEAT;
    uint64_t avg_opt_time = opt_time / REPEAT;
    double speedup = avg_ref_time / avg_opt_time;

    // Imprimir os tempos de execução médios
    printf("Tempo médio de execução da função de referência: %llu ciclos\n", avg_ref_time);
    printf("Tempo médio de execução da função otimizada: %llu ciclos\n", avg_opt_time);
    printf("Speed-up: %.2f x\n", speedup);

/*
     // Liberar a memória alinhada alocada dinamicamente
    free(poly_refA);
    free(poly_refB);
    free(poly_optA);
    free(poly_optB);
    free(poly_refC);
    free(poly_optC);
*/

    

    return 0;
}
