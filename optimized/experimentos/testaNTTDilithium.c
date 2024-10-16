#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>
#include <string.h> // Para função memcpy

#include "params.h"
#include "reduce.h"
#include "ntt.h"

#define Q 8380417
#define R 41978      // 2^32 mod Q, usado para o domínio de Montgomery


// Função de conversão para o domínio de Montgomery
int32_t to_montgomery(int32_t a) {
    return montgomery_reduce((int64_t)a * R);
}

// Função para inicializar o vetor com uma entrada determinística
void initialize_input(int32_t a[N]) {
    for (int i = 0; i < N; i++) {
        a[i] = i * 10;  // Exemplo simples de inicialização determinística
        
        //printf("a[%d] = %d\n", i, a[i]);
    }
}

// Função para comparar resultados
int compare_arrays(int32_t a[N], int32_t b[N]) {
    uint16_t contador = 0;
    for (int i = 0; i < N; i++) {
        if (a[i] != b[i]) {
            contador ++;            
        }       
    }
    if (contador > 0) {
        printf("\nArrays diferentes. %d diferenças encontradas! :(\n", contador);
        return 0; // Arrays são diferentes
    }
    return 1; // Arrays são iguais
}
void testar_montgomery_reduce() {
    // Vamos testar com 4 valores por vez (para usar com NEON)
    int64_t valores[4] = {10, 11, 12, 13};  // Valores de teste para comparação
    int32_t resultados_padroes[4];
    int32_t resultados_neon[4];

    // Teste da função escalar (padrão)
    for (int i = 0; i < 4; i++) {
        resultados_padroes[i] = montgomery_reduce(valores[i]);
        printf("Resultado Padrão [%d] = %d\n", i, resultados_padroes[i]);
    }

    // Carregar os valores no vetor NEON
    int32x4_t a_vec = {valores[0], valores[1], valores[2], valores[3]};
    
    // Teste da função NEON
    int32x4_t resultado_vec = montgomery_reduce_neon(a_vec);
    
    // Armazenar os resultados de volta em um array
    vst1q_s32(resultados_neon, resultado_vec);

    // Comparar os resultados
    for (int i = 0; i < 4; i++) {
        printf("Comparação: Padrão = %d, NEON = %d\n", resultados_padroes[i], resultados_neon[i]);
    }
}

// Função para imprimir o array para debug
void print_array(const char* label, int32_t a[N]) {
    printf("%s: ", label);
    for (int i = 0; i < N; i++) {
        printf(" %d ", a[i]);
    }
    printf("\n");
}

int main() {
    int32_t a[N], a_copy[N] = { 0,  2,  3,  3,  3, -1,  3,  2,  2,  3,  1,  3, -3, -3, -2, -2,  3,   0,  2,  3,  1,  3, -2,  0,  1, -3, -3, -3, -1,  3, -2,  3, -1, -1,   1,  3,  3,  0, -1,  0,  2,  1,  2,  0, -2, -2, -1, -3,  3,  1,  0,   3,  2,  3,  0,  3,  0, -2,  3,  1, -3,  0, -1,  0, -3,  2,  3, -2,  -3, -1,  2, -3, -3,  2,  1, -2,  3, -1,  3, -3, -1,  2,  1,  0,  0,   1,  2,  1, -1, -3,  0,  3,  1,  2, -1, -3, -1, -3,  1, -1,  0,  3,   3, -2,  1, -1,  2,  3,  3, -1, -3, -1, -3, -2, -2,  0,  3,  1, -2,   3, -3,  1,  3,  3,  0,  0,  2,  0,  0,  0, -3,  0,  0, -2,  2,  0,   3,  3, -2, -2, -1,  2, -2,  0,  0,  1, -1,  3, -2,  1, -2,  2, -1,   3, -1,  3,  2, -2, -3, -3,  3,  3, -3,  1, -2,  2, -3, -1, -3, -3,  -2, -3,  0, -3, -2,  1,  0,  0, -3,  3, -3, -1,  3,  1,  1,  3,  1,  -2,  2,  1,  2,  2, -2,  0,  1,  2, -3, -1,  2,  0,  0, -2, -3, -1,  -2, -3,  2, -1,  2,  3,  3,  2,  3, -3,  3,  2, -3,  1, -3, -2,  3,   2, -3, -2,  1,  0, -3, -1,  1,  1, -3,  3,  0,  0, -3, -3, -3, -1,   0,  0, -3,  2, -3, -2, -3,  0, -3,  1,  2,  1,  0,  3,  3,  1, -3,   2};;
    int32_t b[N], b_copy[N] = { 0,  2,  3,  3,  3, -1,  3,  2,  2,  3,  1,  3, -3, -3, -2, -2,  3,   0,  2,  3,  1,  3, -2,  0,  1, -3, -3, -3, -1,  3, -2,  3, -1, -1,   1,  3,  3,  0, -1,  0,  2,  1,  2,  0, -2, -2, -1, -3,  3,  1,  0,   3,  2,  3,  0,  3,  0, -2,  3,  1, -3,  0, -1,  0, -3,  2,  3, -2,  -3, -1,  2, -3, -3,  2,  1, -2,  3, -1,  3, -3, -1,  2,  1,  0,  0,   1,  2,  1, -1, -3,  0,  3,  1,  2, -1, -3, -1, -3,  1, -1,  0,  3,   3, -2,  1, -1,  2,  3,  3, -1, -3, -1, -3, -2, -2,  0,  3,  1, -2,   3, -3,  1,  3,  3,  0,  0,  2,  0,  0,  0, -3,  0,  0, -2,  2,  0,   3,  3, -2, -2, -1,  2, -2,  0,  0,  1, -1,  3, -2,  1, -2,  2, -1,   3, -1,  3,  2, -2, -3, -3,  3,  3, -3,  1, -2,  2, -3, -1, -3, -3,  -2, -3,  0, -3, -2,  1,  0,  0, -3,  3, -3, -1,  3,  1,  1,  3,  1,  -2,  2,  1,  2,  2, -2,  0,  1,  2, -3, -1,  2,  0,  0, -2, -3, -1,  -2, -3,  2, -1,  2,  3,  3,  2,  3, -3,  3,  2, -3,  1, -3, -2,  3,   2, -3, -2,  1,  0, -3, -1,  1,  1, -3,  3,  0,  0, -3, -3, -3, -1,   0,  0, -3,  2, -3, -2, -3,  0, -3,  1,  2,  1,  0,  3,  3,  1, -3,   2};;

    // Inicializando entrada determinística
    //initialize_input(a);
    //initialize_input(b);
    
    testar_montgomery_reduce();


    // Fazendo cópias dos arrays para evitar alterações indesejadas
    memcpy(a_copy, a, sizeof(int32_t) * N);
    memcpy(b_copy, b, sizeof(int32_t) * N);

    // Comparando os arrays originais e as cópias
    compare_arrays(a, a_copy);
    printf("Arrays comparados com sucesso\n");

    // Executando a versão padrão da função NTT
    ntt(a);
    print_array("Resultado NTT padrão", a);

    // Executando a versão otimizada com NEON da função NTT
    ntt_neon(a_copy);
    print_array("\nResultado NTT NEON", a_copy);

    // Comparando resultados para NTT
    if (compare_arrays(a, a_copy)) {
        printf("teste NTT bem-sucedido\n");
    } else {
        printf("teste NTT falhou\n");
    }

    // Executando a versão padrão da função INVNTT
    //invntt_tomont(b);
    //print_array("Resultado INVNTT padrão", b);

    // Executando a versão otimizada com NEON da função INVNTT
    //invntt_tomont_neon(b_copy);
    //print_array("Resultado INVNTT NEON", b_copy);
/*
    // Comparando resultados para INVNTT
    if (compare_arrays(b, b_copy)) {
        printf("teste INVNTT bem-sucedido\n");
    } else {
        printf("teste INVNTT falhou\n");
    }
*/

    return 0;
}
