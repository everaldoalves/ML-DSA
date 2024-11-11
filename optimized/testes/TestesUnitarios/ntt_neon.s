.text
.global ntt_neon

// Define o valor de Q (o módulo) e QINV para a multiplicação modular
#define Q 8380417
#define QINV 58728449
#define N 256

// A função recebe o ponteiro para o polinômio a e os dados precomputados (zetas)
_ntt_neon:
    // Inicializa os registradores
    mov     x2, x0              // Pega o ponteiro para a estrutura 'a'
    mov     x3, x1              // Pega o ponteiro para os valores zetas

    // Definindo os registradores vetoriais
    ld1     {v0.4s}, [x3], #16  // Carrega os zetas em v0 (para a operação modular)
    ld1     {v1.4s}, [x2], #16  // Carrega o polinômio a

    // Loop sobre os blocos de tamanho 'len', reduzindo em cada iteração
    mov     w4, #128            // Define o tamanho inicial do bloco (128)
1:
    cmp     w4, #0              // Verifica se len > 0
    beq     3f                  // Se len for zero, finaliza a função

    // Loop sobre o vetor 'a', aplicando a operação butterfly e a redução modular
    mov     w5, #0              // start = 0
2:
    cmp     w5, #N              // Verifica se alcançou o fim de 'a'
    beq     1b                  // Se sim, vai para a próxima iteração

    // Carrega zeta
    ld1     {v2.4s}, [x3], #16  // Carrega o próximo zeta

    // Calcular o endereço para a[j] e a[j + len] usando x6 como registrador auxiliar
    add     x6, x2, w5, sxtw #2  // x6 = endereço de a[j]

    // Carrega o próximo bloco de a[j] e a[j + len]
    ld1     {v3.4s}, [x6]       // Carrega a[j] (v3)
    add     x6, x6, #16         // Avança para a[j + len]
    ld1     {v4.4s}, [x6]       // Carrega a[j + len] (v4)

    // Multiplica a[j + len] pelo zeta usando mul
    mul     v5.4s, v4.4s, v2.4s  // t = zeta * a[j + len]

    // Carregar constantes grandes (QINV e Q) em vetores
    movz    w6, #0x5f45          // Carrega QINV (58728449) parte baixa
    movk    w6, #0x3812, lsl #16 // Carrega parte alta de QINV
    dup     v6.4s, w6            // Duplicar QINV em um vetor

    movz    w7, #0x7fc1          // Carrega Q (8380417) parte baixa
    movk    w7, #0x0000, lsl #16 // Carrega parte alta de Q
    dup     v7.4s, w7            // Duplicar Q em um vetor

    // Multiplicação por QINV e redução modular
    mul     v6.4s, v5.4s, v6.4s  // Multiplicação por QINV (vetor)

    // Redução modular manual (t - (t * QINV) * Q) >> 32
    mul     v7.4s, v6.4s, v7.4s  // Multiplicação por Q (vetor)
    sub     v5.4s, v5.4s, v7.4s  // t = t - (t * Q)

    // Realiza o butterfly (a[j] e a[j + len])
    sub     v4.4s, v3.4s, v5.4s   // a[j + len] = a[j] - t
    add     v3.4s, v3.4s, v5.4s   // a[j] = a[j] + t

    // Armazena os resultados de volta no vetor
    st1     {v3.4s}, [x6]        // Armazena a[j]
    add     x6, x6, #16          // Avança para a[j + len]
    st1     {v4.4s}, [x6]        // Armazena a[j + len]

    add     w5, w5, #4           // Incrementa o índice j
    b       2b                   // Volta para continuar o loop

3:
    ret                           // Retorna
