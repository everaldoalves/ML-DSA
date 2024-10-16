.section .rodata  // Seção de dados somente leitura

zetas:
    .word 0, 25847, -2608894, -518909
    .word 237124, -777960, -876248, 466468
    .word 1826347, 2353451, -359251, -2091905
    .word 3119733, -2884855, 3111497, 2680103
    .word 2725464, 1024112, -1079900, 3585928
    .word -549488, -1119584, 2619752, -2108549
    .word -2118186, -3859737, -1399561, -3277672
    .word 1757237, -19422, 4010497, 280005
    .word 2706023, 95776, 3077325, 3530437
    .word -1661693, -3592148, -2537516, 3915439
    .word -3861115, -3043716, 3574422, -2867647
    .word 3539968, -300467, 2348700, -539299
    .word -1699267, -1643818, 3505694, -3821735
    .word 3507263, -2140649, -1600420, 3699596
    .word 811944, 531354, 954230, 3881043
    .word 3900724, -2556880, 2071892, -2797779
    .word -3930395, -1528703, -3677745, -3041255
    .word -1452451, 3475950, 2176455, -1585221
    .word -1257611, 1939314, -4083598, -1000202
    .word -3190144, -3157330, -3632928, 126922
    .word 3412210, -983419, 2147896, 2715295
    .word -2967645, -3693493, -411027, -2477047
    .word -671102, -1228525, -22981, -1308169
    .word -381987, 1349076, 1852771, -1430430
    .word -3343383, 264944, 508951, 3097992
    .word 44288, -1100098, 904516, 3958618
    .word -3724342, -8578, 1653064, -3249728
    .word 2389356, -210977, 759969, -1316856
    .word 189548, -3553272, 3159746, -1851402
    .word -2409325, -177440, 1315589, 1341330
    .word 1285669, -1584928, -812732, -1439742
    .word -3019102, -3881060, -3628969, 3839961
    .word 2091667, 3407706, 2316500, 3817976
    .word -3342478, 2244091, -2446433, -3562462
    .word 266997, 2434439, -1235728, 3513181
    .word -3520352, -3759364, -1197226, -3193378
    .word 900702, 1859098, 909542, 819034
    .word 495491, -1613174, -43260, -522500
    .word -655327, -3122442, 2031748, 3207046
    .word -3556995, -525098, -768622, -3595838
    .word 342297, 286988, -2437823, 4108315
    .word 3437287, -3342277, 1735879, 203044
    .word 2842341, 2691481, -2590150, 1265009
    .word 4055324, 1247620, 2486353, 1595974
    .word -3767016, 1250494, 2635921, -3548272
    .word -2994039, 1869119, 1903435, -1050970
    .word -1333058, 1237275, -3318210, -1430225
    .word -451100, 1312455, 3306115, -1962642
    .word -1279661, 1917081, -2546312, -1374803
    .word 1500165, 777191, 2235880, 3406031
    .word -542412, -2831860, -1671176, -1846953
    .word -2584293, -3724270, 594136, -3776993
    .word -2013608, 2432395, 2454455, -164721
    .word 1957272, 3369112, 185531, -1207385
    .word -3183426, 162844, 1616392, 3014001
    .word 810149, 1652634, -3694233, -1799107
    .word -3038916, 3523897, 3866901, 269760
    .word 2213111, -975884, 1717735, 472078
    .word -426683, 1723600, -1803090, 1910376
    .word -1667432, -1104333, -260646, -3833893
    .word -2939036, -2235985, -420899, -2286327
    .word 183443, -976891, 1612842, -3545687
    .word -554416, 3919660, -48306, -1362209
    .word 3937738, 1400424, -846154, 1976782



.global ntt_assembly
.global intt_assembly
.global montgomery_reduce_assembly

// NTT - Transformada Número Teórica (NTT)
ntt_assembly:
    mov x4, #256                // Número de coeficientes
    ldr x5, =zetas               // Carregar o endereço da tabela de zetas
    mov x6, #0                   // Inicializar contador de zetas

loop_ntt:
    ldr q0, [x0], #16            // Carregar quatro coeficientes de poly
    ldr q1, [x5, x6]             // Carregar quatro zetas correspondentes
    bl montgomery_reduce_assembly  // Reduzir cada coeficiente

    mul v0.4s, v0.4s, v1.4s      // Multiplicação ponto a ponto (coeff * zeta)
    add x6, x6, #16              // Incrementar índice para próxima zeta
    str q0, [x1], #16            // Armazenar o resultado no local de saída

    subs x4, x4, #4              // Verificar se terminamos todos os coeficientes
    bne loop_ntt                 // Se não, continuar o loop

    ret

// INTT - Inversa da NTT, reutilizando o vetor zetas
intt:
    mov x4, #256                // Número de coeficientes
    ldr x5, =zetas               // Carregar o endereço da tabela de zetas
    mov x6, #(255 * 4)           // Inicializar contador para o final do vetor (255 * 4 bytes = último valor)

loop_intt:
    ldr q0, [x0], #16            // Carregar quatro coeficientes de poly
    ldr q1, [x5, x6]             // Carregar quatro zetas em ordem inversa
    bl montgomery_reduce_assembly  // Reduzir cada coeficiente

    mul v0.4s, v0.4s, v1.4s      // Multiplicação ponto a ponto (coeff * zeta inversa)
    sub x6, x6, #16              // Decrementar índice para acessar zetas em ordem inversa
    str q0, [x1], #16            // Armazenar o resultado no local de saída

    subs x4, x4, #4              // Verificar se terminamos todos os coeficientes
    bne loop_intt                 // Se não, continuar o loop

    ret

// Função de redução de Montgomery otimizada
montgomery_reduce_assembly:
    // Vetores para Q e QINV
    movi v1.4s, #QINV           // Carregar QINV em todos os elementos do vetor
    movi v2.4s, #Q              // Carregar Q em todos os elementos do vetor

    // Multiplicação t * QINV (parte baixa do vetor)
    smull v3.2d, v0.2s, v1.2s   // Multiplicação da parte baixa dos vetores (primeiros 2 elementos)
    smull2 v4.2d, v0.4s, v1.4s  // Multiplicação da parte alta dos vetores (últimos 2 elementos)

    // Extração da parte baixa da multiplicação para 32 bits
    xtn v3.2s, v3.2d            // Reduzir para 32 bits (primeiros 2 elementos)
    xtn2 v3.4s, v4.2d           // Reduzir para 32 bits (últimos 2 elementos)

    // Multiplicação (u * Q)
    smull v4.2d, v3.2s, v2.2s   // Multiplicação da parte baixa (u * Q)
    smull2 v5.2d, v3.4s, v2.4s  // Multiplicação da parte alta (u * Q)

    // Subtração t - u * Q
    sub v0.4s, v0.4s, v4.4s     // Subtrair a multiplicação (t - u * Q) para os primeiros 4 coeficientes

    // Correção: se o resultado for negativo, adicionar Q
    movi v6.4s, #Q              // Carregar Q para ajuste
    cmgt v7.4s, v0.4s, v6.4s    // Comparar se o valor é maior que Q
    sub v0.4s, v0.4s, v7.4s     // Ajustar resultado subtraindo Q onde necessário

    ret
