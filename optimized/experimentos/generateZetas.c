#include <stdio.h>
#include <stdint.h>

#define N 256
#define Q 8380417  // Módulo para as operações
#define ROOT_OF_UNITY 1753  // Raiz primitiva de N = 256


// Vetor zetas pré-calculado
int32_t zetas[256] = {
    1, 828213, 1433468, 4069333, 2280021, 3047749, 5311036, 2396706, 5653960, 3942877, 8217805, 7013404, 6421120, 5602327, 3151318, 1000946, 5622095, 2890266,
     8262262, 7484404, 7638056, 7748234, 6429413, 6217421, 5386419, 1420466, 1720886, 5641598, 1624212, 5546557, 6534019, 5443883, 4106102, 5475288, 173642, 
     297562, 5067945, 6766964, 3028980, 2593111, 5177733, 2825896, 440123, 671353, 4305375, 6082491, 4892450, 1220399, 3609004, 7539389, 5169732, 5210697, 4871262, 
     1650887, 6847978, 6762089, 5097595, 4452478, 8224738, 40904, 6364835, 5742603, 2937692, 4871444, 3073009, 4760253, 3331704, 5402091, 342106, 7075506, 733500, 
     1492105, 2017426, 7216173, 3093521, 5945716, 1952394, 3664793, 5640090, 2000816, 1650498, 7802400, 8290367, 7784634, 7750798, 7453229, 6194443, 7023434, 4527125, 
     5326179, 6939461, 6757368, 2301042, 6772779, 6427456, 2825061, 6645333, 4099511, 7674456, 6126192, 6482626, 1073160, 7975616, 4194258, 6019936, 6850864, 180769, 
     7344670, 570727, 2466615, 5141997, 1273848, 659405, 4893730, 7944740, 5150063, 803063, 3249772, 4931735, 4760642, 8370447, 23783, 7877352, 2966016, 911086, 4085734, 
     7351584, 1906269, 1753, 2045248, 7124721, 127264, 7798321, 7958936, 4427078, 6541342, 5731819, 4606485, 6116040, 8354472, 8246133, 3442167, 4880203, 6787942, 154976, 
     2757403, 669965, 1660495, 4718678, 6394360, 7442916, 2547640, 4494074, 2521739, 8115521, 3401618, 2101138, 7335652, 2275376, 6803559, 3405228, 6774612, 6853194, 6612550, 
     2017464, 5166599, 7282803, 2519059, 4434173, 8033093, 7773105, 3566801, 2333738, 4682407, 7575693, 6453888, 2382648, 5802416, 3675823, 7773591, 3332069, 6004824, 11707,
      8120372, 7387813, 1126893, 6766971, 1007019, 5687556, 5580884, 522607, 6156142, 2563271, 5360034, 7768162, 3547230, 5149339, 2017577, 2270935, 3526064, 4136400, 7682763, 
      4030377, 8136860, 5309536, 4572714, 7544405, 5596409, 5308362, 4379024, 4930165, 7624659, 5091351, 8001587, 7943069, 8168737, 1427404, 5849276, 7158274, 3860405, 2772284, 
      6606911, 4295817, 1143452, 2266014, 4966345, 6218226, 6225137, 2977176, 5617734, 1702996, 1308826, 6408553, 1748012, 2483221, 4445607, 5831115, 1554217, 6443776, 5828058, 
      1416026, 2037637, 5465186, 8244365, 6485879, 5136192, 7341376, 3355885, 106536, 5398322, 3672643, 5507189, 6110252, 4069527, 4487336, 7574059
};

// Função para calcular a potência modular (a^exp % mod)
uint32_t mod_exp(uint32_t a, uint32_t exp, uint32_t mod) {
    uint32_t res = 1;
    while (exp > 0) {
        if (exp % 2 == 1)
            res = (res * a) % mod;
        a = (a * a) % mod;
        exp /= 2;
    }
    return res;
}

// Função para calcular a inversão de bits de um byte (8 bits)
uint32_t bit_reverse(uint32_t k, int bits) {
    uint32_t result = 0;
    for (int i = 0; i < bits; i++) {
        result <<= 1;         // Move todos os bits para a esquerda
        result |= (k & 1);    // Adiciona o bit menos significativo de k a result
        k >>= 1;              // Desloca k para a direita para processar o próximo bit
    }
    return result;
}

// Função para gerar o vetor zetas com base em ζ^brv(k) mod q
void generate_zetas(int32_t zetas[N]) {
    printf("Vetor zetas = { ");
    for (int k = 0; k < N; k++) {
        // Calcula a inversão de bits de k
        uint32_t brv_k = bit_reverse(k, 8);  // 8 bits para N = 256
        // Calcula ζ^brv(k) mod q
        zetas[k] = mod_exp(ROOT_OF_UNITY, brv_k, Q);
        printf("%d, " , zetas[k]);
    }
    printf("}\n\n");
}

// Função de NTT para calcular a transformada em tempo real
void ntt(int32_t w[N]) {
    int len = N / 2;  // Começa com len = 128
    int k = 0;

    while (len >= 1) {
        for (int start = 0; start < N; start += 2 * len) {
            for (int j = start; j < start + len; j++) {
                // Carrega zeta da tabela pré-calculada
                int32_t zeta = zetas[k++];

                // Operação butterfly com operações modulares
                int64_t t = (int64_t) zeta * w[j + len] % Q;
                w[j + len] = (w[j] - t + Q) % Q;  // Subtração com wrap-around em Q
                w[j] = (w[j] + t) % Q;  // Adição com wrap-around em Q
            }
        }
        len /= 2;  // Reduz o tamanho pela metade a cada iteração
    }
}

int main() {
    // Inicialização dos coeficientes do polinômio w (exemplo simples)
    int32_t w[N];
    printf("Inicialização dos coeficientes do polinômio w: \n");
    for (int i = 0; i < N; i++) {
        w[i] = i % 7 - 3;  // Exemplo de inicialização dos coeficientes
        printf("%d ", w[i]);
    }

    // Aplica a NTT no polinômio w usando o vetor zetas pré-calculado
    ntt(w);

    // Exibe os resultados
    printf("Resultados da NTT:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", w[i]);
    }
    printf("\n");

    return 0;
}