#include <stdio.h>
#include <stdint.h>
#include <arm_neon.h>

// Funções de suporte
uint64_t ROL(uint64_t a, uint64_t offset) {
    return (a << offset) | (a >> (64 - offset));
}

// Função original (para comparação)
void blocoOriginal(uint64_t *Ebi, uint64_t *Ego, uint64_t *Eku, uint64_t *Ema, uint64_t *Ese, uint64_t Di, uint64_t Do, uint64_t Du, uint64_t Da, uint64_t De,
                   uint64_t *Asa, uint64_t *Ase, uint64_t *Asi, uint64_t *Aso, uint64_t *Asu) {
    
    uint64_t BCa, BCe, BCi, BCo, BCu;

    *Ebi ^= Di;
    BCa = ROL(*Ebi, 62);
    *Ego ^= Do;
    BCe = ROL(*Ego, 55);
    *Eku ^= Du;
    BCi = ROL(*Eku, 39);
    *Ema ^= Da;
    BCo = ROL(*Ema, 41);
    *Ese ^= De;
    BCu = ROL(*Ese, 2);

    *Asa = BCa ^ ((~BCe) & BCi);
    *Ase = BCe ^ ((~BCi) & BCo);
    *Asi = BCi ^ ((~BCo) & BCu);
    *Aso = BCo ^ ((~BCu) & BCa);
    *Asu = BCu ^ ((~BCa) & BCe);
}

// Função otimizada (usando NEON)
void blocoOtimizado(uint64_t *Ebi, uint64_t *Ego, uint64_t *Eku, uint64_t *Ema, uint64_t *Ese, uint64_t Di, uint64_t Do, uint64_t Du, uint64_t Da, uint64_t De,
                    uint64_t *Asa, uint64_t *Ase, uint64_t *Asi, uint64_t *Aso, uint64_t *Asu) {

    uint64x2_t BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec;

    *Ebi ^= Di;
    BCa_vec = vdupq_n_u64(ROL(*Ebi, 62));

    *Ego ^= Do;
    BCe_vec = vdupq_n_u64(ROL(*Ego, 55));

    *Eku ^= Du;
    BCi_vec = vdupq_n_u64(ROL(*Eku, 39));

    *Ema ^= Da;
    BCo_vec = vdupq_n_u64(ROL(*Ema, 41));

    *Ese ^= De;
    BCu_vec = vdupq_n_u64(ROL(*Ese, 2));

    // Aplicação do passo "Chi" usando NEON
    uint64x2_t result_Aga = veorq_u64(BCa_vec, vandq_u64(vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(BCe_vec))), BCi_vec));
    uint64x2_t result_Age = veorq_u64(BCe_vec, vandq_u64(vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(BCi_vec))), BCo_vec));
    uint64x2_t result_Agi = veorq_u64(BCi_vec, vandq_u64(vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(BCo_vec))), BCu_vec));
    uint64x2_t result_Ago = veorq_u64(BCo_vec, vandq_u64(vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(BCu_vec))), BCa_vec));
    uint64x2_t result_Agu = veorq_u64(BCu_vec, vandq_u64(vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(BCa_vec))), BCe_vec));

    *Asa = vgetq_lane_u64(result_Aga, 0);
    *Ase = vgetq_lane_u64(result_Age, 0);
    *Asi = vgetq_lane_u64(result_Agi, 0);
    *Aso = vgetq_lane_u64(result_Ago, 0);
    *Asu = vgetq_lane_u64(result_Agu, 0);
}

// Função de teste
void testaBlocoKeccac() {
    // Variáveis de entrada para as duas funções (originais e otimizadas)
    uint64_t Ebi = 0x1234567890ABCDEF, Ego = 0x0FEDCBA098765432;
    uint64_t Eku = 0xABCD1234DCBA5678, Ema = 0x5678ABCD1234DCBA, Ese = 0x9999999999999999;
    uint64_t Di = 0x1234, Do = 0x5678, Du = 0x9ABC, Da = 0xDEF0, De = 0x1357;
    uint64_t Asa1, Ase1, Asi1, Aso1, Asu1;
    uint64_t Asa2, Ase2, Asi2, Aso2, Asu2;

    // Executa a função original
    blocoOriginal(&Ebi, &Ego, &Eku, &Ema, &Ese, Di, Do, Du, Da, De, &Asa1, &Ase1, &Asi1, &Aso1, &Asu1);

    // Reinicializa as variáveis de entrada (para garantir que os dados sejam iguais)
    Ebi = 0x1234567890ABCDEF; Ego = 0x0FEDCBA098765432;
    Eku = 0xABCD1234DCBA5678; Ema = 0x5678ABCD1234DCBA; Ese = 0x90ABCDEF12345678;

    // Executa a função otimizada
    blocoOtimizado(&Ebi, &Ego, &Eku, &Ema, &Ese, Di, Do, Du, Da, De, &Asa2, &Ase2, &Asi2, &Aso2, &Asu2);

    // Comparação das saídas
    printf("Comparação dos resultados:\n");
    printf("Asa: %llx (original) vs %llx (otimizado)\n", Asa1, Asa2);
    printf("Ase: %llx (original) vs %llx (otimizado)\n", Ase1, Ase2);
    printf("Asi: %llx (original) vs %llx (otimizado)\n", Asi1, Asi2);
    printf("Aso: %llx (original) vs %llx (otimizado)\n", Aso1, Aso2);
    printf("Asu: %llx (original) vs %llx (otimizado)\n", Asu1, Asu2);

    // Verifica se todas as saídas são iguais
    if (Asa1 == Asa2 && Ase1 == Ase2 && Asi1 == Asi2 && Aso1 == Aso2 && Asu1 == Asu2) {
        printf("O teste passou! As saídas são equivalentes.\n");
    } else {
        printf("O teste falhou! As saídas são diferentes.\n");
    }
}

int main() {
    testaBlocoKeccac();
    return 0;
}
