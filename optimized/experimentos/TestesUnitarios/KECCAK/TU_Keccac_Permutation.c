//************************************************************************************************
// Autor: Everaldo Alves
// Data: 25 de Setembro/2024
// Função: keccac_permutation de FIPS202.c
// Descrição: KECCAC PERMUTATION do FIPS 202
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: DESISTIR TEMPORARIAMENTE
//************************************************************************************************

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>


// Definição das constantes
#define NROUNDS 24
#define NUM_TESTES 1000
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset))) // ROL (Rotate Left) realiza uma rotação para a esquerda de 64 bits. Como o tamanho da palavra é de 64 bits, o deslocamento é feito para a direita

// Função original (não otimizada)
void process_original(uint64_t* Aga, uint64_t* Age, uint64_t* Agi, uint64_t* Ago, uint64_t* Agu,
                      uint64_t BCa, uint64_t BCe, uint64_t BCi, uint64_t BCo, uint64_t BCu) {

    *Aga = BCa ^ ((~BCe) & BCi);
    *Age = BCe ^ ((~BCi) & BCo);
    *Agi = BCi ^ ((~BCo) & BCu);
    *Ago = BCo ^ ((~BCu) & BCa);
    *Agu = BCu ^ ((~BCa) & BCe);
}

// Função otimizada com NEON
void process_parallel(uint64_t* Aga, uint64_t* Age, uint64_t* Agi, uint64_t* Ago, uint64_t* Agu,
                      uint64x2_t BCa, uint64x2_t BCe, uint64x2_t BCi, uint64x2_t BCo, uint64x2_t BCu) {

    // NOT é realizado com vmvnq_u8, então convertemos os vetores para uint8x16_t
    uint8x16_t not_BCe = vmvnq_u8(vreinterpretq_u8_u64(BCe));
    uint8x16_t not_BCi = vmvnq_u8(vreinterpretq_u8_u64(BCi));
    uint8x16_t not_BCo = vmvnq_u8(vreinterpretq_u8_u64(BCo));
    uint8x16_t not_BCu = vmvnq_u8(vreinterpretq_u8_u64(BCu));
    uint8x16_t not_BCa = vmvnq_u8(vreinterpretq_u8_u64(BCa));

    // Reconverte os vetores para uint64x2_t após a inversão
    uint64x2_t result_Aga = veorq_u64(BCa, vandq_u64(vreinterpretq_u64_u8(not_BCe), BCi));
    uint64x2_t result_Age = veorq_u64(BCe, vandq_u64(vreinterpretq_u64_u8(not_BCi), BCo));
    uint64x2_t result_Agi = veorq_u64(BCi, vandq_u64(vreinterpretq_u64_u8(not_BCo), BCu));
    uint64x2_t result_Ago = veorq_u64(BCo, vandq_u64(vreinterpretq_u64_u8(not_BCu), BCa));
    uint64x2_t result_Agu = veorq_u64(BCu, vandq_u64(vreinterpretq_u64_u8(not_BCa), BCe));

    *Aga = vgetq_lane_u64(result_Aga, 0);
    *Age = vgetq_lane_u64(result_Age, 0);
    *Agi = vgetq_lane_u64(result_Agi, 0);
    *Ago = vgetq_lane_u64(result_Ago, 0);
    *Agu = vgetq_lane_u64(result_Agu, 0);
}

// Função para testar ambas versões
void test_comparison() {
    uint64_t Aga_orig, Age_orig, Agi_orig, Ago_orig, Agu_orig;
    uint64_t Aga_opt, Age_opt, Agi_opt, Ago_opt, Agu_opt;

    // Valores de entrada (pode-se testar diferentes valores)
    uint64_t BCa = 0x123456789ABCDEF0;
    uint64_t BCe = 0x0FEDCBA987654321;
    uint64_t BCi = 0x123456789ABCDEF0;
    uint64_t BCo = 0x0FEDCBA987654321;
    uint64_t BCu = 0x123456789ABCDEF0;

    // Teste da função original
    process_original(&Aga_orig, &Age_orig, &Agi_orig, &Ago_orig, &Agu_orig, BCa, BCe, BCi, BCo, BCu);

    // Teste da função otimizada
    uint64x2_t BCa_vec = vdupq_n_u64(BCa);
    uint64x2_t BCe_vec = vdupq_n_u64(BCe);
    uint64x2_t BCi_vec = vdupq_n_u64(BCi);
    uint64x2_t BCo_vec = vdupq_n_u64(BCo);
    uint64x2_t BCu_vec = vdupq_n_u64(BCu);
    
    process_parallel(&Aga_opt, &Age_opt, &Agi_opt, &Ago_opt, &Agu_opt, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

    // Comparando os resultados
    printf("Resultados da função original:\n");
    printf("Aga: %llu, Age: %llu, Agi: %llu, Ago: %llu, Agu: %llu\n", Aga_orig, Age_orig, Agi_orig, Ago_orig, Agu_orig);

    printf("Resultados da função otimizada com NEON:\n");
    printf("Aga: %llu, Age: %llu, Agi: %llu, Ago: %llu, Agu: %llu\n", Aga_opt, Age_opt, Agi_opt, Ago_opt, Agu_opt);

    // Verificando se os resultados são equivalentes
    if (Aga_orig == Aga_opt && Age_orig == Age_opt && Agi_orig == Agi_opt && Ago_orig == Ago_opt && Agu_orig == Agu_opt) {
        printf("Os resultados são equivalentes!\n");
    } else {
        printf("Há divergência nos resultados!\n");
    }
}


// Função original (sem NEON)
void original_function(uint64_t* Abu, uint64_t* Aga, uint64_t* Ake, uint64_t* Ami, uint64_t* Aso, 
                       uint64_t Du, uint64_t Da, uint64_t De, uint64_t Di, uint64_t Do,
                       uint64_t* BCa, uint64_t* BCe, uint64_t* BCi, uint64_t* BCo, uint64_t* BCu) {
                    
    *Abu ^= Du;
    *Aga ^= Da;    
    *Ake ^= De;
    *Ami ^= Di;
    *Aso ^= Do;
    
}

void xor_parallel(uint64_t* vars[5], uint64_t* xor_vals[5]) {
    // Carrega as primeiras duas variáveis (Eba, Ege) em NEON e faz XOR com os valores XOR
    uint64x2_t vals1 = vld1q_u64((const uint64_t*) &vars[0]);  // Carrega Eba, Ege
    uint64x2_t xor_vals1 = vld1q_u64((const uint64_t*) &xor_vals[0]);  // Carrega Da, De
    
    vals1 = veorq_u64(vals1, xor_vals1);  // Executa XOR
    vst1q_u64((uint64_t*)&vars[0], vals1);  // Armazena os resultados de volta em Eba, Ege

    // Carrega as próximas duas variáveis (Eki, Emo) em NEON e faz XOR com os valores XOR
    uint64x2_t vals2 = vld1q_u64((const uint64_t*) &vars[2]);  // Carrega Eki, Emo
    uint64x2_t xor_vals2 = vld1q_u64((const uint64_t*) &xor_vals[2]);  // Carrega Di, Do
    
    vals2 = veorq_u64(vals2, xor_vals2);  // Executa XOR
    vst1q_u64((uint64_t*)&vars[2], vals2);  // Armazena os resultados de volta em Eki, Emo

    // Processa o último XOR para Esu separadamente
    *vars[4] ^= *xor_vals[4];  // Esu ^= Du
}


// Função para verificar se as saídas são iguais
bool compare_results(uint64_t BCa1, uint64_t BCe1, uint64_t BCi1, uint64_t BCo1, uint64_t BCu1,
                     uint64_t BCa2, uint64_t BCe2, uint64_t BCi2, uint64_t BCo2, uint64_t BCu2) {
    return BCa1 == BCa2 && BCe1 == BCe2 && BCi1 == BCi2 && BCo1 == BCo2 && BCu1 == BCu2;
}

void test_comparison2 () {
// Teste para função XOR Paralela
     // Definindo as variáveis
    uint64_t Eba = 0xDEADBEEF, Ege = 0xFEEDFACE, Eki = 0xBAADF00D, Emo = 0xBEEFCAFE, Esu = 0xCAFEF00D;
    uint64_t Da = 0xABCDEF, De = 0x123456, Di = 0x789ABC, Do = 0xDEF123, Du = 0x456789;

    uint64_t BCa_orig, BCe_orig, BCi_orig, BCo_orig, BCu_orig;
    uint64_t BCa_opt, BCe_opt, BCi_opt, BCo_opt, BCu_opt;

    
    // Array de ponteiros para as variáveis (Eba, Ege, Eki, Emo, Esu)
    uint64_t* vars[5] = { &Eba, &Ege, &Eki, &Emo, &Esu };

    // Array de ponteiros para os valores de XOR (Da, De, Di, Do, Du)
    uint64_t* xor_vals[5] = { &Da, &De, &Di, &Do, &Du };

    // Chama a função de XOR paralelizada
    xor_parallel(vars, xor_vals);
    BCa_opt = *vars[0]; 
    BCe_opt = *vars[1];
    BCi_opt = *vars[2];
    BCo_opt = *vars[3];
    BCu_opt = *vars[4];

    // Exibe os resultados
    printf("Eba: 0x%016llx, Ege: 0x%016llx, Eki: 0x%016llx, Emo: 0x%016llx, Esu: 0x%016llx\n",
           Eba, Ege, Eki, Emo, Esu);


    // Executa a função original e salva os resultados
    original_function(&Eba, &Ege, &Eki, &Emo, &Esu, Da, De, Di, Do, Du, &BCa_orig, &BCe_orig, &BCi_orig, &BCo_orig, &BCu_orig);

 
    // Compara os resultados
    if (compare_results(BCa_orig, BCe_orig, BCi_orig, BCo_orig, BCu_orig, BCa_opt, BCe_opt, BCi_opt, BCo_opt, BCu_opt)) {
        printf("\n\nAs saídas são iguais!\n\n\n");
    } else {
        printf("\n\nAs saídas são diferentes!\n\n\n");
    }

   
}



/* Keccak round constants */
const uint64_t KeccakF_RoundConstants[NROUNDS] = {
  (uint64_t)0x0000000000000001ULL, 
  (uint64_t)0x0000000000008082ULL, 
  (uint64_t)0x800000000000808aULL, 
  (uint64_t)0x8000000080008000ULL, 
  (uint64_t)0x000000000000808bULL, 
  (uint64_t)0x0000000080000001ULL,
  (uint64_t)0x8000000080008081ULL,
  (uint64_t)0x8000000000008009ULL,
  (uint64_t)0x000000000000008aULL,
  (uint64_t)0x0000000000000088ULL,
  (uint64_t)0x0000000080008009ULL,
  (uint64_t)0x000000008000000aULL,
  (uint64_t)0x000000008000808bULL,
  (uint64_t)0x800000000000008bULL,
  (uint64_t)0x8000000000008089ULL,
  (uint64_t)0x8000000000008003ULL,
  (uint64_t)0x8000000000008002ULL,
  (uint64_t)0x8000000000000080ULL,
  (uint64_t)0x000000000000800aULL,
  (uint64_t)0x800000008000000aULL,
  (uint64_t)0x8000000080008081ULL,
  (uint64_t)0x8000000000008080ULL,
  (uint64_t)0x0000000080000001ULL,
  (uint64_t)0x8000000080008008ULL 
};

/*************************************************
* Name:        KeccakF1600_StatePermute
*
* Description: The Keccak F1600 Permutation
*
* Arguments:   - uint64_t *state: pointer to input/output Keccak state
**************************************************/
static void KeccakF1600_StatePermute(uint64_t state[25])
{
        int round;

        uint64_t Aba, Abe, Abi, Abo, Abu;
        uint64_t Aga, Age, Agi, Ago, Agu;
        uint64_t Aka, Ake, Aki, Ako, Aku;
        uint64_t Ama, Ame, Ami, Amo, Amu;
        uint64_t Asa, Ase, Asi, Aso, Asu;
        uint64_t BCa, BCe, BCi, BCo, BCu;
        uint64_t Da, De, Di, Do, Du;
        uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
        uint64_t Ega, Ege, Egi, Ego, Egu;
        uint64_t Eka, Eke, Eki, Eko, Eku;
        uint64_t Ema, Eme, Emi, Emo, Emu;
        uint64_t Esa, Ese, Esi, Eso, Esu;

        //copyFromState(A, state)
        Aba = state[ 0];
        Abe = state[ 1];
        Abi = state[ 2];
        Abo = state[ 3];
        Abu = state[ 4];
        Aga = state[ 5];
        Age = state[ 6];
        Agi = state[ 7];
        Ago = state[ 8];
        Agu = state[ 9];
        Aka = state[10];
        Ake = state[11];
        Aki = state[12];
        Ako = state[13];
        Aku = state[14];
        Ama = state[15];
        Ame = state[16];
        Ami = state[17];
        Amo = state[18];
        Amu = state[19];
        Asa = state[20];
        Ase = state[21];
        Asi = state[22];
        Aso = state[23];
        Asu = state[24];

        for(round = 0; round < NROUNDS; round += 2) {
            //    prepareTheta
            BCa = Aba^Aga^Aka^Ama^Asa;
            BCe = Abe^Age^Ake^Ame^Ase;
            BCi = Abi^Agi^Aki^Ami^Asi;
            BCo = Abo^Ago^Ako^Amo^Aso;
            BCu = Abu^Agu^Aku^Amu^Asu;

            //thetaRhoPiChiIotaPrepareTheta(round, A, E)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            // Bloco 1
            Aba ^= Da;
            BCa = Aba;
            Age ^= De;
            BCe = ROL(Age, 44);
            Aki ^= Di;
            BCi = ROL(Aki, 43);
            Amo ^= Do;
            BCo = ROL(Amo, 21);
            Asu ^= Du;
            BCu = ROL(Asu, 14);
            Eba =   BCa ^((~BCe)&  BCi );
            Eba ^= (uint64_t)KeccakF_RoundConstants[round];
            Ebe =   BCe ^((~BCi)&  BCo );
            Ebi =   BCi ^((~BCo)&  BCu );
            Ebo =   BCo ^((~BCu)&  BCa );
            Ebu =   BCu ^((~BCa)&  BCe );

            Abo ^= Do;
            BCa = ROL(Abo, 28);
            Agu ^= Du;
            BCe = ROL(Agu, 20);
            Aka ^= Da;
            BCi = ROL(Aka,  3);
            Ame ^= De;
            BCo = ROL(Ame, 45);
            Asi ^= Di;
            BCu = ROL(Asi, 61);
            Ega =   BCa ^((~BCe)&  BCi );
            Ege =   BCe ^((~BCi)&  BCo );
            Egi =   BCi ^((~BCo)&  BCu );
            Ego =   BCo ^((~BCu)&  BCa );
            Egu =   BCu ^((~BCa)&  BCe );

            // Bloco 2
            Abe ^= De;
            BCa = ROL(Abe,  1);
            Agi ^= Di;
            BCe = ROL(Agi,  6);
            Ako ^= Do;
            BCi = ROL(Ako, 25);
            Amu ^= Du;
            BCo = ROL(Amu,  8);
            Asa ^= Da;
            BCu = ROL(Asa, 18);
            Eka =   BCa ^((~BCe)&  BCi );
            Eke =   BCe ^((~BCi)&  BCo );
            Eki =   BCi ^((~BCo)&  BCu );
            Eko =   BCo ^((~BCu)&  BCa );
            Eku =   BCu ^((~BCa)&  BCe );

            Abu ^= Du;
            BCa = ROL(Abu, 27);
            Aga ^= Da;
            BCe = ROL(Aga, 36);
            Ake ^= De;
            BCi = ROL(Ake, 10);
            Ami ^= Di;
            BCo = ROL(Ami, 15);
            Aso ^= Do;
            BCu = ROL(Aso, 56);
            Ema =   BCa ^((~BCe)&  BCi );
            Eme =   BCe ^((~BCi)&  BCo );
            Emi =   BCi ^((~BCo)&  BCu );
            Emo =   BCo ^((~BCu)&  BCa );
            Emu =   BCu ^((~BCa)&  BCe );

            // Bloco 3
            Abi ^= Di;
            BCa = ROL(Abi, 62);
            Ago ^= Do;
            BCe = ROL(Ago, 55);
            Aku ^= Du;
            BCi = ROL(Aku, 39);
            Ama ^= Da;
            BCo = ROL(Ama, 41);
            Ase ^= De;
            BCu = ROL(Ase,  2);
            Esa =   BCa ^((~BCe)&  BCi );
            Ese =   BCe ^((~BCi)&  BCo );
            Esi =   BCi ^((~BCo)&  BCu );
            Eso =   BCo ^((~BCu)&  BCa );
            Esu =   BCu ^((~BCa)&  BCe );

            //    prepareTheta
            BCa = Eba^Ega^Eka^Ema^Esa;
            BCe = Ebe^Ege^Eke^Eme^Ese;
            BCi = Ebi^Egi^Eki^Emi^Esi;
            BCo = Ebo^Ego^Eko^Emo^Eso;
            BCu = Ebu^Egu^Eku^Emu^Esu;

            //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            // Bloco 4
            Eba ^= Da;
            BCa = Eba;
            Ege ^= De;
            BCe = ROL(Ege, 44);
            Eki ^= Di;
            BCi = ROL(Eki, 43);
            Emo ^= Do;
            BCo = ROL(Emo, 21);
            Esu ^= Du;
            BCu = ROL(Esu, 14);
            Aba =   BCa ^((~BCe)&  BCi );
            Aba ^= (uint64_t)KeccakF_RoundConstants[round+1];
            Abe =   BCe ^((~BCi)&  BCo );
            Abi =   BCi ^((~BCo)&  BCu );
            Abo =   BCo ^((~BCu)&  BCa );
            Abu =   BCu ^((~BCa)&  BCe );

            Ebo ^= Do;
            BCa = ROL(Ebo, 28);
            Egu ^= Du;
            BCe = ROL(Egu, 20);
            Eka ^= Da;
            BCi = ROL(Eka, 3);
            Eme ^= De;
            BCo = ROL(Eme, 45);
            Esi ^= Di;
            BCu = ROL(Esi, 61);
            Aga =   BCa ^((~BCe)&  BCi );
            Age =   BCe ^((~BCi)&  BCo );
            Agi =   BCi ^((~BCo)&  BCu );
            Ago =   BCo ^((~BCu)&  BCa );
            Agu =   BCu ^((~BCa)&  BCe );

            // Bloco 5
            Ebe ^= De;
            BCa = ROL(Ebe, 1);
            Egi ^= Di;
            BCe = ROL(Egi, 6);
            Eko ^= Do;
            BCi = ROL(Eko, 25);
            Emu ^= Du;
            BCo = ROL(Emu, 8);
            Esa ^= Da;
            BCu = ROL(Esa, 18);
            Aka =   BCa ^((~BCe)&  BCi );
            Ake =   BCe ^((~BCi)&  BCo );
            Aki =   BCi ^((~BCo)&  BCu );
            Ako =   BCo ^((~BCu)&  BCa );
            Aku =   BCu ^((~BCa)&  BCe );
           
            Ebu ^= Du;
            BCa = ROL(Ebu, 27);
            Ega ^= Da;
            BCe = ROL(Ega, 36);
            Eke ^= De;
            BCi = ROL(Eke, 10);
            Emi ^= Di;
            BCo = ROL(Emi, 15);
            Eso ^= Do;
            BCu = ROL(Eso, 56);
            Ama =   BCa ^((~BCe)&  BCi );
            Ame =   BCe ^((~BCi)&  BCo );
            Ami =   BCi ^((~BCo)&  BCu );
            Amo =   BCo ^((~BCu)&  BCa );
            Amu =   BCu ^((~BCa)&  BCe );

             // Bloco 6
            Ebi ^= Di;
            BCa = ROL(Ebi, 62);
            Ego ^= Do;
            BCe = ROL(Ego, 55);
            Eku ^= Du;
            BCi = ROL(Eku, 39);
            Ema ^= Da;
            BCo = ROL(Ema, 41);
            Ese ^= De;
            BCu = ROL(Ese, 2);
            Asa =   BCa ^((~BCe)&  BCi ); 
            Ase =   BCe ^((~BCi)&  BCo );
            Asi =   BCi ^((~BCo)&  BCu );
            Aso =   BCo ^((~BCu)&  BCa );
            Asu =   BCu ^((~BCa)&  BCe );
        }

        //copyToState(state, A)
        state[ 0] = Aba;
        state[ 1] = Abe;
        state[ 2] = Abi;
        state[ 3] = Abo;
        state[ 4] = Abu;
        state[ 5] = Aga;
        state[ 6] = Age;
        state[ 7] = Agi;
        state[ 8] = Ago;
        state[ 9] = Agu;
        state[10] = Aka;
        state[11] = Ake;
        state[12] = Aki;
        state[13] = Ako;
        state[14] = Aku;
        state[15] = Ama;
        state[16] = Ame;
        state[17] = Ami;
        state[18] = Amo;
        state[19] = Amu;
        state[20] = Asa;
        state[21] = Ase;
        state[22] = Asi;
        state[23] = Aso;
        state[24] = Asu;
}


static void KeccakF1600_StatePermute2(uint64_t state[25])
{
        int round;

        uint64_t Aba, Abe, Abi, Abo, Abu;
        uint64_t Aga, Age, Agi, Ago, Agu;
        uint64_t Aka, Ake, Aki, Ako, Aku;
        uint64_t Ama, Ame, Ami, Amo, Amu;
        uint64_t Asa, Ase, Asi, Aso, Asu;
        uint64_t BCa, BCe, BCi, BCo, BCu;
        uint64_t Da, De, Di, Do, Du;
        uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
        uint64_t Ega, Ege, Egi, Ego, Egu;
        uint64_t Eka, Eke, Eki, Eko, Eku;
        uint64_t Ema, Eme, Emi, Emo, Emu;
        uint64_t Esa, Ese, Esi, Eso, Esu;

        //copyFromState(A, state)
        Aba = state[ 0];
        Abe = state[ 1];
        Abi = state[ 2];
        Abo = state[ 3];
        Abu = state[ 4];
        Aga = state[ 5];
        Age = state[ 6];
        Agi = state[ 7];
        Ago = state[ 8];
        Agu = state[ 9];
        Aka = state[10];
        Ake = state[11];
        Aki = state[12];
        Ako = state[13];
        Aku = state[14];
        Ama = state[15];
        Ame = state[16];
        Ami = state[17];
        Amo = state[18];
        Amu = state[19];
        Asa = state[20];
        Ase = state[21];
        Asi = state[22];
        Aso = state[23];
        Asu = state[24];

   for(round = 0; round < NROUNDS; round += 2) {
            //    prepareTheta
            BCa = Aba^Aga^Aka^Ama^Asa;
            BCe = Abe^Age^Ake^Ame^Ase;
            BCi = Abi^Agi^Aki^Ami^Asi;
            BCo = Abo^Ago^Ako^Amo^Aso;
            BCu = Abu^Agu^Aku^Amu^Asu;

            //thetaRhoPiChiIotaPrepareTheta(round, A, E)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            // Bloco 1
            Aba ^= Da;
            BCa = Aba;
            Age ^= De;
            BCe = ROL(Age, 44);
            Aki ^= Di;
            BCi = ROL(Aki, 43);
            Amo ^= Do;
            BCo = ROL(Amo, 21);
            Asu ^= Du;
            BCu = ROL(Asu, 14);

            Eba =   BCa ^((~BCe)&  BCi );
            Eba ^= (uint64_t)KeccakF_RoundConstants[round];
            // Primeiro conjunto de operações

            // Carregar os valores em vetores NEON para operar em paralelo
            uint64x2_t BCa_vec = vcombine_u64(BCa, BCo);  // Carregar BCa e BCo
            uint64x2_t BCe_vec = vcombine_u64(BCe, BCu);  // Carregar BCe e BCu
            uint64x2_t BCi_vec = vcombine_u64(BCi, BCa);  // Carregar BCi e BCa
            uint64x2_t BCo_vec = vcombine_u64(BCo, BCu);  // Carregar BCo e BCu
            uint64x2_t BCu_vec = vcombine_u64(BCu, 0);    // Inicializar BCu_vec com BCu e 0, já que o segundo valor não é necessário

            // Realizar as operações XOR, NOT, AND em paralelo para Eba e Ebi
            uint64x2_t E_vec1 = veorq_u64(BCa_vec, vandq_u64(vmvnq_u64(BCe_vec), BCi_vec));
            Eba = vgetq_lane_u64(E_vec1, 0);  // Eba = BCa ^ ((~BCe) & BCi)
            Ebi = vgetq_lane_u64(E_vec1, 1);  // Ebi = BCi ^ ((~BCo) & BCu)

            // Realizar as operações XOR, NOT, AND em paralelo para Ebo e Ebu
            uint64x2_t E_vec2 = veorq_u64(BCo_vec, vandq_u64(vmvnq_u64(BCu_vec), BCa_vec));
            Ebo = vgetq_lane_u64(E_vec2, 0);  // Ebo = BCo ^ ((~BCu) & BCa)
            Ebu = vgetq_lane_u64(E_vec2, 1);  // Ebu = BCu ^ ((~BCa) & BCe)

            
            Eba =   BCa ^((~BCe)&  BCi );
            Ebi =   BCi ^((~BCo)&  BCu );
            Ebo =   BCo ^((~BCu)&  BCa );
            Ebu =   BCu ^((~BCa)&  BCe );

            Abo ^= Do;
            BCa = ROL(Abo, 28);
            Agu ^= Du;
            BCe = ROL(Agu, 20);
            Aka ^= Da;
            BCi = ROL(Aka,  3);
            Ame ^= De;
            BCo = ROL(Ame, 45);
            Asi ^= Di;
            BCu = ROL(Asi, 61);
            Ega =   BCa ^((~BCe)&  BCi );
            Ege =   BCe ^((~BCi)&  BCo );
            Egi =   BCi ^((~BCo)&  BCu );
            Ego =   BCo ^((~BCu)&  BCa );
            Egu =   BCu ^((~BCa)&  BCe );

            // Bloco 2
            Abe ^= De;
            BCa = ROL(Abe,  1);
            Agi ^= Di;
            BCe = ROL(Agi,  6);
            Ako ^= Do;
            BCi = ROL(Ako, 25);
            Amu ^= Du;
            BCo = ROL(Amu,  8);
            Asa ^= Da;
            BCu = ROL(Asa, 18);
            Eka =   BCa ^((~BCe)&  BCi );
            Eke =   BCe ^((~BCi)&  BCo );
            Eki =   BCi ^((~BCo)&  BCu );
            Eko =   BCo ^((~BCu)&  BCa );
            Eku =   BCu ^((~BCa)&  BCe );

            Abu ^= Du;
            BCa = ROL(Abu, 27);
            Aga ^= Da;
            BCe = ROL(Aga, 36);
            Ake ^= De;
            BCi = ROL(Ake, 10);
            Ami ^= Di;
            BCo = ROL(Ami, 15);
            Aso ^= Do;
            BCu = ROL(Aso, 56);
            Ema =   BCa ^((~BCe)&  BCi );
            Eme =   BCe ^((~BCi)&  BCo );
            Emi =   BCi ^((~BCo)&  BCu );
            Emo =   BCo ^((~BCu)&  BCa );
            Emu =   BCu ^((~BCa)&  BCe );

            // Bloco 3
            Abi ^= Di;
            BCa = ROL(Abi, 62);
            Ago ^= Do;
            BCe = ROL(Ago, 55);
            Aku ^= Du;
            BCi = ROL(Aku, 39);
            Ama ^= Da;
            BCo = ROL(Ama, 41);
            Ase ^= De;
            BCu = ROL(Ase,  2);
            Esa =   BCa ^((~BCe)&  BCi );
            Ese =   BCe ^((~BCi)&  BCo );
            Esi =   BCi ^((~BCo)&  BCu );
            Eso =   BCo ^((~BCu)&  BCa );
            Esu =   BCu ^((~BCa)&  BCe );

            //    prepareTheta
            BCa = Eba^Ega^Eka^Ema^Esa;
            BCe = Ebe^Ege^Eke^Eme^Ese;
            BCi = Ebi^Egi^Eki^Emi^Esi;
            BCo = Ebo^Ego^Eko^Emo^Eso;
            BCu = Ebu^Egu^Eku^Emu^Esu;

            //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            // Bloco 4
            Eba ^= Da;
            BCa = Eba;
            Ege ^= De;
            BCe = ROL(Ege, 44);
            Eki ^= Di;
            BCi = ROL(Eki, 43);
            Emo ^= Do;
            BCo = ROL(Emo, 21);
            Esu ^= Du;
            BCu = ROL(Esu, 14);
            Aba =   BCa ^((~BCe)&  BCi );
            Aba ^= (uint64_t)KeccakF_RoundConstants[round+1];
            Abe =   BCe ^((~BCi)&  BCo );
            Abi =   BCi ^((~BCo)&  BCu );
            Abo =   BCo ^((~BCu)&  BCa );
            Abu =   BCu ^((~BCa)&  BCe );

            Ebo ^= Do;
            BCa = ROL(Ebo, 28);
            Egu ^= Du;
            BCe = ROL(Egu, 20);
            Eka ^= Da;
            BCi = ROL(Eka, 3);
            Eme ^= De;
            BCo = ROL(Eme, 45);
            Esi ^= Di;
            BCu = ROL(Esi, 61);
            Aga =   BCa ^((~BCe)&  BCi );
            Age =   BCe ^((~BCi)&  BCo );
            Agi =   BCi ^((~BCo)&  BCu );
            Ago =   BCo ^((~BCu)&  BCa );
            Agu =   BCu ^((~BCa)&  BCe );

            // Bloco 5
            Ebe ^= De;
            BCa = ROL(Ebe, 1);
            Egi ^= Di;
            BCe = ROL(Egi, 6);
            Eko ^= Do;
            BCi = ROL(Eko, 25);
            Emu ^= Du;
            BCo = ROL(Emu, 8);
            Esa ^= Da;
            BCu = ROL(Esa, 18);
            Aka =   BCa ^((~BCe)&  BCi );
            Ake =   BCe ^((~BCi)&  BCo );
            Aki =   BCi ^((~BCo)&  BCu );
            Ako =   BCo ^((~BCu)&  BCa );
            Aku =   BCu ^((~BCa)&  BCe );
           
            Ebu ^= Du;
            BCa = ROL(Ebu, 27);
            Ega ^= Da;
            BCe = ROL(Ega, 36);
            Eke ^= De;
            BCi = ROL(Eke, 10);
            Emi ^= Di;
            BCo = ROL(Emi, 15);
            Eso ^= Do;
            BCu = ROL(Eso, 56);
            Ama =   BCa ^((~BCe)&  BCi );
            Ame =   BCe ^((~BCi)&  BCo );
            Ami =   BCi ^((~BCo)&  BCu );
            Amo =   BCo ^((~BCu)&  BCa );
            Amu =   BCu ^((~BCa)&  BCe );

             // Bloco 6
            Ebi ^= Di;
            BCa = ROL(Ebi, 62);
            Ego ^= Do;
            BCe = ROL(Ego, 55);
            Eku ^= Du;
            BCi = ROL(Eku, 39);
            Ema ^= Da;
            BCo = ROL(Ema, 41);
            Ese ^= De;
            BCu = ROL(Ese, 2);
            Asa =   BCa ^((~BCe)&  BCi ); 
            Ase =   BCe ^((~BCi)&  BCo );
            Asi =   BCi ^((~BCo)&  BCu );
            Aso =   BCo ^((~BCu)&  BCa );
            Asu =   BCu ^((~BCa)&  BCe );
        }

        //copyToState(state, A)
        state[ 0] = Aba;
        state[ 1] = Abe;
        state[ 2] = Abi;
        state[ 3] = Abo;
        state[ 4] = Abu;
        state[ 5] = Aga;
        state[ 6] = Age;
        state[ 7] = Agi;
        state[ 8] = Ago;
        state[ 9] = Agu;
        state[10] = Aka;
        state[11] = Ake;
        state[12] = Aki;
        state[13] = Ako;
        state[14] = Aku;
        state[15] = Ama;
        state[16] = Ame;
        state[17] = Ami;
        state[18] = Amo;
        state[19] = Amu;
        state[20] = Asa;
        state[21] = Ase;
        state[22] = Asi;
        state[23] = Aso;
        state[24] = Asu;
}


static void KeccakF1600_StatePermute3(uint64_t state[25])
{
        int round;

        uint64_t Aba, Abe, Abi, Abo, Abu;
        uint64_t Aga, Age, Agi, Ago, Agu;
        uint64_t Aka, Ake, Aki, Ako, Aku;
        uint64_t Ama, Ame, Ami, Amo, Amu;
        uint64_t Asa, Ase, Asi, Aso, Asu;
        uint64_t BCa, BCe, BCi, BCo, BCu;
        uint64_t Da, De, Di, Do, Du;
        uint64_t Eba, Ebe, Ebi, Ebo, Ebu;
        uint64_t Ega, Ege, Egi, Ego, Egu;
        uint64_t Eka, Eke, Eki, Eko, Eku;
        uint64_t Ema, Eme, Emi, Emo, Emu;
        uint64_t Esa, Ese, Esi, Eso, Esu;      

        //copyFromState(A, state)
        Aba = state[ 0];
        Abe = state[ 1];
        Abi = state[ 2];
        Abo = state[ 3];
        Abu = state[ 4];
        Aga = state[ 5];
        Age = state[ 6];
        Agi = state[ 7];
        Ago = state[ 8];
        Agu = state[ 9];
        Aka = state[10];
        Ake = state[11];
        Aki = state[12];
        Ako = state[13];
        Aku = state[14];
        Ama = state[15];
        Ame = state[16];
        Ami = state[17];
        Amo = state[18];
        Amu = state[19];
        Asa = state[20];
        Ase = state[21];
        Asi = state[22];
        Aso = state[23];
        Asu = state[24];

        for(round = 0; round < NROUNDS; round += 2) {
            //    prepareTheta
            BCa = Aba^Aga^Aka^Ama^Asa;
            BCe = Abe^Age^Ake^Ame^Ase;
            BCi = Abi^Agi^Aki^Ami^Asi;
            BCo = Abo^Ago^Ako^Amo^Aso;
            BCu = Abu^Agu^Aku^Amu^Asu;

            //thetaRhoPiChiIotaPrepareTheta(round, A, E)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);
       
            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars1[5] = {&Aba, &Age, &Aki, &Amo, &Asu};
            uint64_t* xor_vals1[5] = {&Da, &De, &Di, &Do, &Du};
            xor_parallel(vars1, xor_vals1);

            // Transferindo os valores para os registradores temporários
            // Aba já foi atualizado com Da
            BCe = ROL(Age, 44);                // Age atualizado com De
            BCi = ROL(Aki, 43);                // Aki atualizado com Di
            BCo = ROL(Amo, 21);                // Amo atualizado com Do
            BCu = ROL(Asu, 14);                // Asu atualizado com Du

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            uint64x2_t BCa_vec = vdupq_n_u64(BCa);
            uint64x2_t BCe_vec = vdupq_n_u64(BCe);
            uint64x2_t BCi_vec = vdupq_n_u64(BCi);
            uint64x2_t BCo_vec = vdupq_n_u64(BCo);
            uint64x2_t BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Eba, Ebe, Ebi, Ebo, Ebu
            process_parallel(&Eba, &Ebe, &Ebi, &Ebo, &Ebu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);
            Eba ^= (uint64_t)KeccakF_RoundConstants[round];
            
            // Aplicação dos XORs em paralelo para o segundo conjunto de variáveis
            uint64_t* vars2[5] = {&Abo, &Agu, &Aka, &Ame, &Asi};
            uint64_t* xor_vals2[5] = {&Do, &Du, &Da, &De, &Di};
            xor_parallel(vars2, xor_vals2);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Abo, 28);                // Abo atualizado com Do
            BCe = ROL(Agu, 20);                // Agu atualizado com Du
            BCi = ROL(Aka, 3);                 // Aka atualizado com Da
            BCo = ROL(Ame, 45);                // Ame atualizado com De
            BCu = ROL(Asi, 61);                // Asi atualizado com Di

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Ega, Ege, Egi, Ego, Egu
            process_parallel(&Ega, &Ege, &Egi, &Ego, &Egu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Bloco 2 otimizado com NEON

            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars3[5] = {&Abe, &Agi, &Ako, &Amu, &Asa};
            uint64_t* xor_vals3[5] = {&De, &Di, &Do, &Du, &Da};
            xor_parallel(vars3, xor_vals3);

            // Transferindo os valores para os registradores temporários
            BCa = ROL(Abe, 1);                // Abe atualizado com De
            BCe = ROL(Agi, 6);                // Agi atualizado com Di
            BCi = ROL(Ako, 25);               // Ako atualizado com Do
            BCo = ROL(Amu, 8);                // Amu atualizado com Du
            BCu = ROL(Asa, 18);               // Asa atualizado com Da

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Eka, Eke, Eki, Eko, Eku
            process_parallel(&Eka, &Eke, &Eki, &Eko, &Eku, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Aplicação dos XORs em paralelo para o segundo conjunto de variáveis
            uint64_t* vars4[5] = {&Abu, &Aga, &Ake, &Ami, &Aso};
            uint64_t* xor_vals4[5] = {&Du, &Da, &De, &Di, &Do};
            xor_parallel(vars4, xor_vals4);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Abu, 27);               // Abu atualizado com Du
            BCe = ROL(Aga, 36);               // Aga atualizado com Da
            BCi = ROL(Ake, 10);               // Ake atualizado com De
            BCo = ROL(Ami, 15);               // Ami atualizado com Di
            BCu = ROL(Aso, 56);               // Aso atualizado com Do

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Ema, Eme, Emi, Emo, Emu
            process_parallel(&Ema, &Eme, &Emi, &Emo, &Emu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);


            // Bloco 3 otimizado com NEON

            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars5[5] = {&Abi, &Ago, &Aku, &Ama, &Ase};
            uint64_t* xor_vals5[5] = {&Di, &Do, &Du, &Da, &De};
            xor_parallel(vars5, xor_vals5);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Abi, 62);               // Abi atualizado com Di
            BCe = ROL(Ago, 55);               // Ago atualizado com Do
            BCi = ROL(Aku, 39);               // Aku atualizado com Du
            BCo = ROL(Ama, 41);               // Ama atualizado com Da
            BCu = ROL(Ase, 2);                // Ase atualizado com De

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Esa, Ese, Esi, Eso, Esu
            process_parallel(&Esa, &Ese, &Esi, &Eso, &Esu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);


            //    prepareTheta
            BCa = Eba^Ega^Eka^Ema^Esa;
            BCe = Ebe^Ege^Eke^Eme^Ese;
            BCi = Ebi^Egi^Eki^Emi^Esi;
            BCo = Ebo^Ego^Eko^Emo^Eso;
            BCu = Ebu^Egu^Eku^Emu^Esu;

            //thetaRhoPiChiIotaPrepareTheta(round+1, E, A)
            Da = BCu^ROL(BCe, 1);
            De = BCa^ROL(BCi, 1);
            Di = BCe^ROL(BCo, 1);
            Do = BCi^ROL(BCu, 1);
            Du = BCo^ROL(BCa, 1);

            // Bloco 4 otimizado com NEON

            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars6[5] = {&Eba, &Ege, &Eki, &Emo, &Esu};
            uint64_t* xor_vals6[5] = {&Da, &De, &Di, &Do, &Du};
            xor_parallel(vars6, xor_vals6);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = Eba;                        // Eba atualizado com Da
            BCe = ROL(Ege, 44);               // Ege atualizado com De
            BCi = ROL(Eki, 43);               // Eki atualizado com Di
            BCo = ROL(Emo, 21);               // Emo atualizado com Do
            BCu = ROL(Esu, 14);               // Esu atualizado com Du

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);


            // Aplicação do passo "Chi" em paralelo para Aba, Abe, Abi, Abo, Abu
            process_parallel(&Aba, &Abe, &Abi, &Abo, &Abu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Aplicação da constante de rodada no Aba
            Aba ^= (uint64_t)KeccakF_RoundConstants[round+1];

            // Aplicação dos XORs em paralelo usando NEON (segunda parte do bloco)
            uint64_t* vars7[5] = {&Ebo, &Egu, &Eka, &Eme, &Esi};
            uint64_t* xor_vals7[5] = {&Do, &Du, &Da, &De, &Di};
            xor_parallel(vars7, xor_vals7);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Ebo, 28);               // Ebo atualizado com Do
            BCe = ROL(Egu, 20);               // Egu atualizado com Du
            BCi = ROL(Eka, 3);                // Eka atualizado com Da
            BCo = ROL(Eme, 45);               // Eme atualizado com De
            BCu = ROL(Esi, 61);               // Esi atualizado com Di

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);

            // Aplicação do passo "Chi" em paralelo para Aga, Age, Agi, Ago, Agu
            process_parallel(&Aga, &Age, &Agi, &Ago, &Agu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Bloco 5 otimizado com NEON

            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars8[5] = {&Ebe, &Egi, &Eko, &Emu, &Esa};
            uint64_t* xor_vals8[5] = {&De, &Di, &Do, &Du, &Da};
            xor_parallel(vars8, xor_vals1);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Ebe, 1);                // Ebe atualizado com De
            BCe = ROL(Egi, 6);                // Egi atualizado com Di
            BCi = ROL(Eko, 25);               // Eko atualizado com Do
            BCo = ROL(Emu, 8);                // Emu atualizado com Du
            BCu = ROL(Esa, 18);               // Esa atualizado com Da

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);

            // Aplicação do passo "Chi" em paralelo para Aka, Ake, Aki, Ako, Aku
            process_parallel(&Aka, &Ake, &Aki, &Ako, &Aku, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Aplicação dos XORs em paralelo usando NEON (segunda parte do bloco)
            uint64_t* vars9[5] = {&Ebu, &Ega, &Eke, &Emi, &Eso};
            uint64_t* xor_vals9[5] = {&Du, &Da, &De, &Di, &Do};
            xor_parallel(vars2, xor_vals2);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Ebu, 27);               // Ebu atualizado com Du
            BCe = ROL(Ega, 36);               // Ega atualizado com Da
            BCi = ROL(Eke, 10);               // Eke atualizado com De
            BCo = ROL(Emi, 15);               // Emi atualizado com Di
            BCu = ROL(Eso, 56);               // Eso atualizado com Do

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);
          
            // Aplicação do passo "Chi" em paralelo para Ama, Ame, Ami, Amo, Amu
            process_parallel(&Ama, &Ame, &Ami, &Amo, &Amu,BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

            // Bloco 6 otimizado com NEON

            // Aplicação dos XORs em paralelo usando NEON
            uint64_t* vars10[5] = {&Ebi, &Ego, &Eku, &Ema, &Ese};
            uint64_t* xor_vals10[5] = {&Di, &Do, &Du, &Da, &De};
            xor_parallel(vars1, xor_vals1);

            // Transferindo os valores para os registradores temporários após os XORs
            BCa = ROL(Ebi, 62);               // Ebi atualizado com Di
            BCe = ROL(Ego, 55);               // Ego atualizado com Do
            BCi = ROL(Eku, 39);               // Eku atualizado com Du
            BCo = ROL(Ema, 41);               // Ema atualizado com Da
            BCu = ROL(Ese, 2);                // Ese atualizado com De

            // Converter as variáveis BCa, BCe, BCi, BCo, BCu para vetores NEON de 64 bits antes de chamar a função
            BCa_vec = vdupq_n_u64(BCa);
            BCe_vec = vdupq_n_u64(BCe);
            BCi_vec = vdupq_n_u64(BCi);
            BCo_vec = vdupq_n_u64(BCo);
            BCu_vec = vdupq_n_u64(BCu);

            // Aplicação do passo "Chi" em paralelo para Asa, Ase, Asi, Aso, Asu
            process_parallel(&Asa, &Ase, &Asi, &Aso, &Asu, BCa_vec, BCe_vec, BCi_vec, BCo_vec, BCu_vec);

        }

        //copyToState(state, A)
        state[ 0] = Aba;
        state[ 1] = Abe;
        state[ 2] = Abi;
        state[ 3] = Abo;
        state[ 4] = Abu;
        state[ 5] = Aga;
        state[ 6] = Age;
        state[ 7] = Agi;
        state[ 8] = Ago;
        state[ 9] = Agu;
        state[10] = Aka;
        state[11] = Ake;
        state[12] = Aki;
        state[13] = Ako;
        state[14] = Aku;
        state[15] = Ama;
        state[16] = Ame;
        state[17] = Ami;
        state[18] = Amo;
        state[19] = Amu;
        state[20] = Asa;
        state[21] = Ase;
        state[22] = Asi;
        state[23] = Aso;
        state[24] = Asu;
}





// Função de medição de ciclos usando clock()
uint64_t measure_cycles(void (*func)(uint64_t*), uint64_t *state) {
    clock_t start, end;
    start = clock();
    func(state);
    end = clock();
    return (uint64_t)(end - start);
}

// Função de comparação de estados
int compare_states(uint64_t *state_ref, uint64_t *state_neon) {
    for (int i = 0; i < 25; i++) {
        if (state_ref[i] != state_neon[i]) {
            printf("Divergência encontrada no estado[%d]: ref = %llu, neon = %llu\n", i, state_ref[i], state_neon[i]);
            return 0; // Divergência encontrada
        }
    }
    return 1; // Saídas iguais
}



// Função principal
int main() {
     // Teste da função de referência
   //  test_comparison();
   
    //test_comparison2();

    uint64_t state_ref[25];
    uint64_t state_neon[25];
    
    // Inicializar o estado com valores aleatórios para testar
    for (int i = 0; i < 25; i++) {
        state_ref[i] = (uint64_t)rand() * (uint64_t)rand(); // Inicializando com números aleatórios
    }
    memcpy(state_neon, state_ref, sizeof(state_ref)); // Copiar o estado para o teste NEON

    // Medir o tempo de execução da função de referência
    printf("Medindo o tempo de execução da função de referência...\n");
    uint64_t cycles_ref = measure_cycles(KeccakF1600_StatePermute, state_ref);

    // Medir o tempo de execução da função otimizada com NEON
    printf("Medindo o tempo de execução da função otimizada com NEON...\n");
    uint64_t cycles_neon = measure_cycles(KeccakF1600_StatePermute2, state_neon);

    // Comparar as saídas
    int correct = compare_states(state_ref, state_neon);

    // Mostrar resultados
    if (correct) {
        printf("Saídas são iguais!\n");
    } else {
        printf("Saídas são diferentes!\n");
    }

    printf("Ciclos da função de referência: %llu\n", cycles_ref);
    printf("Ciclos da função otimizada com NEON: %llu\n", cycles_neon);
    if (cycles_neon > 0) {
        printf("Speed-up: %.2f X\n", (double)cycles_ref / cycles_neon);
    }

    /*
    // Executar o teste várias vezes para coletar uma média (opcional)
    uint64_t total_cycles_ref = 0, total_cycles_neon = 0;
    for (int i = 0; i < NUM_TESTES; i++) {
        // Reinicializar os estados para cada teste
        for (int j = 0; j < 25; j++) {
            state_ref[j] = (uint64_t)rand() * (uint64_t)rand();
        }
        memcpy(state_neon, state_ref, sizeof(state_ref));

        // Medir os ciclos
        total_cycles_ref += measure_cycles(KeccakF1600_StatePermute, state_ref);
        total_cycles_neon += measure_cycles(KeccakF1600_StatePermute2, state_neon);
    }

    printf("\nApós %d execuções:\n", NUM_TESTES);
    printf("Ciclos médios da função de referência: %llu\n", total_cycles_ref / NUM_TESTES);
    printf("Ciclos médios da função otimizada com NEON: %llu\n", total_cycles_neon / NUM_TESTES);
    if (total_cycles_neon > 0) {
        printf("Speed-up médio: %.2f X\n", (double)total_cycles_ref / total_cycles_neon);
    }
    */

    return 0;
}