//************************************************************************************************
// Autor: Everaldo Alves
// Data: 20 de Setembro/2024
// Função: load, store de FIPS202.c
// Descrição: Funções LOAD e STORE do FIPS 202
// Objetivo: Comparar implementação de referência com uma versão otimizada para ARMv8 usando NEON
// Situação atual: 
//************************************************************************************************

#include <arm_neon.h>
#include <stdint.h>
#include <string.h>

#define NROUNDS 24
#define ROL(a, offset) ((a << offset) ^ (a >> (64-offset)))


/*************************************************
* Name:        load64
*
* Description: Load 8 bytes into uint64_t in little-endian order
*
* Arguments:   - const uint8_t *x: pointer to input byte array
*
* Returns the loaded 64-bit unsigned integer
**************************************************/
static uint64_t load64(const uint8_t x[8]) {
  unsigned int i;
  uint64_t r = 0;

  for(i=0;i<8;i++)
    r |= (uint64_t)x[i] << 8*i;

  return r;
}

/*************************************************
* Name:        store64
*
* Description: Store a 64-bit integer to array of 8 bytes in little-endian order
*
* Arguments:   - uint8_t *x: pointer to the output byte array (allocated)
*              - uint64_t u: input 64-bit unsigned integer
**************************************************/
static void store64(uint8_t x[8], uint64_t u) {
  unsigned int i;

  for(i=0;i<8;i++)
    x[i] = u >> 8*i;
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

//****************************************************************************************************************************************************************
// Funções otimizadas para ARMv8 usando NEON
//****************************************************************************************************************************************************************
// Carrega 64 bytes de uma vez
static uint64_t load64_opt(const uint8_t x[8]) {
    uint64x1_t v = vld1_u64((const uint64_t *)x);
    return vget_lane_u64(v, 0);
}

// Armazena 64 bytes de uma vez
static void store64_opt(uint8_t x[8], uint64_t u) {
    uint64x1_t v = vdup_n_u64(u);
    vst1_u64((uint64_t *)x, v);
}

void KeccakF1600_StatePermute_neon(uint64_t state[25]) {
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
    uint64x2_t temp_low, temp_high;
    

    for (round = 0; round < NROUNDS; round += 2) {
        // Theta Step
        BCa = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        BCe = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        BCi = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        BCo = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        BCu = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        Da = vdupq_n_u64(BCu ^ ROL64(BCe, 1));
        De = vdupq_n_u64(BCa ^ ROL64(BCi, 1));
        Di = vdupq_n_u64(BCe ^ ROL64(BCo, 1));
        Do = vdupq_n_u64(BCi ^ ROL64(BCu, 1));
        Du = vdupq_n_u64(BCo ^ ROL64(BCa, 1));

        // Aplicar Theta com NEON
        temp_low = vld1q_u64(&state[0]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[0], temp_low);

        temp_low = vld1q_u64(&state[5]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[5], temp_low);

        temp_low = vld1q_u64(&state[10]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[10], temp_low);

        temp_low = vld1q_u64(&state[15]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[15], temp_low);

        temp_low = vld1q_u64(&state[20]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[20], temp_low);

        // Rho Step
        Aba = state[0];
        Age = ROL64(state[6], 44);
        Aki = ROL64(state[12], 43);
        Amo = ROL64(state[18], 21);
        Asu = ROL64(state[24], 14);

        Abe = ROL64(state[1], 1);
        Agi = ROL64(state[7], 6);
        Ako = ROL64(state[13], 25);
        Amu = ROL64(state[19], 8);
        Asa = ROL64(state[20], 18);

        // Pi Step
        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        // Chi Step
        state[0] = Eba ^ ((~Ebe) & Ebi);
        state[1] = Ebe ^ ((~Ebi) & Ebo);
        state[2] = Ebi ^ ((~Ebo) & Ebu);
        state[3] = Ebo ^ ((~Ebu) & Eba);
        state[4] = Ebu ^ ((~Eba) & Ebe);

        // Continuar para o próximo grupo de 5 estados
        Aba = state[5];
        Age = ROL64(state[11], 44);
        Aki = ROL64(state[17], 43);
        Amo = ROL64(state[23], 21);
        Asu = ROL64(state[4], 14);

        // Pi Step para esse bloco
        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        // Chi Step para esse bloco
        state[5] = Eba ^ ((~Ebe) & Ebi);
        state[6] = Ebe ^ ((~Ebi) & Ebo);
        state[7] = Ebi ^ ((~Ebo) & Ebu);
        state[8] = Ebo ^ ((~Ebu) & Eba);
        state[9] = Ebu ^ ((~Eba) & Ebe);

        // Iota Step (Adicionar constante da rodada)
        state[0] ^= KeccakF_RoundConstants[round];

        // Aplicar a segunda rodada (round + 1)
        BCa = state[0] ^ state[5] ^ state[10] ^ state[15] ^ state[20];
        BCe = state[1] ^ state[6] ^ state[11] ^ state[16] ^ state[21];
        BCi = state[2] ^ state[7] ^ state[12] ^ state[17] ^ state[22];
        BCo = state[3] ^ state[8] ^ state[13] ^ state[18] ^ state[23];
        BCu = state[4] ^ state[9] ^ state[14] ^ state[19] ^ state[24];

        // Aplicando Theta para a segunda rodada
        Da = vdupq_n_u64(BCu ^ ROL64(BCe, 1));
        De = vdupq_n_u64(BCa ^ ROL64(BCi, 1));
        Di = vdupq_n_u64(BCe ^ ROL64(BCo, 1));
        Do = vdupq_n_u64(BCi ^ ROL64(BCu, 1));
        Du = vdupq_n_u64(BCo ^ ROL64(BCa, 1));

        // Aplicar as operações de XOR da segunda rodada
        temp_low = vld1q_u64(&state[0]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[0], temp_low);

        temp_low = vld1q_u64(&state[5]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[5], temp_low);

        temp_low = vld1q_u64(&state[10]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[10], temp_low);

        temp_low = vld1q_u64(&state[15]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[15], temp_low);

        temp_low = vld1q_u64(&state[20]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[20], temp_low);

        // Rho, Pi e Chi para a segunda rodada
        Aba = state[0];
        Age = ROL64(state[6], 44);
        Aki = ROL64(state[12], 43);
        Amo = ROL64(state[18], 21);
        Asu = ROL64(state[24], 14);

        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        state[0] = Eba ^ ((~Ebe) & Ebi);
        state[1] = Ebe ^ ((~Ebi) & Ebo);
        state[2] = Ebi ^ ((~Ebo) & Ebu);
        state[3] = Ebo ^ ((~Ebu) & Eba);
        state[4] = Ebu ^ ((~Eba) & Ebe);

        // Segunda parte de Iota: Adicionar a constante da segunda rodada
        state[0] ^= KeccakF_RoundConstants[round + 1];

        // Repetir as etapas de Theta, Rho, Pi e Chi para o restante do estado

        // Atualizar os blocos de 5 estados restantes usando as operações Rho e Pi
        Aba = state[5];
        Age = ROL64(state[11], 44);
        Aki = ROL64(state[17], 43);
        Amo = ROL64(state[23], 21);
        Asu = ROL64(state[4], 14);

        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        state[5] = Eba ^ ((~Ebe) & Ebi);
        state[6] = Ebe ^ ((~Ebi) & Ebo);
        state[7] = Ebi ^ ((~Ebo) & Ebu);
        state[8] = Ebo ^ ((~Ebu) & Eba);
        state[9] = Ebu ^ ((~Eba) & Ebe);

        // Theta Step: Calcular o XOR de todas as fatias restantes de 64 bits
        BCa = state[10] ^ state[15] ^ state[20] ^ state[0];
        BCe = state[11] ^ state[16] ^ state[21] ^ state[1];
        BCi = state[12] ^ state[17] ^ state[22] ^ state[2];
        BCo = state[13] ^ state[18] ^ state[23] ^ state[3];
        BCu = state[14] ^ state[19] ^ state[24] ^ state[4];

        // Executar o segundo conjunto de operações Theta com NEON
        Da = vdupq_n_u64(BCu ^ ROL64(BCe, 1));
        De = vdupq_n_u64(BCa ^ ROL64(BCi, 1));
        Di = vdupq_n_u64(BCe ^ ROL64(BCo, 1));
        Do = vdupq_n_u64(BCi ^ ROL64(BCu, 1));
        Du = vdupq_n_u64(BCo ^ ROL64(BCa, 1));

        temp_low = vld1q_u64(&state[10]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[10], temp_low);

        temp_low = vld1q_u64(&state[15]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[15], temp_low);

        temp_low = vld1q_u64(&state[20]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[20], temp_low);

        temp_low = vld1q_u64(&state[0]);
        temp_low = veorq_u64(temp_low, Da);
        vst1q_u64(&state[0], temp_low);

        // Rho, Pi e Chi Step: Aplicar as rotações, permutações e máscaras no restante dos estados
        Aba = state[10];
        Age = ROL64(state[11], 44);
        Aki = ROL64(state[12], 43);
        Amo = ROL64(state[13], 21);
        Asu = ROL64(state[14], 14);

        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        state[10] = Eba ^ ((~Ebe) & Ebi);
        state[11] = Ebe ^ ((~Ebi) & Ebo);
        state[12] = Ebi ^ ((~Ebo) & Ebu);
        state[13] = Ebo ^ ((~Ebu) & Eba);
        state[14] = Ebu ^ ((~Eba) & Ebe);

        Aba = state[15];
        Age = ROL64(state[16], 44);
        Aki = ROL64(state[17], 43);
        Amo = ROL64(state[18], 21);
        Asu = ROL64(state[19], 14);

        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        state[15] = Eba ^ ((~Ebe) & Ebi);
        state[16] = Ebe ^ ((~Ebi) & Ebo);
        state[17] = Ebi ^ ((~Ebo) & Ebu);
        state[18] = Ebo ^ ((~Ebu) & Eba);
        state[19] = Ebu ^ ((~Eba) & Ebe);

        Aba = state[20];
        Age = ROL64(state[21], 44);
        Aki = ROL64(state[22], 43);
        Amo = ROL64(state[23], 21);
        Asu = ROL64(state[24], 14);

        Eba = Aba;
        Ebe = Age;
        Ebi = Aki;
        Ebo = Amo;
        Ebu = Asu;

        state[20] = Eba ^ ((~Ebe) & Ebi);
        state[21] = Ebe ^ ((~Ebi) & Ebo);
        state[22] = Ebi ^ ((~Ebo) & Ebu);
        state[23] = Ebo ^ ((~Ebu) & Eba);
        state[24] = Ebu ^ ((~Eba) & Ebe);

        // Aplicar a constante final de Iota
        state[0] ^= KeccakF_RoundConstants[round + 1];

        // Continuar com a cópia final para o estado após a última rodada
        // copyToState(state, A)
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
}

static void keccak_init(uint64_t s[25])
{
    memset(s, 0, 25 * sizeof(uint64_t));
}
