#!/bin/bash

# Array com as versões do Dilithium
DILITHIUM_VERSIONS=(2 3 5)

# Loop para compilar e executar para cada versão
for VERSION in "${DILITHIUM_VERSIONS[@]}"
do
    # Quebra de linha antes da mensagem
    echo -e "\n\nCompilando e executando benchmark para Dilithium versão $VERSION\n"

    # Definir o DILITHIUM_MODE, compilar e suprimir warnings com a flag -w
    g++ -O3 -w -std=c++11 -DDILITHIUM_MODE=$VERSION -I /opt/homebrew/include test/googleBenchmarkDilithiun.cpp sign.c poly.c polyvec.c randombytes.c ntt.c reduce.c fips202.c fips202x2.c packing.c rounding.c symmetric-shake.c feat.S -L /opt/homebrew/lib -lbenchmark -lpthread -o test/googleBenchmarkDilithiun_mode$VERSION

    # Executar o benchmark
    ./test/googleBenchmarkDilithiun_mode$VERSION

    # Quebra de linha após a execução
    echo -e "\n"
done








    

   






