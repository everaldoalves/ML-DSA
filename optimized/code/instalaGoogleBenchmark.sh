#!/bin/bash

# Atualiza os repositórios e instala g++
apt update
apt install -y g++

# Instala dependências do Google Benchmark
apt install -y cmake git

# Clona o repositório do Google Benchmark
git clone https://github.com/google/benchmark.git
cd benchmark

# Clona as dependências de benchmarking do Google
git clone https://github.com/google/googletest.git

# Cria o diretório build
mkdir build
cd build

# Compila e instala o Google Benchmark
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install

# Finalizado
echo "Instalação do g++ e Google Benchmark concluída com sucesso!"
