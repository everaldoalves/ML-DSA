<div style="background-color: white; padding: 10px; display: inline-block;">

<img src="https://github.com/user-attachments/assets/214dfa1b-5536-4cdc-9df4-acef1aff5e7f" alt="Imagem de Fundo" width="200" height="auto">

# ML-DSA

Implementação do esquema de assinatura digital pós-quântico Module-Lattice-Based Digital Signature Standard - ML-DSA (FIPS 204) para plataforma ARMv8-A.

## Instruções de compilação
A implementação contêm programas de teste e benchmarking e um Makefile para facilitar a compilação.

### Pré-requisitos

Alguns dos programas de teste requerem o OpenSSL. Se os arquivos de cabeçalho e/ou bibliotecas compartilhadas do OpenSSL não estiverem em um dos locais padrão em seu sistema, é necessário especificar seu local através de flags do compilador e linker nas variáveis de ambiente CFLAGS, NISTFLAGS e LDFLAGS.

Por exemplo, no macOS você pode instalar o OpenSSL via Homebrew executando

```sh 
brew install openssl
```

Em seguida, execute:

```sh
export CFLAGS="-I/opt/homebrew/opt/openssl@1.1/include

export NISTFLAGS="-I/opt/homebrew/opt/openssl@1.1/include

export LDFLAGS="-L/opt/homebrew/opt/openssl@1.1/lib
```

antes da compilação para adicionar os locais dos cabeçalhos e bibliotecas OpenSSL aos respectivos caminhos de busca.

## Programas de Teste
Para compilar os programas de teste no Linux ou macOS, vá para o diretório code/ e execute:

```sh
make
```

Isso produz o executável:

```sh
test/test_dilithium$ALG
```

onde $ALG varia sobre os conjuntos de parâmetros 2, 3 e 5.

test/test_dilithium$ALG testa 10.000 vezes a geração de chaves, assinatura de uma mensagem aleatória de 59 bytes e verificação da assinatura produzida. Além disso, o programa tentará verificar assinaturas incorretas onde um único byte aleatório de uma assinatura válida foi distorcido aleatoriamente. O programa abortará com uma mensagem de erro e retornará -1 nesta situação. Caso contrário, ele exibirá os tamanhos da chave e da assinatura e retornará 0.

Também é possível verificar a assertividade da implementação com o script testaDilithium.sh. Este script realizará testes de geração de chaves, assinatura e verificação exibindo os resultados para cada uma das versões do esquema.

## Programas de Benchmarking

Para realizar o benchmark de maneira simples utilize o script googleBenchmark.sh. Para isso, primeiro você precisa instalar o Google Benchmark. O script instalaGoogleBenchmark.sh pode ser usado para auxiliá-lo nessa tarefa. Com a instalação realizada, basta acessar a pasta test e executar o comando:

```sh
./googleBenchmark.sh
```

Também estão disponíveis os programas de teste de velocidade para CPUs que usam o contador de ciclos real fornecido pelos Performance Measurement Counters (PMC) para medir o desempenho. Para compilar os programas, execute:

```sh
make speed
```

Isso produz os executáveis:

```sh
test/test_speed$ALG
```

para todos os conjuntos de parâmetros $ALG mencionados anteriormente. Os programas relatam as contagens de ciclos medianas e médias de 10.000 execuções de várias funções internas e das funções da API para geração de chaves, assinatura e verificação. Por padrão, o Time Step Counter é usado. Se você quiser obter as contagens de ciclos reais dos Performance Measurement Counters, exporte CFLAGS="-DUSE_RDPMC" antes da compilação.

## Resultados
As tabelas a seguir apresentam os resultados alcançados comparando os ciclos da implementação de referência [1] e deste trabalho para os três níveis de segurança do ML-DSA. Os experimentos para avaliação do desempenho foram realizados em dois dispositivos da Apple, o MacBook Air com o chip Apple M1 (8GB RAM) e o MacBook Air com o chip Apple M2 (8GB RAM), que possuem arquitetura ARMv8 com suporte a instruções NEON. O compilador utilizado foi o Clang 18.1.8 e o sistema operacional o MacOS Sonoma 14.4 no M1 e 14.6 no M2.

## Apple M1

| Versão     | Algoritmo | Impl. Ref. | Este Trabalho | Aceleração (x) |
|------------|-----------|------------|---------------|----------------|
| ML-DSA-44  | KeyGen    | 1282       | 613           | 2.09           |
|            | Sign      | 5937       | 2518          | 2.36           |
|            | Verify    | 1418       | 659           | 2.15           |
| ML-DSA-65  | KeyGen    | 2553       | 997           | 2.56           |
|            | Sign      | 10964      | 4110          | 2.67           |
|            | Verify    | 2472       | 1081          | 2.29           |
| ML-DSA-87  | KeyGen    | 3515       | 1646          | 2.14           |
|            | Sign      | 12290      | 4739          | 2.59           |
|            | Verify    | 3708       | 1666          | 2.23           |

*Tabela: Contagens de ciclos no Apple M1 para os três níveis de segurança do ML-DSA em comparação com [1]*

## Apple M2

| Versão     | Algoritmo | Impl. Ref. | Este Trabalho | Aceleração (x) |
|------------|-----------|------------|---------------|----------------|
| ML-DSA-44  | KeyGen    | 1134       | 565           | 2.00           |
|            | Sign      | 5348       | 2333          | 2.29           |
|            | Verify    | 1252       | 610           | 2.05           |
| ML-DSA-65  | KeyGen    | 2298       | 926           | 2.48           |
|            | Sign      | 10044      | 3801          | 2.64           |
|            | Verify    | 2222       | 1008          | 2.20           |
| ML-DSA-87  | KeyGen    | 3115       | 1550          | 2.00           |
|            | Sign      | 10338      | 4490          | 2.30           |
|            | Verify    | 3286       | 1570          | 2.09           |


*Tabela: Contagens de ciclos no Apple M2 para os três níveis de segurança do ML-DSA em comparação com [1]*


[1] Ducas, L. et al. (2021). CRYSTALS-Dilithium (round 3)
</div>
