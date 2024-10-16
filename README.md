<div style="background-color: white; padding: 10px; display: inline-block;">

<img src="https://github.com/user-attachments/assets/214dfa1b-5536-4cdc-9df4-acef1aff5e7f" alt="Imagem de Fundo" width="200" height="auto">

# ML-DSA

Implementação do esquema de assinatura digital pós-quântico Module-Lattice-Based Digital Signature Standard - ML-DSA (FIPS 204) para plataforma ARMv8-A.

## Instruções de compilação
A implementação contêm programas de teste e benchmarking e um Makefile para facilitar a compilação.

### Pré-requisitos

Alguns dos programas de teste requerem o OpenSSL. Se os arquivos de cabeçalho e/ou bibliotecas compartilhadas do OpenSSL não estiverem em um dos locais padrão em seu sistema, é necessário especificar seu local através de flags do compilador e linker nas variáveis de ambiente CFLAGS, NISTFLAGS e LDFLAGS.

Por exemplo, no macOS você pode instalar o OpenSSL via Homebrew executando

```brew install openssl```

Em seguida, execute:

```export CFLAGS="-I/opt/homebrew/opt/openssl@1.1/include"```

```export NISTFLAGS="-I/opt/homebrew/opt/openssl@1.1/include"```

```export LDFLAGS="-L/opt/homebrew/opt/openssl@1.1/lib"```

antes da compilação para adicionar os locais dos cabeçalhos e bibliotecas OpenSSL aos respectivos caminhos de busca.

## Programas de Teste
Para compilar os programas de teste no Linux ou macOS, vá para o diretório ref/ e execute:

```make```

Isso produz o executável:

```test/test_dilithium$ALG```

onde $ALG varia sobre os conjuntos de parâmetros 2, 3 e 5.

test/test_dilithium$ALG testa 10.000 vezes a geração de chaves, assinatura de uma mensagem aleatória de 59 bytes e verificação da assinatura produzida. Além disso, o programa tentará verificar assinaturas incorretas onde um único byte aleatório de uma assinatura válida foi distorcido aleatoriamente. O programa abortará com uma mensagem de erro e retornará -1 nesta situação. Caso contrário, ele exibirá os tamanhos da chave e da assinatura e retornará 0.

Também é possível verificar a assertividade da implementação com o script testaDilithium.sh. Este script realizará testes de geração de chaves, assinatura e verificação exibindo os resultados para cada uma das versões do esquema.

## Programas de Benchmarking
Para realizar o benchmarking da implementação, estão disponíveis os programas de teste de velocidade para CPUs x86 que usam o Time Step Counter (TSC) ou o contador de ciclos real fornecido pelos Performance Measurement Counters (PMC) para medir o desempenho. Para compilar os programas, execute:

```make speed```

Isso produz os executáveis:

```test/test_speed$ALG```

para todos os conjuntos de parâmetros $ALG mencionados anteriormente. Os programas relatam as contagens de ciclos medianas e médias de 10.000 execuções de várias funções internas e das funções da API para geração de chaves, assinatura e verificação. Por padrão, o Time Step Counter é usado. Se você quiser obter as contagens de ciclos reais dos Performance Measurement Counters, exporte CFLAGS="-DUSE_RDPMC" antes da compilação.

Também é possível realizar o benchmark de maneira mais simples com o emprego do script googleBenchmark.sh. Para utilizá-lo, basta acessar a pasta test e executar o comando:

```./googleBenchmark.sh```


</div>
