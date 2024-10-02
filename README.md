<div style="background-color: white; padding: 10px; display: inline-block;">

<img src="https://github.com/user-attachments/assets/214dfa1b-5536-4cdc-9df4-acef1aff5e7f" alt="Imagem de Fundo" width="200" height="auto">

# ML-DSA

Implementação do esquema de assinatura digital pós-quântico ML-DSA (FIPS 204) para plataforma ARMv8-A.

Instruções de compilação
A implementação contêm vários programas de teste e benchmarking e um Makefile para facilitar a compilação.

Pré-requisitos
Alguns dos programas de teste requerem o OpenSSL. Se os arquivos de cabeçalho e/ou bibliotecas compartilhadas do OpenSSL não estiverem em um dos locais padrão em seu sistema, é necessário especificar seu local através de flags do compilador e linker nas variáveis de ambiente CFLAGS, NISTFLAGS e LDFLAGS.

Por exemplo, no macOS você pode instalar o OpenSSL via Homebrew executando
'brew install openssl'

Em seguida, execute:
'export CFLAGS="-I/opt/homebrew/opt/openssl@1.1/include"
export NISTFLAGS="-I/opt/homebrew/opt/openssl@1.1/include"
export LDFLAGS="-L/opt/homebrew/opt/openssl@1.1/lib"'


</div>
