#include <stdio.h>
#include "cpucycles.h"

int main() {
    uint64_t start, end;
    start = cpucycles();
    
    for (volatile int i = 0; i < 1000000; ++i) {}  // Loop vazio

    end = cpucycles();
    printf("Ciclos: %llu\n", end - start);
    return 0;
}