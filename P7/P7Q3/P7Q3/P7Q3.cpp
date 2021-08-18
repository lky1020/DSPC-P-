// P7Q3.cpp : Defines the entry point for the console application.
//

#include "stdio.h"


int main()
{
    int a = 1;
    int b = 2;
    int c = 4;

#pragma omp parallel 
    {
        a = a + b;
        printf("%i\n", a);
        printf("%i\n", b);
        printf("%i\n", c);
    }

    printf("Outside A: %i\n", a);

    return 0;
}