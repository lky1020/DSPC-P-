#include "stdio.h"
#include "omp.h"
#include <stdlib.h>

const int N = 100000;

void fill_rand(int nval, double* A) {
    for (int i = 0; i < nval; i++) A[i] = (double)rand();
}

double Sum_array(int nval, double* A) {
    double sum = 0.0;
    for (int i = 0; i < nval; i++) sum = sum + A[i];
    return sum;
}


int main()
{
    double* A, sum, runtime;	int flag = 0;
    A = (double*)malloc(N * sizeof(double));
    runtime = omp_get_wtime();
    //The section construct is one way to distribute different tasks to different threads.
#pragma omp parallel sections
    {
#pragma omp section
        {
            fill_rand(N, A);
#pragma omp flush  
            flag = 1;
#pragma omp flush (flag)
        }
#pragma omp section
        {
#pragma omp flush (flag)
            while (flag != 1) {
#pragma omp flush (flag)
            }
#pragma omp flush
            sum = Sum_array(N, A);
        }
    }
    runtime = omp_get_wtime() - runtime;
    printf(" In %lf seconds, The sum is %lf \n", runtime, sum);

}
