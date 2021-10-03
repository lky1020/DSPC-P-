// P4Q1.cpp : Defines the entry point for the console application.
//

#include "stdio.h"
#include "omp.h"
#define NUM_THREADS 8
static long num_steps = 100000;
double step;

int main()
{
    int i, nthreads;
    double pi = 0;
    double sum;
    step = 1.0 / (double)num_steps;
    omp_set_num_threads(NUM_THREADS);
    double start_time = omp_get_wtime();

//Create local sum
#pragma omp parallel private (sum)
    {
        int i, id, nthrds;
        double x;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();
        if (id == 0) nthreads = nthrds;

        for (i = id, sum = 0.0; i < num_steps; i = i + nthrds)
        {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        //Prevent race condition (only one thread can write shared variable)
        #pragma omp critical
        {
            pi += sum * step;
        }
    }

    double end_time = omp_get_wtime();

    //This is for partial sum
    //for (i = 0, pi = 0.0; i < nthreads; i++) {
    //    pi += sum * step;
    //}

    printf("%f\n", pi);

    printf("Work took %f seconds\n", end_time - start_time);
    return 0;
}