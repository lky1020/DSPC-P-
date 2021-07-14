// P3Q2a.cpp : Defines the entry point for the console application.
//

//#include "stdio.h"
//#include "omp.h"
//#define NUM_THREADS 4
//static long num_steps = 100000;
//double step;
//
//int main()
//{
//    int i, nthreads;
//    double pi, sum[NUM_THREADS];
//    step = 1.0 / (double)num_steps;
//    omp_set_num_threads(NUM_THREADS);
//    double start_time = omp_get_wtime();
//
//#pragma omp parallel
//    {
//        int i, id, nthrds;
//        double x;
//        id = omp_get_thread_num();
//        nthrds = omp_get_num_threads();
//        if (id == 0) nthreads = nthrds;
//
//        for (int j = 0; j < nthreads; j++) {
//            for (i = id, sum[j] = 0.0; i < num_steps; i = i + nthrds)
//            {
//                x = (i + 0.5) * step;
//                sum[j] += 4.0 / (1.0 + x * x);
//            }
//        }
//        
//    }
//
//    double end_time = omp_get_wtime();
//    for (i = 0, pi = 0.0; i < nthreads; i++) {
//        pi += sum[i] * step;
//    }
//    printf("%f\n", pi);
//
//    printf("Work took %f seconds\n", end_time - start_time);
//    return 0;
//}


// P3Q2c.cpp : Defines the entry point for the console application.
//
#include <stdio.h>
#include <omp.h>
static long num_steps = 10000;
double step;
int main()
{
    int i;
    double x, pi, sum = 0.0;
    double start_time, run_time;

    step = 1.0 / (double)num_steps;
    for (i = 1; i <= 8; i++) {
        sum = 0.0;
        omp_set_num_threads(i);
        start_time = omp_get_wtime();

            #pragma omp parallel  
                {
                    int i; i = 1;

                    #pragma omp single
                        {
                            printf(" num_threads = %d", omp_get_num_threads());

                        #pragma omp parallel for reduction(+:sum)
                            for (i = 1; i <= num_steps; i++) {
                                x = (i - 0.5) * step;
                                sum = sum + 4.0 / (1.0 + x * x);
                            }

                            pi = sum * step;
                        }
                }

        run_time = omp_get_wtime() - start_time;
        printf("\n pi is %f in %f seconds and %d threads\n", pi, run_time, i);
    }
}





