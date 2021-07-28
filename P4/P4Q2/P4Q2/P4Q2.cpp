// P4Q2.cpp : Defines the entry point for the console application.
//

#include "stdio.h"
#include "omp.h"

const long MAX = 100000;

int main()
{

    double ave = 0.0, A[MAX];
    int i;
    omp_set_num_threads(8);
    for (i = 0; i < MAX; i++)
    {
        A[i] = i;
    }
    double start_time = omp_get_wtime();

    //Method 1
    #pragma omp parallel
    {
        int id, nthrds;
        id = omp_get_thread_num();
        nthrds = omp_get_num_threads();

        for (i = id; i < MAX; i = i + nthrds) {
            ave += A[i];
        }

        #pragma omp critical 
        {
            ave = ave / MAX;
        }
    }

    // Method 2
    //#pragma omp parallel for reduction(+:ave)
    //{
    //    for (i = 0; i < MAX; i++) {
    //        ave += A[i];
    //    }
    //}
    
    double end_time = omp_get_wtime();
    //ave = ave / MAX;
    printf("%f\n", ave);
    printf("Work took %f seconds\n", end_time - start_time);
    return 0;
}