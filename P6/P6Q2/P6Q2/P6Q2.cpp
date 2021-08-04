// This program is to caculate PI using MPI
// The algorithm is based on integral representation of PI. If f(x)=4/(1+x^2), then PI is the intergral of f(x) from 0 to 1

#include <stdio.h>
#include <mpi.h>

#define N 1E7
#define d 1E-7
#define d2 1E-14

int main(int argc, char* argv[])
{
    int rank, size, error, i;
    double pi = 0.0, result = 0.0, sum = 0.0, begin = 0.0, end = 0.0, x2;

    MPI_Init(&argc, &argv);

    //Get process ID
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //The default communicator is called MPI_COMM_WORLD. It basically groups all the processes when the program started. If you take a look at the example below, you see a depiction of a program ran with five processes. Every process is connected and can communicate inside this communicator.
    
    //Get processes Number
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    //Synchronize all processes and get the begin time
    MPI_Barrier(MPI_COMM_WORLD);
    begin = MPI_Wtime();

    //Each process will caculate a part of the sum
    for (i = rank; i < N; i += size)
    {
        x2 = d2 * i * i;
        result += 1.0 / (1.0 + x2);
    }

    //Sum up all results
    MPI_Reduce(&result, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //Synchronize all processes and get the end time
    MPI_Barrier(MPI_COMM_WORLD);
    end = MPI_Wtime();

    //Caculate and print PI
    if (rank == 0)
    {
        pi = 4 * d * sum;
        printf("np=%2d;    Time=%fs;    PI=%lf\n", size, end - begin, pi);
    }

    error = MPI_Finalize();

    return 0;
}