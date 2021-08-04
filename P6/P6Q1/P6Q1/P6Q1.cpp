//run in cmd/powershell with mpiexec -n 4 MPI_Example
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int args, char** argvs)
{
    cout << "Hello World" << endl;
    int rank = 0, numOfProcess = 0, len;
    char name[100];
    MPI_Init(&args, &argvs);
    MPI_Get_processor_name(name, &len);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //The default communicator is called MPI_COMM_WORLD. It basically groups all the processes when the program started. If you take a look at the example below, you see a depiction of a program ran with five processes. Every process is connected and can communicate inside this communicator.
    MPI_Comm_size(MPI_COMM_WORLD, &numOfProcess);
    cout << "Hello World from process rank(number) " << rank << " from " << numOfProcess << " on " << name << endl;
    MPI_Finalize();
    return 0;
}