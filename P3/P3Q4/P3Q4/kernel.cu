#include <stdio.h>

#define SIZE	1024

__global__ void VectorAdd(int* a, int* b, int* c, int n)
{
    // 0 - 1023 thread id
    int i = threadIdx.x;

    //check whether num of threads used more than num of threads available
    if (i < n)
        c[i] = a[i] + b[i];
}

int main()
{
    int* a, * b, * c;
    int* d_a, * d_b, * d_c;

    //Allocate host memory
    a = (int*)malloc(SIZE * sizeof(int));
    b = (int*)malloc(SIZE * sizeof(int));
    c = (int*)malloc(SIZE * sizeof(int));

    //Allocate device memory
    cudaMalloc(&d_a, SIZE * sizeof(int));
    cudaMalloc(&d_b, SIZE * sizeof(int));
    cudaMalloc(&d_c, SIZE * sizeof(int));

    //Initialize Host Memory
    for (int i = 0; i < SIZE; ++i)
    {
        a[i] = i;
        b[i] = i;
        c[i] = 0;
    }

    //Copy host memory to device
    cudaMemcpy(d_a, a, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, SIZE * sizeof(int), cudaMemcpyHostToDevice);

    //One block size with all threads available && invoke kernel function
    VectorAdd << < 1, SIZE >> > (d_a, d_b, d_c, SIZE);

    //Copy device memory to host
    cudaMemcpy(c, d_c, SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i)
        printf("c[%d] = %d\n", i, c[i]);

    //free host memory resources
    free(a);
    free(b);
    free(c);

    //free device memory resources
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}