#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstdlib>

int main(void)
{
    //@@ generate random data serially
    thrust::host_vector<int> h_vec(200);
    std::generate(h_vec.begin(), h_vec.end(), rand);

    //@@ transfer to device and compute sum
    thrust::device_vector<int> d_vec = h_vec;

    //Parallel Vector Addition
    int x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());

    //@@ Display the sum
    for (int i = 0; i < h_vec.size(); i++) {
        printf("h_vec [%d] = %d\n", i, h_vec[i]);
    }
    printf("Sum x = %d\n", x);

    return 0;
}