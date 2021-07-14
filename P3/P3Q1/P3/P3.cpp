#include "stdio.h"
#include "omp.h"

int main() {

	//int x = 5;

	//#pragma omp parallel
	//	{
	//		//int ID = omp_get_thread_num();
	//		//printf("hello(%d)", ID);
	//		//printf("world(%d)\n", ID);

	//		x = x + 1;
	//		printf("shared: x is %d\n", x);
	//	}

	int x = 5;

	#pragma omp parallel
		{

			int x; x = 3;
			printf("local: x is %d\n", x);

		}
	return 0;
}