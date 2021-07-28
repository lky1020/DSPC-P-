// P5Q1.cpp : Defines the entry point for the console application.
//

#include "stdio.h"
#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <thread>
#include <omp.h>

enum class Co
{
    Company1, Company2
};

int fetchTheSalary(const int employee,
    const Co company)
{
    if (company == Co::Company1)
    {
        return (employee < 100) ? 99340 : 54300;
    }
    else
    {
        return (employee < 1000) ? 88770 : 57330;
    }
}


int main()
{

    int salaries1 = 0;
    int salaries2 = 0;
    double runtime;
    omp_set_num_threads(8);
    runtime = omp_get_wtime();
#pragma omp parallel shared(salaries1, salaries2)
    {
#pragma omp for reduction(+:salaries1)
        for (int employee = 0; employee < 25000; employee++)
        {
            salaries1 += fetchTheSalary(employee, Co::Company1);
        }

        #pragma omp single
        std::cout << "Salaries1: " << salaries1 << std::endl;

#pragma omp for reduction(+:salaries2)
        for (int employee = 0; employee < 25000; employee++)
        {
            salaries2 += fetchTheSalary(employee, Co::Company2);
        }

        #pragma omp single
        std::cout << "Salaries2: " << salaries2 << std::endl;

    }
    runtime = omp_get_wtime() - runtime;
    printf(" In %lf seconds", runtime);
    return 0;

}
