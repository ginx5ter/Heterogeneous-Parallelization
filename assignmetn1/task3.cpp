#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const int size = 1000000;
    int *array = new int[size];

    for (int i = 0; i < size; ++i)
    {
        array[i] = std::rand();
    }

    auto start = std::chrono::high_resolution_clock::now();

    int min_val = array[0];
    int max_val = array[0];
#pragma omp parallel for reduction(min : min_val) reduction(max : max_val)
    for (int i = 1; i < size; ++i)
    {
        if (array[i] < min_val)
            min_val = array[i];
        if (array[i] > max_val)
            max_val = array[i];
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "min: " << min_val << std::endl;
    std::cout << "max: " << max_val << std::endl;
    std::cout << "execution time: " << duration.count() << std::endl;

    delete[] array;

    return 0;
}

// задача 2 - execution time: 0.0011341
// задача 3 - execution time: 0.0011359

// Для массивов размером 10 000 элементов и простых операций параллелизм с OpenMP на обычном CPU почти не даёт выигрыша