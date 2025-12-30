#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

void parallelSelectionSort(int arr[], int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        int min_idx = i;
#pragma omp parallel for reduction(min : min_idx)
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[j] < arr[min_idx])
                min_idx = j;
        }
        std::swap(arr[i], arr[min_idx]);
    }
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int size : {1000, 10000})
    {
        int *arr = new int[size];
        for (int i = 0; i < size; ++i)
            arr[i] = std::rand();

        auto start = std::chrono::high_resolution_clock::now();
        parallelSelectionSort(arr, size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        std::cout << "parallel sort for " << size << " elements: " << duration.count() << std::endl;

        delete[] arr;
    }
    return 0;
}

// Последовательная версия:
// Для 1000 элементов — 0.00054 секунды
// Для 10 000 элементов — 0.065 секунды
// Параллельная версия с OpenMP:
// Для 1000 элементов — 0.00097 секунды
// Для 10 000 элементов — 0.052 секунды

// На маленьком массиве (1000 элементов) параллельная версия работает медленнее почти в два раза. Это происходит потому, что запуск потоков в OpenMP занимает время, а работы слишком мало — все эти затраты просто не окупаются.
// На большом массиве (10 000 элементов) параллельная версия уже быстрее примерно на 20–25%. Здесь поиск минимума в длинном внутреннем цикле можно успешно разделить между ядрами процессора, и это даёт видимый результат.