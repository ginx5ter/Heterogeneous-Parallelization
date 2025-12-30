#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const int size = 10000;
    int *array = new int[size];

    for (int i = 0; i < size; ++i)
    {
        array[i] = std::rand();
    }

    // seqential
    auto start_seq = std::chrono::high_resolution_clock::now();
    int min_seq = array[0];
    int max_seq = array[0];
    for (int i = 1; i < size; ++i)
    {
        if (array[i] < min_seq)
            min_seq = array[i];
        if (array[i] > max_seq)
            max_seq = array[i];
    }
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;

    std::cout << "seqential: Min = " << min_seq << ", Max = " << max_seq << std::endl;
    std::cout << "time seqential: " << duration_seq.count() << std::endl;

    // parallel
    auto start_par = std::chrono::high_resolution_clock::now();
    int min_par = array[0];
    int max_par = array[0];
#pragma omp parallel for reduction(min : min_par) reduction(max : max_par)
    for (int i = 1; i < size; ++i)
    {
        if (array[i] < min_par)
            min_par = array[i];
        if (array[i] > max_par)
            max_par = array[i];
    }
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;

    std::cout << "parallel: Min = " << min_par << ", Max = " << max_par << std::endl;
    std::cout << "time parallel: " << duration_par.count() << std::endl;

    delete[] array;
    return 0;
}

// Последовательная версия нашла Min = 1, Max = 32767 за 1.18e-05 секунды.
// Параллельная версия с OpenMP нашла те же значения Min = 1, Max = 32767 за 1.25e-05 секунды.

// Для массива из 10 000 элементов параллельная версия с OpenMP не дала ускорения, а даже немного замедлила выполнение.
// Это происходит потому, что работа очень простая и быстрая — обычный цикл по небольшому массиву выполняется почти мгновенно на одном ядре.