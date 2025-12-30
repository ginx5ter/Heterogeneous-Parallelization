#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    const int size = 5000000;
    int *array = new int[size];

    for (int i = 0; i < size; ++i)
    {
        array[i] = std::rand() % 100 + 1;
    }

    auto start_seq = std::chrono::high_resolution_clock::now();
    long long sum_seq = 0;
    for (int i = 0; i < size; ++i)
    {
        sum_seq += array[i];
    }
    double average_seq = static_cast<double>(sum_seq) / size;
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;

    std::cout << "avg sequential: " << average_seq << std::endl;
    std::cout << "execution time: " << duration_seq.count() << std::endl;

    auto start_par = std::chrono::high_resolution_clock::now();
    long long sum_par = 0;
#pragma omp parallel for reduction(+ : sum_par)
    for (int i = 0; i < size; ++i)
    {
        sum_par += array[i];
    }
    double average_par = static_cast<double>(sum_par) / size;
    auto end_par = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_par = end_par - start_par;

    std::cout << "avg parallel: " << average_par << std::endl;
    std::cout << "execution time: " << duration_par.count() << std::endl;

    delete[] array;

    return 0;
}