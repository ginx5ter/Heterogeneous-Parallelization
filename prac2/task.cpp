#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <vector>
#include <omp.h>

using namespace std;

void bubbleSort(int *arr, int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        for (int j = 0; j < n - i - 1; ++j)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(int *arr, int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
#pragma omp parallel for
        for (int j = 0; j < n - i - 1; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
#pragma omp parallel for
        for (int j = 1; j < n - i - 2; j += 2)
        {
            if (arr[j] > arr[j + 1])
            {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void selectionSort(int *arr, int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        int min_idx = i;
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[j] < arr[min_idx])
            {
                min_idx = j;
            }
        }
        swap(arr[i], arr[min_idx]);
    }
}

void parallelSelectionSort(int *arr, int n)
{
    for (int i = 0; i < n - 1; ++i)
    {
        int min_idx = i;
#pragma omp parallel for reduction(min : min_idx)
        for (int j = i + 1; j < n; ++j)
        {
            if (arr[j] < arr[min_idx])
            {
                min_idx = j;
            }
        }
        swap(arr[i], arr[min_idx]);
    }
}

void insertionSort(int *arr, int n)
{
    for (int i = 1; i < n; ++i)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

void parallelInsertionSort(int *arr, int n)
{
#pragma omp parallel for
    for (int i = 1; i < n; ++i)
    {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

void copyArray(int *src, int *dest, int n)
{
    for (int i = 0; i < n; ++i)
    {
        dest[i] = src[i];
    }
}

int main()
{
    srand(static_cast<unsigned int>(time(nullptr)));

    vector<int> sizes = {1000, 10000, 100000};

    for (int size : sizes)
    {
        cout << "\n=== Размер массива: ";

        int *original = new int[size];
        for (int i = 0; i < size; ++i)
        {
            original[i] = rand() % 100000;
        }

        int *arr = new int[size];

        copyArray(original, arr, size);
        auto start = chrono::high_resolution_clock::now();
        bubbleSort(arr, size);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> time = end - start;
        cout << "bubble (seq): " << time.count();

        copyArray(original, arr, size);
        start = chrono::high_resolution_clock::now();
        parallelBubbleSort(arr, size);
        end = chrono::high_resolution_clock::now();
        time = end - start;
        cout << "bubble (par):     " << time.count();

        copyArray(original, arr, size);
        start = chrono::high_resolution_clock::now();
        selectionSort(arr, size);
        end = chrono::high_resolution_clock::now();
        time = end - start;
        cout << "selection (seq):   " << time.count();

        copyArray(original, arr, size);
        start = chrono::high_resolution_clock::now();
        parallelSelectionSort(arr, size);
        end = chrono::high_resolution_clock::now();
        time = end - start;
        cout << "selection (par):       " << time.count();

        copyArray(original, arr, size);
        start = chrono::high_resolution_clock::now();
        insertionSort(arr, size);
        end = chrono::high_resolution_clock::now();
        time = end - start;
        cout << "insertion (seq): " << time.count();

        copyArray(original, arr, size);
        start = chrono::high_resolution_clock::now();
        parallelInsertionSort(arr, size);
        end = chrono::high_resolution_clock::now();
        time = end - start;
        cout << "insertion (par):     " << time.count();

        delete[] original;
        delete[] arr;
    }

    return 0;
}

//  Алгоритмы сортировки с квадратичной сложностью трудно хорошо распараллелить на CPU с помощью OpenMP из-за зависимостей между шагами и затрат на запуск потоков.
//  Из трёх сортировок лучше всего параллелится выбором — на 10 000 элементов она ускорилась благодаря reduction.
//  Сортировка вставкой тоже дала выигрыш на большом массиве, но на маленьком (1000 элементов) оказалась медленнее.
//  Пузырьковая сортировка в этой реализации не ускорилась, а даже замедлилась — для неё нужна более удачная параллельная схема.
//  В целом, простое добавление #pragma omp parallel for не всегда помогает: результат сильно зависит от алгоритма и размера данных.
//  Для реального ускорения лучше брать более эффективные сортировки (quicksort, mergesort) или стандартные параллельные реализации из C++17.