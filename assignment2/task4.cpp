#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

// Kernel для сортировки подмассивов (простая bubble для примера, лучше bitonic)
__global__ void sortSubarrays(int *d_arr, int n, int sub_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * sub_size >= n)
        return;

    int start = idx * sub_size;
    int end = min(start + sub_size, n);

    for (int i = start; i < end - 1; ++i)
    {
        for (int j = i + 1; j < end; ++j)
        {
            if (d_arr[i] > d_arr[j])
            {
                int temp = d_arr[i];
                d_arr[i] = d_arr[j];
                d_arr[j] = temp;
            }
        }
    }
}

// Kernel для слияния
__global__ void mergeSubarrays(int *d_arr, int *d_temp, int n, int sub_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * 2 * sub_size >= n)
        return;

    int left = idx * 2 * sub_size;
    int mid = left + sub_size;
    int right = min(mid + sub_size, n);

    int i = left, j = mid, k = left;
    while (i < mid && j < right)
    {
        if (d_arr[i] <= d_arr[j])
            d_temp[k++] = d_arr[i++];
        else
            d_temp[k++] = d_arr[j++];
    }
    while (i < mid)
        d_temp[k++] = d_arr[i++];
    while (j < right)
        d_temp[k++] = d_arr[j++];

    for (int m = left; m < right; ++m)
        d_arr[m] = d_temp[m];
}

void mergeSortGPU(int *h_arr, int n)
{
    int *d_arr, *d_temp;
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_temp, n * sizeof(int));
    cudaMemcpy(d_arr, h_arr, n * sizeof(int), cudaMemcpyHostToDevice);

    int sub_size = 256; // Размер подмассива, подогнать под warp
    int num_blocks = (n + sub_size - 1) / sub_size;

    // Сортировка подмассивов
    sortSubarrays<<<num_blocks, 1>>>(d_arr, n, sub_size);
    cudaDeviceSynchronize();

    // Итеративное слияние
    while (sub_size < n)
    {
        num_blocks = (n + 2 * sub_size - 1) / (2 * sub_size);
        mergeSubarrays<<<num_blocks, 1>>>(d_arr, d_temp, n, sub_size);
        cudaDeviceSynchronize();
        sub_size *= 2;
    }

    cudaMemcpy(h_arr, d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);
    cudaFree(d_temp);
}

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    for (int size : {10000, 100000})
    {
        int *arr = new int[size];
        for (int i = 0; i < size; ++i)
            arr[i] = std::rand();

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);
        mergeSortGPU(arr, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "Сортировка на GPU для " << size << " элементов: " << ms << " мс" << std::endl;

        delete[] arr;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    return 0;
}