#include <cstdio>
#include <cuda_runtime.h>

__global__ void good_access(float *__restrict__ buffer, int count) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) {
    buffer[gid] *= 2;
  }
}

__global__ void bad_access(float *__restrict__ buffer, int count) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  int scrambled = (gid * 37) % count;
  buffer[scrambled] *= 2;
}

int main() {
  constexpr int ELEMENTS = 1 << 24;
  float *dev_ptr = nullptr;
  cudaMalloc(&dev_ptr, ELEMENTS * sizeof(float));

  cudaEvent_t begin, finish;
  cudaEventCreate(&begin);
  cudaEventCreate(&finish);

  constexpr int BLOCK = 256;
  int grid = (ELEMENTS + BLOCK - 1) / BLOCK;

  cudaEventRecord(begin);
  good_access<<<grid, BLOCK>>>(dev_ptr, ELEMENTS);
  cudaEventRecord(finish);
  cudaEventSynchronize(finish);

  float ms_coalesced = 0;
  cudaEventElapsedTime(&ms_coalesced, begin, finish);

  cudaEventRecord(begin);
  bad_access<<<grid, BLOCK>>>(dev_ptr, ELEMENTS);
  cudaEventRecord(finish);
  cudaEventSynchronize(finish);

  float ms_scattered = 0;
  cudaEventElapsedTime(&ms_scattered, begin, finish);

  printf("Coalesced version:    %.3f ms\n", ms_coalesced);
  printf("Scattered version:    %.3f ms\n", ms_scattered);

  cudaFree(dev_ptr);
  cudaEventDestroy(begin);
  cudaEventDestroy(finish);

  return 0;
}