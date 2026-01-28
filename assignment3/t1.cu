#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CHECK_CUDA(err)                                                        \
  do {                                                                         \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA fail: %s (%s:%d)\n", cudaGetErrorString(err),      \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void scale_global(float *data, float factor, int count) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < count) {
    data[gid] *= factor;
  }
}

__global__ void scale_with_shared(float *data, float factor, int count) {
  __shared__ float temp[256];

  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lid = threadIdx.x;

  if (gid < count) {
    temp[lid] = data[gid];
  }
  __syncthreads();

  if (gid < count) {
    data[gid] = temp[lid] * factor;
  }
}

double elapsed_ms(void (*kernel)(float *, float, int), float *dev_ptr,
                  float mul, int n, int bs) {
  cudaEvent_t begin, finish;
  CHECK_CUDA(cudaEventCreate(&begin));
  CHECK_CUDA(cudaEventCreate(&finish));

  int grid = (n + bs - 1) / bs;

  CHECK_CUDA(cudaEventRecord(begin));
  kernel<<<grid, bs>>>(dev_ptr, mul, n);
  CHECK_CUDA(cudaEventRecord(finish));
  CHECK_CUDA(cudaEventSynchronize(finish));

  float ms = 0;
  CHECK_CUDA(cudaEventElapsedTime(&ms, begin, finish));

  CHECK_CUDA(cudaEventDestroy(begin));
  CHECK_CUDA(cudaEventDestroy(finish));

  return ms;
}

int main() {
  constexpr int SIZE = 1000000;
  constexpr int BLOCK = 256;
  constexpr float MULTIPLY_BY = 3.14f;

  std::vector<float> host(SIZE, 1.0f);

  float *device = nullptr;
  CHECK_CUDA(cudaMalloc(&device, SIZE * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(device, host.data(), SIZE * sizeof(float),
                        cudaMemcpyHostToDevice));

  double t_global = elapsed_ms(scale_global, device, MULTIPLY_BY, SIZE, BLOCK);
  double t_shared =
      elapsed_ms(scale_with_shared, device, MULTIPLY_BY, SIZE, BLOCK);

  printf("Global memory version : %.3f ms\n", t_global);
  printf("Shared memory version : %.3f ms\n", t_shared);

  CHECK_CUDA(cudaFree(device));
  return 0;
}