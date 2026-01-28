#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_SAFE(call)                                                        \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(err),     \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void atomic_reduce(const float *__restrict__ input,
                              float *__restrict__ result, int count) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < count) {
    atomicAdd(result, input[pos]);
  }
}

int main() {
  const int SIZE = 100000;
  const int BLOCK = 256;

  std::vector<float> host_data(SIZE, 1.0f);

  auto cpu_start = std::chrono::high_resolution_clock::now();

  float cpu_result = 0.0f;
  for (float v : host_data) {
    cpu_result += v;
  }

  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_ms =
      std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

  float *dev_input = nullptr;
  float *dev_sum = nullptr;

  CUDA_SAFE(cudaMalloc(&dev_input, SIZE * sizeof(float)));
  CUDA_SAFE(cudaMalloc(&dev_sum, sizeof(float)));

  CUDA_SAFE(cudaMemcpy(dev_input, host_data.data(), SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));
  CUDA_SAFE(cudaMemset(dev_sum, 0, sizeof(float)));

  cudaEvent_t evt_start, evt_stop;
  CUDA_SAFE(cudaEventCreate(&evt_start));
  CUDA_SAFE(cudaEventCreate(&evt_stop));

  CUDA_SAFE(cudaEventRecord(evt_start));

  int grid = (SIZE + BLOCK - 1) / BLOCK;
  atomic_reduce<<<grid, BLOCK>>>(dev_input, dev_sum, SIZE);

  CUDA_SAFE(cudaGetLastError());
  CUDA_SAFE(cudaEventRecord(evt_stop));
  CUDA_SAFE(cudaEventSynchronize(evt_stop));

  float gpu_ms = 0.0f;
  CUDA_SAFE(cudaEventElapsedTime(&gpu_ms, evt_start, evt_stop));

  float gpu_result = 0.0f;
  CUDA_SAFE(
      cudaMemcpy(&gpu_result, dev_sum, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_SAFE(cudaEventDestroy(evt_start));
  CUDA_SAFE(cudaEventDestroy(evt_stop));
  CUDA_SAFE(cudaFree(dev_input));
  CUDA_SAFE(cudaFree(dev_sum));

  printf("Atomic reduction (N = %d)\n", SIZE);
  printf("  CPU result = %.1f   time = %.3f ms\n", cpu_result, cpu_ms);
  printf("  GPU result = %.1f   time = %.3f ms\n", gpu_result, gpu_ms);

  return 0;
}