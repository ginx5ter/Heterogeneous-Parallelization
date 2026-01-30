#include <chrono>
#include <cuda_runtime.h>
#include <iostream>

__global__ void trivial_work(float *__restrict__ ptr) {
  int pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < 1024)
    ptr[pos] += 0.001f;
}

int main() {
  constexpr int ELEMENTS = 1024;

  float *dev_ptr = nullptr;
  cudaMalloc(&dev_ptr, ELEMENTS * sizeof(float));

  cudaEvent_t ev_begin, ev_end;
  cudaEventCreate(&ev_begin);
  cudaEventCreate(&ev_end);

  cudaEventRecord(ev_begin);

  trivial_work<<<4, 256>>>(dev_ptr);

  cudaEventRecord(ev_end);
  cudaEventSynchronize(ev_end);

  float gpu_elapsed_ms = 0.0f;
  cudaEventElapsedTime(&gpu_elapsed_ms, ev_begin, ev_end);

  cudaEventDestroy(ev_begin);
  cudaEventDestroy(ev_end);

  auto cpu_start = std::chrono::steady_clock::now();

  volatile float accumulator = 0.0f;
  for (int i = 0; i < ELEMENTS; ++i) {
    accumulator += 0.001f;
  }

  auto cpu_finish = std::chrono::steady_clock::now();
  double cpu_elapsed_ms =
      std::chrono::duration<double, std::milli>(cpu_finish - cpu_start).count();

  std::cout << "GPU:  " << gpu_elapsed_ms << " ms\n";
  std::cout << "CPU:  " << cpu_elapsed_ms << " ms\n";

  cudaFree(dev_ptr);

  return 0;
}