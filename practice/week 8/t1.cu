#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <vector>

#define CHECK_CUDA(call)                                                       \
  {                                                                            \
    cudaError_t e = call;                                                      \
    if (e != cudaSuccess) {                                                    \
      std::cerr << "CUDA fail: " << cudaGetErrorString(e) << "  [" << __FILE__ \
                << ":" << __LINE__ << "]\n";                                   \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

__global__ void double_elements(float *__restrict__ arr, size_t len) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    arr[i] *= 2.0f;
  }
}

void multiply_cpu(float *arr, size_t count) {
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < count; ++i) {
    arr[i] *= 2.0f;
  }
}

int main() {
  constexpr size_t TOTAL_ELEMENTS = 1'000'000;
  constexpr size_t HALF_ELEMENTS = TOTAL_ELEMENTS / 2;
  constexpr int BLOCK_SIZE = 256;

  std::cout << "Hybrid processing: CPU vs GPU vs Hybrid\n\n";

  std::vector<float> host_data(TOTAL_ELEMENTS, 1.0f);

  auto t_cpu_start = std::chrono::steady_clock::now();

  multiply_cpu(host_data.data(), TOTAL_ELEMENTS);

  auto t_cpu_end = std::chrono::steady_clock::now();
  double cpu_duration_ms =
      std::chrono::duration<double, std::milli>(t_cpu_end - t_cpu_start)
          .count();

  std::vector<float> gpu_input(TOTAL_ELEMENTS, 1.0f);
  float *dev_buffer = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_buffer, TOTAL_ELEMENTS * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(dev_buffer, gpu_input.data(),
                        TOTAL_ELEMENTS * sizeof(float),
                        cudaMemcpyHostToDevice));

  cudaEvent_t ev_start, ev_finish;
  CHECK_CUDA(cudaEventCreate(&ev_start));
  CHECK_CUDA(cudaEventCreate(&ev_finish));

  CHECK_CUDA(cudaEventRecord(ev_start));

  int grid_dim = (TOTAL_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  double_elements<<<grid_dim, BLOCK_SIZE>>>(dev_buffer, TOTAL_ELEMENTS);

  CHECK_CUDA(cudaEventRecord(ev_finish));
  CHECK_CUDA(cudaEventSynchronize(ev_finish));

  float gpu_duration_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&gpu_duration_ms, ev_start, ev_finish));

  CHECK_CUDA(cudaMemcpy(gpu_input.data(), dev_buffer,
                        TOTAL_ELEMENTS * sizeof(float),
                        cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(dev_buffer));
  CHECK_CUDA(cudaEventDestroy(ev_start));
  CHECK_CUDA(cudaEventDestroy(ev_finish));

  std::vector<float> hybrid_data(TOTAL_ELEMENTS, 1.0f);

  float *dev_half = nullptr;
  CHECK_CUDA(cudaMalloc(&dev_half, HALF_ELEMENTS * sizeof(float)));

  auto t_hybrid_start = std::chrono::steady_clock::now();

#pragma omp parallel sections num_threads(2)
  {
#pragma omp section
    {
      // Первая половина — CPU
      multiply_cpu(hybrid_data.data(), HALF_ELEMENTS);
    }

#pragma omp section
    {
      // Вторая половина — GPU
      CHECK_CUDA(cudaMemcpy(dev_half, hybrid_data.data() + HALF_ELEMENTS,
                            HALF_ELEMENTS * sizeof(float),
                            cudaMemcpyHostToDevice));

      int grid_half = (HALF_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
      double_elements<<<grid_half, BLOCK_SIZE>>>(dev_half, HALF_ELEMENTS);

      CHECK_CUDA(cudaDeviceSynchronize());

      CHECK_CUDA(cudaMemcpy(hybrid_data.data() + HALF_ELEMENTS, dev_half,
                            HALF_ELEMENTS * sizeof(float),
                            cudaMemcpyDeviceToHost));
    }
  }

  auto t_hybrid_end = std::chrono::steady_clock::now();
  double hybrid_duration_ms =
      std::chrono::duration<double, std::milli>(t_hybrid_end - t_hybrid_start)
          .count();

  CHECK_CUDA(cudaFree(dev_half));

  std::cout << "Size of the array:       " << TOTAL_ELEMENTS << "\n";
  std::cout << "Time (only CPU):   " << cpu_duration_ms << " ms\n";
  std::cout << "Time (only GPU):   " << gpu_duration_ms << " ms\n";
  std::cout << "Time (hybrid):       " << hybrid_duration_ms << " ms\n";

  return 0;
}