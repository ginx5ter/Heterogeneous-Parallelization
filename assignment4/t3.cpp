#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CHK_CUDA(stmt)                                                         \
  do {                                                                         \
    cudaError_t res = stmt;                                                    \
    if (res != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(res),     \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void scale_kernel(float *arr, int count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    arr[idx] *= 2.0f;
  }
}

int main() {
  constexpr int TOTAL_ELEMENTS = 1000000;
  constexpr int BLOCK_SIZE = 256;

  std::vector<float> vec(TOTAL_ELEMENTS, 1.0f);
  const int half_size = TOTAL_ELEMENTS / 2;

  auto cpu_begin = std::chrono::steady_clock::now();

  for (auto &v : vec) {
    v *= 2.0f;
  }

  auto cpu_finish = std::chrono::steady_clock::now();
  double cpu_duration =
      std::chrono::duration<double, std::milli>(cpu_finish - cpu_begin).count();

  float *dev_ptr = nullptr;
  CHK_CUDA(cudaMalloc(&dev_ptr, TOTAL_ELEMENTS * sizeof(float)));
  CHK_CUDA(cudaMemcpy(dev_ptr, vec.data(), TOTAL_ELEMENTS * sizeof(float),
                      cudaMemcpyHostToDevice));

  cudaEvent_t ev1, ev2;
  CHK_CUDA(cudaEventCreate(&ev1));
  CHK_CUDA(cudaEventCreate(&ev2));

  CHK_CUDA(cudaEventRecord(ev1));

  int grid_full = (TOTAL_ELEMENTS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  scale_kernel<<<grid_full, BLOCK_SIZE>>>(dev_ptr, TOTAL_ELEMENTS);
  CHK_CUDA(cudaGetLastError());

  CHK_CUDA(cudaEventRecord(ev2));
  CHK_CUDA(cudaEventSynchronize(ev2));

  float gpu_duration = 0.0f;
  CHK_CUDA(cudaEventElapsedTime(&gpu_duration, ev1, ev2));

  CHK_CUDA(cudaEventDestroy(ev1));
  CHK_CUDA(cudaEventDestroy(ev2));

  auto hybrid_start = std::chrono::steady_clock::now();

  for (int i = 0; i < half_size; ++i) {
    vec[i] *= 2.0f;
  }

  CHK_CUDA(cudaMemcpy(dev_ptr, vec.data() + half_size,
                      half_size * sizeof(float), cudaMemcpyHostToDevice));

  int grid_half = (half_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
  scale_kernel<<<grid_half, BLOCK_SIZE>>>(dev_ptr, half_size);
  CHK_CUDA(cudaGetLastError());

  CHK_CUDA(cudaMemcpy(vec.data() + half_size, dev_ptr,
                      half_size * sizeof(float), cudaMemcpyDeviceToHost));

  auto hybrid_end = std::chrono::steady_clock::now();
  double hybrid_duration =
      std::chrono::duration<double, std::milli>(hybrid_end - hybrid_start)
          .count();

  printf("Hybrid CPU+GPU\n");
  printf("Array size  : %d elements\n", TOTAL_ELEMENTS);
  printf("CPU only    : %.3f ms\n", cpu_duration);
  printf("GPU only    : %.3f ms\n", gpu_duration);
  printf("Hybrid      : %.3f ms\n", hybrid_duration);

  CHK_CUDA(cudaFree(dev_ptr));

  return 0;
}