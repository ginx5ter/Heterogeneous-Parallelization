#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>

#define CU_ERR(expr)                                                           \
  do {                                                                         \
    cudaError_t status = (expr);                                               \
    if (status != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA failure: %s  line %d\n",                           \
              cudaGetErrorString(status), __LINE__);                           \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void inclusive_block_scan(float *input, float *output) {
  __shared__ float buffer[1024];

  int lane = threadIdx.x;

  buffer[lane] = (lane < 1024) ? input[lane] : 0.0f;
  __syncthreads();

  for (unsigned int step = 1; step < 1024; step *= 2) {
    float prev = (lane >= step) ? buffer[lane - step] : 0.0f;
    __syncthreads();
    buffer[lane] += prev;
    __syncthreads();
  }

  output[lane] = buffer[lane];
}

int main() {
  const int len = 1024;
  std::vector<float> data(len, 1.0f);

  auto t_start = std::chrono::steady_clock::now();

  float acc = 0.0f;
  for (int i = 0; i < len; ++i) {
    acc += data[i];
    data[i] = acc;
  }

  auto t_end = std::chrono::steady_clock::now();
  double cpu_duration_ms =
      std::chrono::duration<double, std::milli>(t_end - t_start).count();

  float cpu_final = data.back();

  float *d_src = nullptr;
  float *d_dst = nullptr;

  CU_ERR(cudaMalloc(&d_src, len * sizeof(float)));
  CU_ERR(cudaMalloc(&d_dst, len * sizeof(float)));

  CU_ERR(cudaMemcpy(d_src, data.data(), len * sizeof(float),
                    cudaMemcpyHostToDevice));

  cudaEvent_t begin, finish;
  CU_ERR(cudaEventCreate(&begin));
  CU_ERR(cudaEventCreate(&finish));

  CU_ERR(cudaEventRecord(begin));

  inclusive_block_scan<<<1, len>>>(d_src, d_dst);

  CU_ERR(cudaGetLastError());
  CU_ERR(cudaEventRecord(finish));
  CU_ERR(cudaEventSynchronize(finish));

  float gpu_duration_ms = 0.0f;
  CU_ERR(cudaEventElapsedTime(&gpu_duration_ms, begin, finish));

  CU_ERR(cudaMemcpy(data.data(), d_dst, len * sizeof(float),
                    cudaMemcpyDeviceToHost));

  float gpu_final = data.back();

  printf("Prefix sum (inclusive, shared memory, block size 1024)\n");
  printf("CPU time: %.4f ms\n", cpu_duration_ms);
  printf("GPU time: %.4f ms\n", gpu_duration_ms);

  CU_ERR(cudaEventDestroy(begin));
  CU_ERR(cudaEventDestroy(finish));
  CU_ERR(cudaFree(d_src));
  CU_ERR(cudaFree(d_dst));

  return 0;
}