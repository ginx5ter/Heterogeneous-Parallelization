#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

#define CUDACHECK(expr)                                                        \
  do {                                                                         \
    cudaError_t err = expr;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " → "    \
                << cudaGetErrorString(err) << std::endl;                       \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__global__ void block_reduce_sum(const float *__restrict__ values,
                                 float *__restrict__ block_sums, size_t count) {
  extern __shared__ float shared_mem[];

  int lane = threadIdx.x;
  int global = blockIdx.x * blockDim.x + threadIdx.x;

  shared_mem[lane] = (global < count) ? values[global] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (lane < stride) {
      shared_mem[lane] += shared_mem[lane + stride];
    }
    __syncthreads();
  }

  if (lane == 0) {
    block_sums[blockIdx.x] = shared_mem[0];
  }
}

int main() {
  constexpr size_t ARRAY_SIZE = 1 << 20; // 1 миллион элементов
  constexpr int BLOCK_SIZE = 512;

  std::vector<float> host_data(ARRAY_SIZE);
  std::mt19937_64 rnd(12345);
  std::uniform_real_distribution<float> uniform(0.0f, 2.0f);

  float reference = 0.0f;
  for (auto &v : host_data) {
    v = uniform(rnd);
    reference += v;
  }

  int num_blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

  float *dev_input = nullptr;
  float *dev_block_results = nullptr;

  CUDACHECK(cudaMalloc(&dev_input, ARRAY_SIZE * sizeof(float)));
  CUDACHECK(cudaMalloc(&dev_block_results, num_blocks * sizeof(float)));

  CUDACHECK(cudaMemcpy(dev_input, host_data.data(), ARRAY_SIZE * sizeof(float),
                       cudaMemcpyHostToDevice));

  block_reduce_sum<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
      dev_input, dev_block_results, ARRAY_SIZE);

  CUDACHECK(cudaDeviceSynchronize());

  std::vector<float> block_results(num_blocks);
  CUDACHECK(cudaMemcpy(block_results.data(), dev_block_results,
                       num_blocks * sizeof(float), cudaMemcpyDeviceToHost));

  float gpu_total = 0.0f;
  for (float s : block_results)
    gpu_total += s;

  std::cout << "Size of array:      " << ARRAY_SIZE << "\n";
  std::cout << "Sum (CPU)  = " << reference << "\n";
  std::cout << "Sum (GPU)  = " << gpu_total << "\n";
  std::cout << "Difference      = " << std::abs(reference - gpu_total) << "\n";

  CUDACHECK(cudaFree(dev_input));
  CUDACHECK(cudaFree(dev_block_results));

  return 0;
}