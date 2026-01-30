#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void inclusive_scan_block(float *__restrict__ input,
                                     float *__restrict__ output, int len) {
  extern __shared__ float buffer[];

  int tid = threadIdx.x;

  if (tid < len)
    buffer[tid] = input[tid];
  else
    buffer[tid] = 0.0f;

  __syncthreads();

  for (int step = 1; step < blockDim.x; step *= 2) {
    float add = 0.0f;
    if (tid >= step)
      add = buffer[tid - step];

    __syncthreads();

    buffer[tid] += add;

    __syncthreads();
  }

  if (tid < len)
    output[tid] = buffer[tid];
}

int main() {
  constexpr int SIZE = 1024;

  std::vector<float> host_input(SIZE, 1.0f);

  float *dev_in = nullptr;
  float *dev_out = nullptr;

  cudaMalloc(&dev_in, SIZE * sizeof(float));
  cudaMalloc(&dev_out, SIZE * sizeof(float));

  cudaMemcpy(dev_in, host_input.data(), SIZE * sizeof(float),
             cudaMemcpyHostToDevice);

  inclusive_scan_block<<<1, SIZE, SIZE * sizeof(float)>>>(dev_in, dev_out,
                                                          SIZE);

  cudaDeviceSynchronize();

  cudaMemcpy(host_input.data(), dev_out, SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);

  std::cout << "Last element: " << host_input.back() << "  (waiting for "
            << SIZE << ")\n";

  cudaFree(dev_in);
  cudaFree(dev_out);

  return 0;
}