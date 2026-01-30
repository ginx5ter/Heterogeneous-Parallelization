#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void scaleKernel(float *data, size_t len) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len) {
    data[idx] *= 2.0f;
  }
}

int main() {
  constexpr size_t COUNT = 1'000'000;
  std::vector<float> host_data(COUNT, 1.0f);

  float *device_data = nullptr;
  cudaMalloc(&device_data, COUNT * sizeof(float));

  cudaStream_t workStream = nullptr;
  cudaStreamCreate(&workStream);

  cudaMemcpyAsync(device_data, host_data.data(), COUNT * sizeof(float),
                  cudaMemcpyHostToDevice, workStream);

  constexpr int BS = 256;
  int gridDim = (COUNT + BS - 1) / BS;

  scaleKernel<<<gridDim, BS, 0, workStream>>>(device_data, COUNT);

  cudaMemcpyAsync(host_data.data(), device_data, COUNT * sizeof(float),
                  cudaMemcpyDeviceToHost, workStream);

  cudaStreamSynchronize(workStream);

  std::cout << "Asynchronous finished\n";

  cudaFree(device_data);
  cudaStreamDestroy(workStream);

  return 0;
}