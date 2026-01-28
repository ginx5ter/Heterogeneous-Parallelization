#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << "\n";                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void vec_add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

float measure(int block_size, float *da, float *db, float *dc, int n) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  int grid = (n + block_size - 1) / block_size;

  CHECK(cudaEventRecord(start));
  vec_add<<<grid, block_size>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0;
  CHECK(cudaEventElapsedTime(&ms, start, stop));

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return ms;
}

int main() {
  constexpr int N = 1'000'000;

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, N * sizeof(float)));
  CHECK(cudaMalloc(&db, N * sizeof(float)));
  CHECK(cudaMalloc(&dc, N * sizeof(float)));

  std::vector<float> temp(N, 1.0f);
  CHECK(cudaMemcpy(da, temp.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, temp.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  measure(256, da, db, dc, N);

  std::cout << "Block size\tTime (ms)\n";
  std::cout << "------------------------\n";

  for (int bs : {128, 256, 512}) {
    float t = measure(bs, da, db, dc, N);
    std::cout << bs << "\t\t" << t << "\n";
  }

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);
  return 0;
}