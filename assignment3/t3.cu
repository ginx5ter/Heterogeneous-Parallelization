#include <cstdio>
#include <cuda_runtime.h>


__global__ void nice_access(float *buf, int length) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  if (pos < length)
    buf[pos] += 0.5f;
}

__global__ void bad_access(float *buf, int length) {
  int pos = blockIdx.x * blockDim.x + threadIdx.x;
  int scattered = (pos * 17) % length; // сильное расхождение
  if (pos < length)
    buf[scattered] += 0.5f;
}

float timing(float *dev, int n, void (*func)(float *, int)) {
  cudaEvent_t st, en;
  cudaEventCreate(&st);
  cudaEventCreate(&en);

  constexpr int BS = 256;
  int GS = (n + BS - 1) / BS;

  cudaEventRecord(st);
  func<<<GS, BS>>>(dev, n);
  cudaEventRecord(en);
  cudaEventSynchronize(en);

  float ms = 0.f;
  cudaEventElapsedTime(&ms, st, en);

  cudaEventDestroy(st);
  cudaEventDestroy(en);
  return ms;
}

int main() {
  constexpr int N = 1'000'000;
  float *d_ptr = nullptr;
  cudaMalloc(&d_ptr, N * sizeof(float));

  float t_good = timing(d_ptr, N, nice_access);
  float t_bad = timing(d_ptr, N, bad_access);

  printf("Coalesced   access : %.3f ms\n", t_good);
  printf("Uncoalesced access : %.3f ms\n", t_bad);

  cudaFree(d_ptr);
  return 0;
}