#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << " (" << err    \
                << ") at " << __FILE__ << ":" << __LINE__ << "\n";             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

__global__ void vec_add(const float *__restrict__ a,
                        const float *__restrict__ b, float *__restrict__ c,
                        int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + b[i];
}

float benchmark(int block_size, float *da, float *db, float *dc, int n,
                int repeats = 8) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));

  int grid = (n + block_size - 1) / block_size;

  vec_add<<<grid, block_size>>>(da, db, dc, n);
  CHECK(cudaDeviceSynchronize());

  float total_ms = 0.0f;

  for (int i = 0; i < repeats; ++i) {
    CHECK(cudaEventRecord(start));
    vec_add<<<grid, block_size>>>(da, db, dc, n);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    total_ms += ms;
  }

  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));

  return total_ms / repeats;
}

int main(int argc, char **argv) {
  int N = 1'000'000;
  if (argc >= 2) {
    N = std::atoi(argv[1]);
    if (N < 10000)
      N = 10000;
  }

  std::vector<float> ha(N, 1.1f);
  std::vector<float> hb(N, 2.2f);

  float *da = nullptr, *db = nullptr, *dc = nullptr;
  CHECK(cudaMalloc(&da, N * sizeof(float)));
  CHECK(cudaMalloc(&db, N * sizeof(float)));
  CHECK(cudaMalloc(&dc, N * sizeof(float)));

  CHECK(cudaMemcpy(da, ha.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, hb.data(), N * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<int> block_sizes = {64, 128, 192, 256, 384, 512, 768, 1024};

  std::cout << "\n  block   |   grid   |   avg time (ms)\n";
  std::cout << "------------------------------------------\n";

  float best_time = std::numeric_limits<float>::max();
  float worst_time = 0.0f;
  int best_bs = 0;
  int worst_bs = 0;

  for (int bs : block_sizes) {
    int grid = (N + bs - 1) / bs;
    float t = benchmark(bs, da, db, dc, N);

    std::cout << std::right << std::setw(7) << bs << "  |  " << std::setw(6)
              << grid << "  |  " << std::fixed << std::setprecision(3)
              << std::setw(10) << t << " ms\n";

    if (t < best_time) {
      best_time = t;
      best_bs = bs;
    }
    if (t > worst_time) {
      worst_time = t;
      worst_bs = bs;
    }
  }

  std::cout << "\nResults:\n";
  std::cout << "Best: block = " << best_bs << ", time = " << std::fixed
            << std::setprecision(3) << best_time << " ms\n";
  std::cout << "Worst: block = " << worst_bs << ", time = " << std::fixed
            << std::setprecision(3) << worst_time << " ms\n";

  CHECK(cudaFree(da));
  CHECK(cudaFree(db));
  CHECK(cudaFree(dc));

  return 0;
}