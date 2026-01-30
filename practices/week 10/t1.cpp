#include <iostream>
#include <omp.h>
#include <random>
#include <vector>

int main() {
  constexpr int ARRAY_SIZE = 10'000'000;
  std::vector<double> values(ARRAY_SIZE);

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  for (auto &v : values) {
    v = uniform(rng);
  }

  double seq_time = 0.0;
  double seq_partial = 0.0;

  {
    double start = omp_get_wtime();
    for (int i = 0; i < ARRAY_SIZE / 10; ++i) {
      seq_partial += values[i];
    }
    seq_time = omp_get_wtime() - start;
  }

  double parallel_sum = 0.0;
  double parallel_time = 0.0;

  {
    double t0 = omp_get_wtime();
#pragma omp parallel for reduction(+ : parallel_sum)
    for (int i = ARRAY_SIZE / 10; i < ARRAY_SIZE; ++i) {
      parallel_sum += values[i];
    }
    parallel_time = omp_get_wtime() - t0;
  }

  std::cout << "Number of threads: " << omp_get_max_threads() << "\n";
  std::cout << "Sequential part:   " << seq_time << " seconds\n";
  std::cout << "Parallel part:     " << parallel_time << " seconds\n";
  std::cout << "Final sum:         " << (seq_partial + parallel_sum) << "\n";

  return 0;
}