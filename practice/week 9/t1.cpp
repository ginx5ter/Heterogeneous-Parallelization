#include <cmath>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

long long get_problem_size(int argc, char **argv) {
  return (argc >= 2) ? std::atoll(argv[1]) : 1'000'000LL;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  long long total_elements = get_problem_size(argc, argv);
  if (total_elements <= 0) {
    if (rank == 0)
      std::cerr << "Invalid data size\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double t_start = MPI_Wtime();

  std::vector<int> chunk_sizes(nprocs);
  std::vector<int> offsets(nprocs);

  long long base_chunk = total_elements / nprocs;
  long long remainder = total_elements % nprocs;

  for (int p = 0; p < nprocs; ++p) {
    chunk_sizes[p] = base_chunk + (p < remainder ? 1 : 0);
  }

  offsets[0] = 0;
  for (int p = 1; p < nprocs; ++p) {
    offsets[p] = offsets[p - 1] + chunk_sizes[p - 1];
  }

  std::vector<double> global_data;
  if (rank == 0) {
    global_data.resize(total_elements);
    std::mt19937_64 engine{42};
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
    for (auto &val : global_data) {
      val = uniform(engine);
    }
  }

  int my_count = chunk_sizes[rank];
  std::vector<double> my_chunk(my_count);

  MPI_Scatterv(rank == 0 ? global_data.data() : nullptr, chunk_sizes.data(),
               offsets.data(), MPI_DOUBLE, my_chunk.data(), my_count,
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double sum_local = 0.0;
  double sum_squares_local = 0.0;

  for (double v : my_chunk) {
    sum_local += v;
    sum_squares_local += v * v;
  }

  double total_sum = 0.0;
  double total_sum_sq = 0.0;

  MPI_Reduce(&sum_local, &total_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&sum_squares_local, &total_sum_sq, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  double t_end = MPI_Wtime();

  if (rank == 0) {
    double avg = total_sum / total_elements;
    double variance = (total_sum_sq / total_elements) - (avg * avg);
    if (variance < 0)
      variance = 0.0; // защита от погрешности
    double std_dev = std::sqrt(variance);

    std::cout << "Elements: " << total_elements << "    Processes: " << nprocs
              << "\n";
    std::cout << "Mean:     " << avg << "\n";
    std::cout << "Std dev:  " << std_dev << "\n";
    std::cout << "Time:     " << (t_end - t_start) << " s\n";
  }

  MPI_Finalize();
  return 0;
}