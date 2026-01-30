#include <iostream>
#include <mpi.h>
#include <vector>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank = 0, comm_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  constexpr long long TOTAL_ELEMENTS = 10'000'000LL;
  long long local_count = TOTAL_ELEMENTS / comm_size;

  std::vector<double> local_data(local_count, 1.0);

  double start_time = MPI_Wtime();

  double local_result = 0.0;
  for (double val : local_data) {
    local_result += val;
  }

  double total_result = 0.0;
  MPI_Reduce(&local_result, &total_result, 1, MPI_DOUBLE, MPI_SUM, 0,
             MPI_COMM_WORLD);

  double elapsed = MPI_Wtime() - start_time;

  if (rank == 0) {
    std::cout << "Number of processes: " << comm_size << "\n";
    std::cout << "Execution time:      " << elapsed << " s\n";
    std::cout << "Computed sum:        " << total_result << "\n";
  }

  MPI_Finalize();
  return 0;
}
