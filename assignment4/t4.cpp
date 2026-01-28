#include <cstdio>
#include <mpi.h>
#include <vector>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int my_rank = 0;
  int num_procs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  constexpr int GLOBAL_SIZE = 1000000;
  int chunk_size = GLOBAL_SIZE / num_procs;

  std::vector<float> my_chunk(chunk_size, 1.0f);

  double start_time = MPI_Wtime();

  float local_total = 0.0f;
  for (float val : my_chunk) {
    local_total += val;
  }

  float global_total = 0.0f;
  MPI_Reduce(&local_total, &global_total, 1, MPI_FLOAT, MPI_SUM, 0,
             MPI_COMM_WORLD);

  double end_time = MPI_Wtime();

  if (my_rank == 0) {
    printf("Distributed sum using MPI\n");
    printf("Number of processes : %d\n", num_procs);
    printf("Total elements      : %d\n", GLOBAL_SIZE);
    printf("Global sum          : %.1f\n", global_total);
    printf("Time                : %.4f seconds\n", end_time - start_time);
  }

  MPI_Finalize();
  return 0;
}