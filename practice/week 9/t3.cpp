#include <iomanip>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <random>
#include <vector>

int get_graph_size(int argc, char **argv) {
  return (argc >= 2) ? std::atoi(argv[1]) : 256;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  int n = get_graph_size(argc, argv);
  if (n <= 0 && rank == 0) {
    std::cerr << "Invalid graph size\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int n_padded = n;
  if (n % nprocs != 0) {
    n_padded = ((n / nprocs) + 1) * nprocs;
    if (rank == 0) {
      std::cout << "Padding to " << n_padded << " for even distribution\n";
    }
  }
  int local_row_count = n_padded / nprocs;

  const int NO_PATH = 1'000'000'000;

  double t_begin = MPI_Wtime();

  std::vector<int> dist_matrix_global;
  if (rank == 0) {
    dist_matrix_global.assign(size_t(n_padded) * n_padded, NO_PATH);

    std::mt19937 gen{7};
    std::uniform_int_distribution<int> weight{1, 20};
    std::uniform_real_distribution<double> prob{0.0, 1.0};

    for (int i = 0; i < n; ++i) {
      dist_matrix_global[size_t(i) * n_padded + i] = 0;
      for (int j = 0; j < n; ++j) {
        if (i != j && prob(gen) < 0.22) {
          dist_matrix_global[size_t(i) * n_padded + j] = weight(gen);
        }
      }
    }
    for (int i = n; i < n_padded; ++i) {
      dist_matrix_global[size_t(i) * n_padded + i] = 0;
    }
  }

  std::vector<int> my_rows(local_row_count * n_padded, NO_PATH);

  MPI_Scatter(dist_matrix_global.data(), local_row_count * n_padded, MPI_INT,
              my_rows.data(), local_row_count * n_padded, MPI_INT, 0,
              MPI_COMM_WORLD);

  std::vector<int> dist_global(n_padded * n_padded, NO_PATH);

  MPI_Allgather(my_rows.data(), local_row_count * n_padded, MPI_INT,
                dist_global.data(), local_row_count * n_padded, MPI_INT,
                MPI_COMM_WORLD);

  for (int k = 0; k < n; ++k) {
    const int *via_k = &dist_global[size_t(k) * n_padded];

    for (int lr = 0; lr < local_row_count; ++lr) {
      int global_i = rank * local_row_count + lr;
      if (global_i >= n)
        break;

      int *row_i = &my_rows[size_t(lr) * n_padded];
      int d_ik = row_i[k];
      if (d_ik >= NO_PATH / 2)
        continue;

      for (int j = 0; j < n; ++j) {
        long long candidate = (long long)d_ik + via_k[j];
        if (candidate < row_i[j]) {
          row_i[j] = (candidate < NO_PATH) ? candidate : NO_PATH;
        }
      }
    }

    MPI_Allgather(my_rows.data(), local_row_count * n_padded, MPI_INT,
                  dist_global.data(), local_row_count * n_padded, MPI_INT,
                  MPI_COMM_WORLD);
  }

  double t_finish = MPI_Wtime();

  if (rank == 0) {
    std::cout << "Nodes: " << n << "    Processes: " << nprocs << "\n";
    std::cout << "Time:  " << (t_finish - t_begin) << " s\n";

    int preview_size = std::min(n, 10);
    std::cout << "Top-left " << preview_size << "x" << preview_size
              << " distances:\n";
    for (int i = 0; i < preview_size; ++i) {
      for (int j = 0; j < preview_size; ++j) {
        int d = dist_global[size_t(i) * n_padded + j];
        if (d >= NO_PATH / 2)
          std::cout << "  inf ";
        else
          std::cout << std::right << std::setw(4) << d << " ";
      }
      std::cout << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}