#include <cmath>
#include <iostream>
#include <mpi.h>
#include <random>
#include <vector>

int read_matrix_size(int argc, char **argv) {
  return (argc >= 2) ? std::atoi(argv[1]) : 512;
}

void generate_dominant_diagonal_system(std::vector<double> &mat,
                                       std::vector<double> &rhs, int n) {
  std::mt19937_64 gen{123};
  std::uniform_real_distribution<double> uniform{-1.0, 1.0};

  for (int i = 0; i < n; ++i) {
    double abs_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      double val = uniform(gen);
      mat[size_t(i) * n + j] = val;
      abs_sum += std::abs(val);
    }
    mat[size_t(i) * n + i] = abs_sum + 1.2;
    rhs[i] = uniform(gen);
  }
}

int find_row_owner(int global_row, const std::vector<int> &counts,
                   const std::vector<int> &displs) {
  for (size_t p = 0; p < counts.size(); ++p) {
    if (global_row >= displs[p] && global_row < displs[p] + counts[p]) {
      return p;
    }
  }
  return -1;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int n = read_matrix_size(argc, argv);
  if (n <= 0 && rank == 0) {
    std::cerr << "Matrix size must be positive\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  double t0 = MPI_Wtime();

  int n_padded = n;
  if (n % np != 0) {
    n_padded = ((n / np) + 1) * np;
    if (rank == 0) {
      std::cout << "Padding matrix from " << n << " → " << n_padded << "\n";
    }
  }
  int local_rows = n_padded / np;

  std::vector<double> A_global, b_global;
  if (rank == 0) {
    A_global.assign(size_t(n_padded) * n_padded, 0.0);
    b_global.assign(size_t(n_padded), 0.0);

    std::vector<double> A_orig(n * n);
    std::vector<double> b_orig(n);
    generate_dominant_diagonal_system(A_orig, b_orig, n);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        A_global[size_t(i) * n_padded + j] = A_orig[size_t(i) * n + j];
      }
      b_global[i] = b_orig[i];
    }
    for (int i = n; i < n_padded; ++i) {
      A_global[size_t(i) * n_padded + i] = 1.0;
    }
  }

  std::vector<double> A_my(local_rows * n_padded, 0.0);
  std::vector<double> b_my(local_rows, 0.0);

  MPI_Scatter(A_global.data(), local_rows * n_padded, MPI_DOUBLE, A_my.data(),
              local_rows * n_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatter(b_global.data(), local_rows, MPI_DOUBLE, b_my.data(), local_rows,
              MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<int> row_counts(np, local_rows);
  std::vector<int> row_offsets(np, 0);
  for (int i = 1; i < np; ++i) {
    row_offsets[i] = row_offsets[i - 1] + row_counts[i - 1];
  }

  std::vector<double> pivot_buffer(n_padded + 1);

  for (int col = 0; col < n; ++col) {
    int pivot_owner = find_row_owner(col, row_counts, row_offsets);
    int local_pivot_idx = col - row_offsets[pivot_owner];

    if (rank == pivot_owner) {
      double piv = A_my[size_t(local_pivot_idx) * n_padded + col];
      if (std::abs(piv) < 1e-14) {
        std::cerr << "Small pivot at column " << col << "\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
      }

      double inv_piv = 1.0 / piv;
      for (int j = col; j < n; ++j) {
        double val = A_my[size_t(local_pivot_idx) * n_padded + j] * inv_piv;
        pivot_buffer[j] = val;
        A_my[size_t(local_pivot_idx) * n_padded + j] = val;
      }
      pivot_buffer[n_padded] = b_my[local_pivot_idx] * inv_piv;
      b_my[local_pivot_idx] = pivot_buffer[n_padded];
    }

    MPI_Bcast(pivot_buffer.data(), n_padded + 1, MPI_DOUBLE, pivot_owner,
              MPI_COMM_WORLD);

    for (int lr = 0; lr < local_rows; ++lr) {
      int global_r = row_offsets[rank] + lr;
      if (global_r <= col || global_r >= n)
        continue;

      double coeff = A_my[size_t(lr) * n_padded + col];
      if (std::abs(coeff) < 1e-16)
        continue;

      for (int j = col; j < n; ++j) {
        A_my[size_t(lr) * n_padded + j] -= coeff * pivot_buffer[j];
      }
      b_my[lr] -= coeff * pivot_buffer[n_padded];
      A_my[size_t(lr) * n_padded + col] = 0.0;
    }
  }

  if (rank == 0) {
    A_global.assign(size_t(n_padded) * n_padded, 0.0);
    b_global.assign(size_t(n_padded), 0.0);
  }

  MPI_Gather(A_my.data(), local_rows * n_padded, MPI_DOUBLE, A_global.data(),
             local_rows * n_padded, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Gather(b_my.data(), local_rows, MPI_DOUBLE, b_global.data(), local_rows,
             MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::vector<double> solution(n);

    for (int i = n - 1; i >= 0; --i) {
      double s = b_global[i];
      for (int j = i + 1; j < n; ++j) {
        s -= A_global[size_t(i) * n_padded + j] * solution[j];
      }
      solution[i] = s / A_global[size_t(i) * n_padded + i];
    }

    double t1 = MPI_Wtime();

    std::cout << "Size: " << n << "    Processes: " << np << "\n";
    std::cout << "Time: " << (t1 - t0) << " s\n";

    int preview = std::min(n, 8);
    std::cout << "First " << preview << " solution components:\n";
    for (int i = 0; i < preview; ++i) {
      std::cout << "  x[" << i << "] = " << solution[i] << "\n";
    }
  }

  MPI_Finalize();
  return 0;
}