#include <omp.h>
#include <bits/stdc++.h>
#include <mpi.h>
#include <sys/time.h>
using namespace std;

#define THREADS 4

#define LENGTHOF(arr) (sizeof(arr) / sizeof((arr)[0]))
#define matrix(i, j)  (A[(i) * (n) + (j)])
#define pmatrix(i, j) (A + ((i) * (n) + (j)))
#define prow(i)       (pmatrix(i, 0))

int num[] = {8,    32,   128,  256,  512,  1024, 1100, 1200,
             1300, 1400, 1500, 1600, 1700, 1800, 1900, 2048};
int rep[] = {100, 50, 15, 10, 10, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3};

float A[4194304];

ostringstream coutBuffer;

void printCoutBuffer() {
  std::cout << coutBuffer.str();
}
#define cout coutBuffer

void parallelGauss(float A[], int n) {
  int comm_sz;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  MPI_Bcast(A, n * n, MPI_FLOAT, 0, MPI_COMM_WORLD);

  int block_sz  = n / comm_sz;
  int row_begin = block_sz * my_rank;
  int row_end   = (my_rank + 1 == comm_sz ? n : row_begin + block_sz);

  int   i, j, k;
  int   bc_rank;
  float tmp;
#pragma omp parallel num_threads(THREADS), private(i, j, k, tmp, bc_rank)
  for (k = 0; k < n; ++k) {
    if (row_begin <= k && k < row_end) {
      tmp = matrix(k, k);
#pragma omp for
      for (j = k + 1; j < n; ++j) matrix(k, j) /= tmp;
      matrix(k, k) = 1.0;
    }
    bc_rank = comm_sz - 1;
    if (block_sz && k / block_sz < bc_rank) bc_rank = k / block_sz;
#pragma omp barrier
    if (omp_get_thread_num() == 0)
      MPI_Bcast(prow(k), n, MPI_FLOAT, bc_rank, MPI_COMM_WORLD);
#pragma omp barrier
#pragma omp for
    for (i = max(row_begin, k + 1); i < row_end; ++i) {
      tmp = matrix(i, k);
      for (j = k + 1; j < n; ++j) matrix(i, j) -= tmp * matrix(k, j);
      matrix(i, k) = 0;
    }
  }
}

void setMatrix(float A[], int n) {
  uniform_real_distribution<float> dist(0, 100);
  mt19937                          mt(12345687);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) matrix(i, j) = dist(mt);
}

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int comm_sz;
  int my_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  if (my_rank == 0) {
    freopen((argc > 1 ? argv[1] : "MPI_openmp.out"), "w", stdout);
    atexit(printCoutBuffer);
  }
  int tot = LENGTHOF(num);
  for (int P = 0; P < tot; P++) {
    unsigned long tim = 0;
    for (int t = 0; t < rep[P]; t++) {
      setMatrix(A, num[P]);
      timeval start, end;
      MPI_Barrier(MPI_COMM_WORLD);
      gettimeofday(&start, nullptr);
      parallelGauss(A, num[P]);
      gettimeofday(&end, nullptr);
      tim +=
        1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    }
    // print(A, num[P]);
    cout << "N: " << num[P] << " time: " << tim / rep[P]
         << " (us) repeat: " << rep[P] << endl;
  }
  exit(MPI_Finalize());
}