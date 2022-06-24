#include <bits/stdc++.h>
#include <sys/time.h>
using namespace std;

#define LENGTHOF(arr) (sizeof(arr) / sizeof((arr)[0]))
#define matrix(i, j)  (A[(i) * (n) + (j)])
#define pmatrix(i, j) (A + ((i) * (n) + (j)))
#define prow(i)       (pmatrix(i, 0))

int num[] = {8,    32,   128,  256,  512,  1024, 1100, 1200,
             1300, 1400, 1500, 1600, 1700, 1800, 1900, 2048};
int rep[] = {100, 50, 15, 10, 10, 5, 5, 5, 5, 4, 4, 4, 4, 3, 3, 3};

float A[4194304];

void parallelGauss(float A[], int n) {
  for (int k = 0; k < n; k++) {
    for (int j = k + 1; j < n; j++) matrix(k, j) = matrix(k, j) / matrix(k, k);
    matrix(k, k) = 1.0;
    for (int i = k + 1; i < n; i++) {
      for (int j = k + 1; j < n; j++)
        matrix(i, j) = matrix(i, j) - matrix(k, j) * matrix(i, k);
      matrix(i, k) = 0;
    }
  }
}

void print(float A[], int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) cout << matrix(i, j) << " ";
    cout << endl;
  }
}

void setMatrix(float A[], int n) {
  uniform_real_distribution<float> dist(0, 100);
  mt19937                          mt(12345687);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) matrix(i, j) = dist(mt);
}

int main(int argc, char* argv[]) {
  freopen((argc > 1 ? argv[1] : "normal.out"), "w", stdout);
  int tot = LENGTHOF(num);
  for (int P = 0; P < tot; P++) {
    unsigned long tim = 0;
    for (int t = 0; t < rep[P]; t++) {
      setMatrix(A, num[P]);
      timeval start, end;
      gettimeofday(&start, nullptr);
      parallelGauss(A, num[P]);
      gettimeofday(&end, nullptr);
      tim +=
        1000000 * (end.tv_sec - start.tv_sec) + end.tv_usec - start.tv_usec;
    }
    //print(A, num[P]);
    cout << "N: " << num[P] << " time: " << tim / rep[P]
         << " (us) repeat: " << rep[P] << endl;
  }
}
