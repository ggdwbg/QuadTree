#include <iostream>
#include <cassert>
#include "sparse_matrix.h"
#include "fast_cuda_mul.h"
#include "slow_mul.h"
#include <random>

std::mt19937 gen;

sparse_matrix generate_random_matrix(size_t n, size_t m, double fillrate) {
  sparse_matrix ans(n, m);
  size_t points = (size_t) (n * m * fillrate);
  while (points--) {
    int row = gen() % n, col = gen() % m;
    ans.set(row, col, true);
  }
  return ans;
}

int main(int argc, char** argv) {
  int n = 64;
  auto m1 = generate_random_matrix(n, n, 0.15);
  auto m2 = generate_random_matrix(n, n, 0.15);

  auto ans1 = fast_cuda_mul::multiply(m1, m2), ans2 = slow_mul::multiply(m1, m2);

  bool ok = ans1 == ans2;
  if (ok)
    std::cout << "Validation passed\n";
  else
    std::cout << "Validation failed\n";
  return 0;
}