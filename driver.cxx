#include <iostream>
#include "binary_sparse_matrix.h"
#include "CUDA/fast_cuda_mul.h"
#include "slow_mul.h"
#include <random>
#include <chrono>

std::mt19937 gen;

binary_sparse_matrix generate_random_matrix(size_t n, size_t m, double fillrate) {
  binary_sparse_matrix ans(n, m);
  size_t points = (size_t) (n * m * fillrate);
  while (points--) {
    int row = gen() % n, col = gen() % m;
    ans.set(row, col, true);
  }
  return ans;
}

template<class Callback>
void measure(const std::string &activity, Callback f) {
  auto __start = std::chrono::system_clock::now();
  f();
  auto __end = std::chrono::system_clock::now();
  int ems = std::chrono::duration_cast
    <std::chrono::milliseconds>(__end - __start).count();
  std::cout << "Activity \"" << activity << "\" took " << ems << " ms." << std::endl;
}

int main(int argc, char** argv) {
  int n = 1 << 7;
  auto m1 = generate_random_matrix(n, n, 0.05);
  auto m2 = generate_random_matrix(n, n, 0.05);

  binary_sparse_matrix ans1(n, n), ans2(n, n);

  measure("fast CUDA multiplication", [&]() {
    ans1 = fast_cuda_mul::multiply(m1, m2);
  });
  measure("slow schoolbook multiplication", [&]() {
    ans2 = slow_mul::multiply(m1, m2);
  });

  //std::cout << m1 << m2 << ans2;

  bool ok = ans1 == ans2;
  if (ok)
    std::cout << "Validation passed\n";
  else
    std::cout << "Validation failed\n";
  return 0;
}