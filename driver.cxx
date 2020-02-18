#include <iostream>
#include "binary_sparse_matrix.h"
#include "CUDA/fast_cuda_mul.h"
#include "slow_mul.h"
#include "CUDA/quadtree.h"
#include "CUDA/converter.h"
#include <random>
#include <chrono>
#include <bitset>

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
  static_assert(sizeof(tile) == 1024);
  int nnz = 10000, szm = 1024;
  std::vector<int> col_indices(nnz), row_sind(szm);
  for (int i = 0; i < nnz; i++)
    col_indices[i] = gen() % szm;
  row_sind[0] = 0;
  for (int i = 1; i < szm; i++)
    row_sind[i] = row_sind[i - 1] + nnz / szm;
  for (auto p : col_indices)
    std::cout << p << ' ';
  std::cout << '\n';
  for (auto g : row_sind)
    std::cout << g << ' ';
  std::cout << '\n';

  auto qt = converter::build_quadtree_from_csr(col_indices, row_sind);
  std::cout << "tiles:\n";
  for (int i = 0; i < qt.tiles.size(); i++) {
    std::cout << "tile " << i << std::endl;
    for (int j = 0; j < 64; j++)
      std::cout << std::bitset<64>(qt.tiles[i].rows[j]) << '\n';
  }
  std::cout << "quadtree structure:" << std::endl;
  for (int i = 0; i < qt.tree_structure_data.size(); i++) {
    std::cout << "Children of node " << (i + 1) << ": ";
    for (int j = 0; j < 4; j++)
      std::cout << qt.tree_structure_data[i][j] << " ";
    std::cout << '\n';
  }
  /*
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
    std::cout << "Validation failed\n";*/
  return 0;
}