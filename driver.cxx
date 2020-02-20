#include <iostream>
#include "binary_sparse_matrix.h"
#include "CUDA/fast_cuda_mul.h"
#include "slow_mul.h"
#include "CUDA/quadtree.h"
#include "converter.h"
#include <random>
#include <chrono>
#include <bitset>
#include <fstream>

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
  int k = 64;
  // 32/8 = 4.
  // 32 -> 4 x 16 -> 4 x ( 4 x 8)

  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < k; i++)
    pairs.push_back({i, i});
  for (int i = 0; i < k; i++)
    pairs.push_back({i, k - 1 - i});


  quadtree qt = converter::build_quadtree_from_coo(pairs, k);

  std::cout << "Tree structure:\n";
  auto &s = qt.tree_structure_data;

  for (int i = 0; i < s.size(); i++) {
    std::cout << (i + 1) << ":[";
    for (int j = 0; j < 4; j++) {
      if (j)
        std::cout << ", ";
      if (s[i][j] & quadtree::TILE_REFERENCE)
        std::cout << "tile ";
      if (s[i][j])
        std::cout << (s[i][j] & (~quadtree::TILE_REFERENCE));
      else
        std::cout << "null";
    }
    std::cout << "]\n";
  };

  std::cout << "\nTiles:\n";
  for (int i = 0; i < qt.tiles.size(); i++) {
    std::cout << "tile " << i << ":\n";
    for (uint32_t row = 0; row < 8; row++)
      std::cout << std::bitset<8>(qt.tiles[i] >> (8u * row) & 0xff) << std::endl;
  }

  return 0;
}