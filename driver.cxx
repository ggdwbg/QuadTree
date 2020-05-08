#include <iostream>
#include "CUDA/quadtree.h"
#include "converter.h"
#include <random>
#include <chrono>
#include <bitset>
#include <fstream>
#include <algorithm>

std::mt19937 gen;

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
#ifdef TEST
  std::cout << "works" << std::endl;
#endif
  int klog = 18;
  int k = 1 << klog;
  // 32/8 = 4.
  // 32 -> 4 x 16 -> 4 x ( 4 x 8)
  // 7 million nodes in a 1 mil x 1 mil matrix
  std::vector<std::pair<int, int>> pairs;
  for (int i = 0; i < k; i++)
    pairs.push_back({i, i});
  for (int i = 0; i < k; i++)
    pairs.push_back({i, k - 1 - i});
  for (int i = 0; i < k; i++)
    pairs.push_back({k / 2, i});
  for (int i = 0; i < k; i++)
    pairs.push_back({i, k / 2});
  for (int i = 0; i < k; i++)
    for (int s = 0; s < 8; s++)
      pairs.push_back({i, gen() % k});
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
  quadtree qt;

  measure("building quadtree", [&]() {
    qt = converter::build_quadtree_from_coo(pairs, klog);
  });
  std::cout << qt.tree_structure_data.size() << std::endl;
  /*std::cout << "n = " << k << ", m = " << pairs.size() << std::endl;
  for (int i = 0; i < qt.tree_structure_data.size(); i++) {
    std::cout << i << ": [";
    for (int j = 0; j < 4; j++) {
      if (j) std::cout << ", ";
      std::cout << qt.tree_structure_data[i][j];
    }
    std::cout << "]\n";
  }*/
  // std::cout << (16 * qt.tree_structure_data.size() + 2 * qt.tiles.size()) << " bytes in memory" << std::endl;

  return 0;
}