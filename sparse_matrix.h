//
// Created by murch on 2/16/20.
//

#ifndef QUADTREE_SPARSE_MATRIX_H
#define QUADTREE_SPARSE_MATRIX_H

#include <vector>
#include <unordered_set>

class sparse_matrix {
public:
  // row <-> set of indices of 1-bits
  std::vector<std::unordered_set<size_t>> data;
  // n rows, m columns
  size_t n, m;

  sparse_matrix(size_t _n, size_t _m);

  void set(int row, int col, bool v);

  bool get(int row, int col);

  bool operator==(const sparse_matrix &oth);

private:
  void check_dim(int r, int c);
};


#endif //QUADTREE_SPARSE_MATRIX_H
