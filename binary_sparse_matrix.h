#ifndef QUADTREE_BINARY_SPARSE_MATRIX_H
#define QUADTREE_BINARY_SPARSE_MATRIX_H

#include <vector>
#include <unordered_set>

class binary_sparse_matrix {
public:
  // row <-> set of indices of 1-bits
  std::vector<std::unordered_set<size_t>> data;
  // n rows, m columns
  size_t n, m;

  binary_sparse_matrix(size_t _n, size_t _m);

  void set(int row, int col, bool v);

  bool get(int row, int col) const;

  bool operator==(const binary_sparse_matrix &oth) const;

  friend std::ostream &operator<<(std::ostream &s, const binary_sparse_matrix &sm);
private:
  void check_dim(int r, int c) const;
};


#endif //QUADTREE_BINARY_SPARSE_MATRIX_H
