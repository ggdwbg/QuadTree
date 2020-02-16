#include "binary_sparse_matrix.h"
#include <cassert>
#include <iostream>

binary_sparse_matrix::binary_sparse_matrix(size_t _n, size_t _m) {
  n = _n, m = _m;
  data.resize(n);
}

void binary_sparse_matrix::set(int row, int col, bool v) {
  check_dim(row, col);
  auto &p = data[row];
  if (v)
    p.insert(col);
  else
    p.erase(col);
}

bool binary_sparse_matrix::get(int row, int col) const {
  check_dim(row, col);
  return data[row].count(col);
}

void binary_sparse_matrix::check_dim(int row, int col) const {
  assert(row >= 0 && row < n && col >= 0 && col < m);
}

bool binary_sparse_matrix::operator==(const binary_sparse_matrix &oth) const {
  if (oth.m != m || oth.n != n)
    return false;
  bool ret = true;
  for (int i = 0; i < n; i++)
    ret = ret && (data[i] == oth.data[i]);
  return ret;
}

std::ostream &operator<<(std::ostream &s, const binary_sparse_matrix &sm) {
  std::cout << sm.n << "*" << sm.m << " sparse matrix\n";
  for (int i = 0; i < sm.n; i++) {
    for (int j = 0; j < sm.m; j++)
      std::cout << sm.get(i, j) << ' ';
    std::cout << '\n';
  }
  return s;
}

