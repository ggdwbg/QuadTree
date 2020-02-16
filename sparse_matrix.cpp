//
// Created by murch on 2/16/20.
//

#include "sparse_matrix.h"
#include <cassert>

sparse_matrix::sparse_matrix(size_t _n, size_t _m) {
  n = _n, m = _m;
  data.resize(n);
}

void sparse_matrix::set(int row, int col, bool v) {
  check_dim(row, col);
  auto &p = data[row];
  if (v)
    p.insert(col);
  else
    p.erase(col);
}

bool sparse_matrix::get(int row, int col) {
  check_dim(row, col);
  return data[row].count(col);
}

void sparse_matrix::check_dim(int row, int col) {
  assert(row >= 0 && row < n && col >= 0 && col < m);
}

bool sparse_matrix::operator==(const sparse_matrix &oth) {
  if (oth.m != m || oth.n != n)
    return false;
  bool ret = true;
  for (int i = 0; i < n; i++)
    ret = ret && (data[i] == oth.data[i]);
  return ret;
}
