//
// Created by murch on 2/16/20.
//

#include "slow_mul.h"

sparse_matrix slow_mul::multiply(const sparse_matrix &a, const sparse_matrix &b) {
  auto ans = sparse_matrix(1, 1);
  ans.set(0, 0, 1);
  return ans;
}
