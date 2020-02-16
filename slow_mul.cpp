#include "slow_mul.h"
#include <cassert>

binary_sparse_matrix slow_mul::multiply(const binary_sparse_matrix &a, const binary_sparse_matrix &b) {
  assert(a.m == b.n);
  binary_sparse_matrix ans(a.n, b.m);

  for (int i = 0; i < a.n; i++)
    for (int j = 0; j < b.m; j++) {
      bool v = false;
      for (int k = 0; k < a.m && !v; k++)
        v = v || (a.get(i, k) && b.get(k, j));
      ans.set(i, j, v);
    }

  return ans;
}
