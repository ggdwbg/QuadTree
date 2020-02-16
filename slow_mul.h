#ifndef QUADTREE_SLOW_MUL_H
#define QUADTREE_SLOW_MUL_H


#include "binary_sparse_matrix.h"

class slow_mul {
public:
  static binary_sparse_matrix multiply(const binary_sparse_matrix &a, const binary_sparse_matrix &b);
};


#endif //QUADTREE_SLOW_MUL_H
