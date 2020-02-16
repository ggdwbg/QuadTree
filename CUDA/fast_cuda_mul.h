#ifndef QUADTREE_FAST_CUDA_MUL_H
#define QUADTREE_FAST_CUDA_MUL_H


#include "../binary_sparse_matrix.h"

class fast_cuda_mul {
public:
  static binary_sparse_matrix multiply(const binary_sparse_matrix &a, const binary_sparse_matrix &b);
};


#endif //QUADTREE_FAST_CUDA_MUL_H
