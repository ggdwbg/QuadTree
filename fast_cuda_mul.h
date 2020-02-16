//
// Created by murch on 2/16/20.
//

#ifndef QUADTREE_FAST_CUDA_MUL_H
#define QUADTREE_FAST_CUDA_MUL_H


#include "sparse_matrix.h"

class fast_cuda_mul {
public:
  static sparse_matrix multiply(const sparse_matrix &a, const sparse_matrix &b);
};


#endif //QUADTREE_FAST_CUDA_MUL_H
