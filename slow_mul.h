//
// Created by murch on 2/16/20.
//

#ifndef QUADTREE_SLOW_MUL_H
#define QUADTREE_SLOW_MUL_H


#include "sparse_matrix.h"

class slow_mul {
public:
  static sparse_matrix multiply(const sparse_matrix &a, const sparse_matrix &b);
};


#endif //QUADTREE_SLOW_MUL_H
