#ifndef QUADTREE_CONVERTER_H
#define QUADTREE_CONVERTER_H


#include <vector>
#include "CUDA/quadtree.h"

class converter {
public:
  static quadtree build_quadtree_from_csr(const std::vector<int> &col_index, const std::vector<int> &row_index);
};


#endif //QUADTREE_CONVERTER_H
