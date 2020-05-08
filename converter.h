#ifndef QUADTREE_CONVERTER_H
#define QUADTREE_CONVERTER_H


#include <vector>
#include "CUDA/quadtree.h"

class converter {
public:
  static quadtree build_quadtree_from_csr(const std::vector<int> &col_index, const std::vector<int> &row_index);

  static std::vector<std::pair<int, int>> get_nnz_from_quadtree(const quadtree &qt);
  static quadtree build_quadtree_from_coo(const std::vector<std::pair<int, int>> &els, int n);
};


#endif //QUADTREE_CONVERTER_H
