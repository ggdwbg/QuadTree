#ifndef QUADTREE_QUADTREE_H
#define QUADTREE_QUADTREE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

typedef uint16_t tile;

class quadtree {
public:
  static const uint32_t TILE_SZ_LOG = 2;
  static const uint32_t LEAF_MASK = (1u << 31u);

  /* Stores euler tour of the tree
   * that is obtained via dfs traversing the tree
   * with the following order on children: 0th is upper left,
   * 1st is upper right, 2nd is the lower left, 3rd is lower right.
   * if tree_structure_data[i][j] ($ j \in { 0, 1, 2, 3 } $) is zero,
   * then $ j $-th child of vertex $ i $ doesn't contain a single non-zero entry.
   * */
  std::vector<std::array<uint32_t, 4>> tree_structure_data;
};

#endif //QUADTREE_QUADTREE_H
