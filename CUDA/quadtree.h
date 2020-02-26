#ifndef QUADTREE_QUADTREE_H
#define QUADTREE_QUADTREE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

typedef uint16_t tile;

class quadtree {
public:
  static const uint32_t TILE_REFERENCE = 1u << 31u;
  static const uint32_t TILE_SIZE = 4; // tiles are TILE_SIZE x TILE_SIZE matrices
  std::vector<std::array<uint32_t, 4>> tree_structure_data;
  std::vector<tile> tiles;
};

#endif //QUADTREE_QUADTREE_H
