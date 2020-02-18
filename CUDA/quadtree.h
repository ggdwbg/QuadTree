#ifndef QUADTREE_QUADTREE_H
#define QUADTREE_QUADTREE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

class tile {
public:
  static const size_t TILE_SIZE = 64;
  uint64_t rows[TILE_SIZE]; // uint{TILE_SIZE/8}_t
  uint64_t cols[TILE_SIZE];
};

class quadtree {
public:
  // 0b00......000  -> NULL
  // 0b10......ind  -> index in tiles (indices start with 0)
  // otherwise index of a child in tree_structure_data (indices start with 1)
  static const uint32_t NO_CHILD = 0;
  static const uint32_t TILE_CHILD_MASK = 1u << 31u;
  std::vector<std::array<uint32_t, 4>> tree_structure_data;
  std::vector<tile> tiles;
};

#endif //QUADTREE_QUADTREE_H
