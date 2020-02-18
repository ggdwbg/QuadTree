#include <set>
#include <map>
#include <assert.h>
#include "converter.h"

#define row first
#define col second

template<class T, size_t N>
std::array<T, N> offset(std::array<T, N> in, T l) {
  for (int i = 0; i < N; i++)
    in[i] += l;
  return in;
}

// TODO: test if this works at all lol

// TODO: refactor and prettify

quadtree converter::build_quadtree_from_csr(const std::vector<int> &col_index, const std::vector<int> &row_index) {
  std::map<std::pair<uint32_t, uint32_t>, tile> raw_tiles; // TODO: unordered_map (with a good hash function on pairs) for truly linear time
  size_t matr_sz = row_index.size();
  assert((matr_sz & (matr_sz - 1)) == 0); // for now assume matr_sz is 2^k

  // step 1: construct nonzero tiles, level -1
  auto tile_sz = tile::TILE_SIZE;
  for (size_t i = 0; i < row_index.size(); i++) {
    int max = col_index.size();
    if (i != matr_sz - 1)
      max = row_index[i + 1];
    for (int k = row_index[i]; k < max; ++k) {
      auto j = col_index[k];
      auto &t = raw_tiles[{i / tile_sz, j / tile_sz}];
      auto a = i % tile_sz, b = j % tile_sz;
      t.rows[a] |= (1ull << b);
      t.cols[b] |= (1ull << a);
    }
  }
  quadtree ans;
  ans.tiles.resize(raw_tiles.size());

  size_t tile_id = 0;
  // level 0 connects to tiles, levels k, k > 0 connect to lvl (k-1)
  std::vector<std::map<std::pair<int, int>, std::array<uint32_t, 4>>> levels(1);
  for (auto &p : raw_tiles) {
    auto coord = p.first;
    auto &flvl = levels[0][{coord.row / 2, coord.col / 2}];
    // flvl[0] - UL, flvl[1] - UR, flvl[2] - DL, flfl[3] - DR
    flvl[((coord.row & 1u) << 1u) + (coord.col & 1u)] = tile_id | quadtree::TILE_CHILD_MASK;
    ans.tiles[tile_id] = p.second;
    ++tile_id;
  }

  /* meta-matrix corresponding to level k:
   * tiles (level -1) correspond to matrix of size matr_sz / tile_sz
   * level 0 corresponds to size matr_sz / (tile_sz << 1)
   * ...
   * level k corresponds to size matr_sz / (tile_sz << (k+1))
   */
  // (row, col) <-> index in the current layer
  std::vector<std::vector<std::array<uint32_t, 4>>> layers_not_linearized;
  size_t k = 0;
  while (matr_sz != (tile_sz << (k + 1))) {
    // kth level -> (k+1)th
    levels.emplace_back();
    layers_not_linearized.emplace_back();
    auto &k_layer = layers_not_linearized.back();
    int s = 0;
    for (auto &p : levels[k]) {
      auto coord = p.first;
      auto parent_coord = std::make_pair(coord.row / 2, coord.col / 2);
      auto &flvl = levels[k + 1][parent_coord];
      // flvl[0] - UL, flvl[1] - UR, flvl[2] - DL, flfl[3] - DR
      flvl[((coord.row & 1u) << 1u) + (coord.col & 1u)] = s;
      k_layer.emplace_back(p.second);
      ++s;
    }
    ++k;
  }
  /*
   * quadtree is almost built
   * final task is to `linearize` indices in all layers except 0-th
   * (it connects to tiles, and that we already have figured out)
   */
  layers_not_linearized.emplace_back();
  layers_not_linearized.back().emplace_back(levels.back()[{0, 0}]);
  auto &p = ans.tree_structure_data;
  uint32_t cumul_ind = 1; // indices start with 1
  int tls = layers_not_linearized.size();
  for (int j = tls - 1; j >= 0; j--) {
    auto &cur_layer = layers_not_linearized[j];
    cumul_ind += cur_layer.size();
    for (auto el : cur_layer)
      if (j == 0)
        p.emplace_back(el);
      else
        p.emplace_back(offset(el, cumul_ind));
  }

  return ans;
}
