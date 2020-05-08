#include <set>
#include <unordered_map>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include "converter.h"

using u32 = uint32_t;
typedef std::pair<int, int> point;

std::vector<std::array<u32, 4>> build(const std::vector<point> &points, size_t level) {
#define row first
#define col second
  std::vector<std::array<u32, 4>> ret;
  if (points.empty())
    return ret;
  if (level == quadtree::TILE_SZ_LOG + 1) {
    u32 rsz = 1u << quadtree::TILE_SZ_LOG;
    std::array<u32, 4> tiles = {0};
    for (auto p : points) {
      u32 r = p.row, c = p.col;
      u32 subid = ((r / rsz) << 1u) | (c / rsz);
      r &= rsz - 1, c &= rsz - 1;
      tiles[subid] |= 1u << (4u * r + c);
    }
    for (auto &t : tiles)
      t |= quadtree::LEAF_MASK;
    return {tiles};
  }
  u32 tkm1 = 1u << (level - 1);
  std::array<std::vector<point>, 4> sub;
  for (auto p : points) {
    u32 r = p.row, c = p.col;
    u32 subid = ((r / tkm1) << 1u) | (c / tkm1);
    r &= tkm1 - 1, c &= tkm1 - 1;
    sub[subid].push_back({r, c});
  }
  std::array<std::vector<std::array<u32, 4>>, 4> to_merge;
  u32 cur_shift = 0;
  for (auto &tmi : to_merge)
    tmi = build(sub[cur_shift++], level - 1);
  std::array<u32, 4> shifts = {1};
  for (int i = 1; i < 4; i++)
    shifts[i] = shifts[i - 1] + to_merge[i - 1].size();
  ret.emplace_back();
  cur_shift = 0;
  for (auto &tmi : to_merge) {
    ret.back()[cur_shift] = tmi.empty() ? 0 : shifts[cur_shift];
    ++cur_shift;
  }
  cur_shift = 0;
  for (auto &tmi : to_merge) {
    std::for_each(tmi.begin(), tmi.end(),
                  [&](std::array<u32, 4> &cur) {
                    for (u32 k = 0; k < 4; k++) {
                      if (!cur[k]) continue;
                      if (cur[k] & quadtree::LEAF_MASK) continue;
                      cur[k] += shifts[cur_shift];
                    }
                    ret.push_back(cur);
                  });
    ++cur_shift;
  }
#undef row
#undef col
  return ret;
}

quadtree converter::build_quadtree_from_csr(const std::vector<int> &col_index, const std::vector<int> &row_index) {
  size_t matr_sz = row_index.size();
  /* prep_consts(matr_sz);

   auto tile_sz = quadtree::TILE_SIZE;

   for (size_t i = 0; i < row_index.size(); i++) {
     int max = col_index.size();
     if (i != matr_sz - 1)
       max = row_index[i + 1];
     for (int k = row_index[i]; k < max; ++k) {
       auto j = col_index[k];
       add_point(i, j);
     }
   }
   construct_tree();
   dfs(root);

   return finalize();*/
  return quadtree();
}

quadtree converter::build_quadtree_from_coo(const std::vector<std::pair<int, int>> &els, int k) {
  quadtree ret;
  ret.tree_structure_data = build(els, k);
  return ret;
}
