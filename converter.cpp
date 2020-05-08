#include <set>
#include <unordered_map>
#include <assert.h>
#include <algorithm>
#include <iostream>
#include <queue>
#include <stack>
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
      tiles[subid] |= 1u << (rsz * r + c);
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
  std::vector<std::vector<std::array<u32, 4>>> to_merge(4);
  u32 cur_shift = 0;
  for (auto &tmi : to_merge) {
    tmi = build(sub[cur_shift++], level - 1);
    reverse(tmi.begin(), tmi.end());
  }
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
  std::reverse(to_merge.begin(), to_merge.end());
  while (!to_merge.empty()) {
    auto &tmi = to_merge.back();
    while (!tmi.empty()) {
      auto cur = tmi.back();
      tmi.pop_back();
      for (u32 k = 0; k < 4; k++) {
        if (!cur[k]) continue;
        if (cur[k] & quadtree::LEAF_MASK) continue;
        cur[k] += shifts[cur_shift];
      }
      ret.push_back(cur);
    }
    to_merge.pop_back();
    ++cur_shift;
  }
#undef row
#undef col
  return ret;
}

quadtree converter::build_quadtree_from_coo(const std::vector<std::pair<int, int>> &els, int k) {
  return quadtree(k, build(els, k));
}

std::vector<std::pair<int, int>> converter::get_nnz_from_quadtree(const quadtree &qt) {
  std::vector<std::pair<int, int>> ans;
  std::stack<std::tuple
    <int, // vertex
      int, // level
      int, // offset_row
      int> // offset_col
  > dfs;
  dfs.emplace(0, qt.k, 0, 0);
  while (!dfs.empty()) {
    auto s = dfs.top();
    dfs.pop();
    int cur_v = std::get<0>(s), cur_k = std::get<1>(s),
      cur_shift_row = std::get<2>(s), cur_shift_col = std::get<3>(s);

    if (cur_k == quadtree::TILE_SZ_LOG + 1) {
      constexpr int tile_sz = 1u << quadtree::TILE_SZ_LOG;
      auto tiles = qt.tree_structure_data[cur_v];
      for (u32 gx = 0; gx < 2; gx++) {
        for (u32 gy = 0; gy < 2; gy++) {
          u32 cur_tile = tiles[gx * 2 + gy];
          for (u32 r = 0; r < tile_sz; r++) {
            for (u32 c = 0; c < tile_sz; c++) {
              if (cur_tile & (1u << (tile_sz * r + c))) {
                int cx = gx * tile_sz + cur_shift_row + r,
                  cy = gy * tile_sz + cur_shift_col + c;
                ans.emplace_back(cx, cy);
              }
            }
          }
        }
      }
    } else {
      int shift_add = 1u << (cur_k - 1);
      for (u32 gx = 0; gx < 2; gx++) {
        for (u32 gy = 0; gy < 2; gy++) {
          u32 gv = qt.tree_structure_data[cur_v][(gx << 1u) | gy];
          if (!gv)
            continue;
          dfs.emplace(gv, cur_k - 1, cur_shift_row + gx * shift_add, cur_shift_col + gy * shift_add);
        }
      }
    }
  }
  return ans;
}
