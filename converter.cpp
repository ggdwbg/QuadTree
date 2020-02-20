#include <set>
#include <unordered_map>
#include <assert.h>
#include <iostream>
#include "converter.h"

#define row first
#define col second

typedef std::pair<uint32_t, uint32_t> coord;

uint64_t splitmix64(uint64_t x) {
  // http://xorshift.di.unimi.it/splitmix64.c
  x += 0x9e3779b97f4a7c15ll;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ll;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebll;
  return x ^ (x >> 31);
}

/*
 * A (hopefully) good hash function on pairs.
 */
template<>
struct std::hash<coord> {
  size_t operator()(std::pair<uint32_t, uint32_t> p) const {
    return splitmix64(((uint64_t) p.row << 32ull) + p.col);
  }
};

template<>
struct std::hash<void *> {
  size_t operator()(void *p) const {
    return splitmix64((uint64_t) p);
  }
};

std::unordered_map<void *, int> v_to_depth;
/*
 * Leaves <=> nodes of depth $ levels $ relative to $ root $.
 */
size_t levels = 0;

struct qt_node {
  /*
   * Leaves are quad_tree nodes that have tiles as their children
   * if this node is not a leaf, sub[j] is a qt_node*
   * otherwise it is a tile*
   * sub[0]: upper left
   * sub[1]: upper right
   * sub[2]: lower left
   * sub[3]: lower right
   */
  void *sub[4] = {nullptr};

  qt_node() {
    //std::cout << "node constructed" << std::endl;
  }

  ~qt_node() {
    bool leaf = v_to_depth[this] == levels;
    for (auto v : sub)
      if (!leaf)
        delete (qt_node *) v;
      else
        delete (tile *) v;
  }
};

/*
 * Suppose we have constructed the tree.
 * Now we need to store it as a contiguous array in memory.
 * This is done by storing the euler tour of the tree.
 */

std::unordered_map<void *, int> euler_tour_ids, tile_ids;

int euler_time = 0, leaf_time = 0;

void dfs(qt_node *v, int depth = 1) {
  v_to_depth[v] = depth;
  euler_tour_ids[v] = euler_time++;
  if (depth == levels) {
    for (auto p : v->sub)
      if (p)
        tile_ids[p] = leaf_time++;
    return;
  }
  for (auto p : v->sub) {
    if (!p)
      continue;
    dfs((qt_node *) p, depth + 1);
  }
}

std::unordered_map<coord, tile *> raw_tiles;

qt_node *root = nullptr;

void add_point(int i, int j) {
  auto tile_sz = quadtree::TILE_SIZE;
  auto tid = std::make_pair(i / tile_sz, j / tile_sz);
  if (!raw_tiles.count(tid))
    raw_tiles[tid] = new tile(0);
  uint32_t a = i % tile_sz, b = j % tile_sz;
  *raw_tiles[{i / tile_sz, j / tile_sz}] |= 1ull << (tile_sz * a + b);
}

void prep_consts(int matr_sz) {
  assert((matr_sz & (matr_sz - 1)) == 0); // assume matr_sz is 2^k
  auto tile_sz = quadtree::TILE_SIZE;
  while ((1u << levels) != matr_sz / tile_sz)
    ++levels;
}

coord parent_coord(coord s) {
  return {s.row / 2, s.col / 2};
}

uint32_t parent_sub_id(coord s) {
  return ((s.row & 1u) << 1u) + (s.col & 1u);
}

void construct_tree() {
  std::unordered_map<coord, qt_node *> cur_lvl, next_lvl;
  for (auto p : raw_tiles) {
    auto pid = parent_coord(p.first);
    if (!cur_lvl.count(pid))
      cur_lvl.insert({pid, new qt_node()});
    qt_node &par_node = *cur_lvl[pid];
    par_node.sub[parent_sub_id(p.first)] = p.second;
  }
  int lvl_at = 1;

  while (lvl_at != levels) {
    for (auto p : cur_lvl) {
      auto pid = parent_coord(p.first);
      if (!next_lvl.count(pid))
        next_lvl.insert({pid, new qt_node()});
      qt_node &par_node = *next_lvl[pid];
      par_node.sub[parent_sub_id(p.first)] = p.second;
    }
    int d = next_lvl.size();
    cur_lvl = next_lvl;
    next_lvl.clear();
    ++lvl_at;
  }

  assert(cur_lvl.size() == 1);
  root = cur_lvl[{0, 0}];
}

void cleanup() {
  raw_tiles.clear();
  euler_tour_ids.clear();
  tile_ids.clear();
  v_to_depth.clear();

  levels = euler_time = leaf_time = 0;

  delete root;
  root = nullptr;
}

quadtree finalize() {
  quadtree ans;
  auto &ts = ans.tree_structure_data;
  auto &tls = ans.tiles;
  ts.resize(euler_tour_ids.size());
  tls.resize(tile_ids.size());
  for (auto p : euler_tour_ids) {
    auto node = ((qt_node *) p.first);
    for (int r = 0; r < 4; r++)
      ts[p.second][r] = node->sub[r] ? v_to_depth[p.first] != levels ? euler_tour_ids[node->sub[r]] + 1 : (
        (uint32_t) tile_ids[((qt_node *) p.first)->sub[r]] | quadtree::TILE_REFERENCE) : 0;
  }
  for (auto p : tile_ids)
    tls[p.second] = *(tile *) p.first;
  cleanup();
  return ans;
}

/*
 * Constructs a quadtree representation of a sparse matrix
 * in O(m) time where m is the number of non-zero elements in the matrix.
 */
quadtree converter::build_quadtree_from_csr(const std::vector<int> &col_index, const std::vector<int> &row_index) {
  size_t matr_sz = row_index.size();
  prep_consts(matr_sz);

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

  return finalize();
}

quadtree converter::build_quadtree_from_coo(const std::vector<std::pair<int, int>> &els, int matr_sz) {
  prep_consts(matr_sz);
  for (auto p : els)
    add_point(p.row, p.col);
  construct_tree();
  dfs(root);
  return finalize();
}
