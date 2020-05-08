#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include "quadtree.h"

using namespace std;

using u32 = uint32_t;

// Registry
// #id_from_quadtree -> quadtree
// 0: [1, 0, 0, 2] [3, 4, 5, 0] [LEAF..., LEAF..,LE....]
// 1: [0, 1, 0, 0], [3, 0, 0, 0], [LEAF, ...]
// mul(0, 1, 1, 1, LEVEL)

u32 *transfer_quadtree_to_gpu(const quadtree &qt) { // sz <-> length of euler your, actual size in bytes is sz * 4 * 4
  size_t sz_alloc = qt.tree_structure_data.size() * 4 * 4;
  u32 *ret;
  cudaMalloc(&ret, sz_alloc);
  cudaMemcpy(ret, qt.tree_structure_data.data(), sz_alloc, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  return ret;
}

void cuda_multiply_quadtree
  (u32 *base_left, int vertex_left,
   u32 *base_right, int vertex_right,
   int level) {

}