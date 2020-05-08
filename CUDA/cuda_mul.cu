#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include "quadtree.h"

using namespace std;

using u32 = uint32_t;

u32 *transfer_quadtree_to_gpu(const quadtree &qt) { // sz <-> length of euler your, actual size in bytes is sz * 4 * 4
  size_t sz_alloc = qt.tree_structure_data.size() * 4 * 4;
  u32 *ret;
  cudaMalloc(&ret, sz_alloc);
  cudaMemcpy(ret, qt.tree_structure_data.data(), sz_alloc, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  return ret;
}

// C = A + B
__global__
void add_quadtree(
  u32 *left, int u, int ssz_u,
  u32 *right, int v, int ssz_v,
  int level,
  u32 **ret_val, int *ret_size) {
  if (level == 3) {
    // TODO: what if left[u] = right[u] = NULL?
    // not null by induction hypothesis.
    *ret_size = 1;
    u32 *ans = new u32[4]; // this allocates on gpu hopefully
    for (int i = 0; i < 4; i++)
      ans[i] = left[4 * u + i] | right[4 * v + i];
    *ret_val = ans;
    return;
  }
  int *sizes = new int[4];
  u32 **res_children = new u32 *[4];
  for (int i = 0; i < 4; i++) {
    u32 ssz_li, ssz_ri;
    // [2, 0, 0, 3]
    //        i
    if (i != 3)
      ssz_li = left[4 * u + i + 1] - left[4 * u + i], ssz_ri = right[4 * v + i + 1] - right[4 * v + i];
    else
      ssz_li = ssz_u - left[4 * u + i], ssz_ri = ssz_v - right[4 * v + i];
    res_children[i] = 0, sizes[i] = 0;
    if (left[4 * u + i] != 0 && right[4 * v + i] != 0) {
      add_quadtree << < 1, 1 >> >
                           (left, 4 * u + i, ssz_li,
                             right, 4 * v + i, ssz_ri,
                             level - 1,
                             &res_children[i], &sizes[i]);
    } else {
      if (left[4 * u + i] == 0 && right[4 * v + i] == 0) {
        sizes[i] = 0, res_children[i] = 0;
      } else {
        int non_zero = -1;
        if (left[4 * u + i]) {
          //TODO
        }
      }
    }
  }
  cudaDeviceSynchronize();

}

__global__
void cuda_multiply_quadtree
  (u32 *base_left, int vertex_left,
   u32 *base_right, int vertex_right,
   int level, u32 **ret_val) {

}