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
  u32 *left, int u, u32 ussz, // base for left operand, vertex in left operand, subtree size of u-th vertex
  u32 *right, int v, u32 vssz, // the same but for right operand
  int level, // log2 of size of current block
  u32 **ret_val, // pointer to where to write the result, allocation is responsibility of this procedure
  int *ret_size // answer size in vertices (actual size in bytes is 16 times this value)
) {
  // TODO: what if everything is zero?
  /*
   * TODO: cutoff on the number of threads running simultaneously.
   * Say, there are 2020 threads currently running, then we're not creating new threads
   * but instead proceed with requested addition using this thread only.
   */
  if (level == 3) {
    // If we're in a leaf, just OR what is needed to be ORed and quit.
    *ret_size = 1;
    u32 *ans = new u32[4]; // this allocates on gpu hopefully
    for (int i = 0; i < 4; i++)
      ans[i] = left[4 * u + i] | right[4 * v + i];
    *ret_val = ans;
    return;
  }
  // Otherwise, we need to recursively add corresponding subtrees
  int *sizes = new int[4]; // sizes of A_k + B_k
  int *add_to_merged_tours = new int[4]; // self-explanatory
  u32 **res_children = new u32 *[4]; // euler tours of (A_k + B_k)
  for (int i = 0; i < 4; i++) {
    res_children[i] = 0, sizes[i] = 0;
    // Euler tour storage format turned out to be extremely convenient:
    // to calculate the size of the subtree of vertex u, we only need to subtract
    // left[u][i] from the left[u][j] with j > i, left[u][j] != 0, j -> min
    // (or if there is none, then that entry is just the size of u)
    // TODO: rewrite this messy code, possibly optimize it in the process.
    int u_next_nnz = i + 1;
    while (u_next_nnz < 4 && !left[4 * u + u_next_nnz])
      ++u_next_nnz;
    u32 cur_u_subtree_size = (u_next_nnz != 4 ? left[4 * u + u_next_nnz] : ussz) - left[4 * u + i];
    int v_next_nnz = i + 1;
    while (v_next_nnz < 4 && !right[4 * v + v_next_nnz])
      ++v_next_nnz;
    u32 cur_v_subtree_size = (v_next_nnz != 4 ? right[4 * v + v_next_nnz] : vssz) - right[4 * v + i];

    if (left[4 * u + i] != 0 && right[4 * v + i] != 0) {
      // If both subtrees are non-zero, then issue another thread recursively.
      add_quadtree << < 1, 1 >> > (left, 4 * u + i, cur_u_subtree_size,
        right, 4 * v + i, cur_v_subtree_size,
                             level - 1,
                             &res_children[i], &sizes[i]);
      // Euler tour is correct by induction hypothesis, so we need not add anything extra here.
      add_to_merged_tours[i] = 0;
    } else {
      if (left[4 * u + i] == 0 && right[4 * v + i] == 0) {
        // If both are zero, do nothing.
        sizes[i] = 0, res_children[i] = 0;
        add_to_merged_tours[i] = 0;
      } else {
        // The most interesting case: if one is zero and the other one is not.
        if (left[4 * u + i]) {
          sizes[i] = cur_u_subtree_size;
          add_to_merged_tours[i] = -left[4 * u + i];
          res_children[i] = new u32[4 * cur_u_subtree_size];
          memcpy(res_children[i], left + (left[4 * u + i]), 4 * cur_u_subtree_size);
        } else {
          sizes[i] = cur_v_subtree_size;
          add_to_merged_tours[i] = -right[4 * v + i];
          res_children[i] = new u32[4 * cur_v_subtree_size];
          memcpy(res_children[i], right + (right[4 * v + i]), 4 * cur_v_subtree_size);
        }
      }
    }
  }
  // Wait for all computations to finish before merging the euler tours.
  cudaDeviceSynchronize();
  // Merge euler tours:
  int sz_prefix_sums[4] = {1, 0, 0, 0};
  for (int i = 1; i < 4; i++)
    sz_prefix_sums[i] = sz_prefix_sums[i - 1] + sizes[i - 1];
  *ret_size = sz_prefix_sums[3] + sizes[3];
  *ret_val = new u32[4 * (*ret_size)];
  for (int i = 0; i < 4; i++) {
    (*ret_val)[i] = sz_prefix_sums[i];
    add_to_merged_tours[i] += sz_prefix_sums[i];
  }
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < sizes[i]; j++) {
      for (int k = 0; k < 4; k++) {
        int index = 4 * (sz_prefix_sums[i] + j) + k;
        if (!res_children[i][4 * j + k] || (res_children[i][4 * j + k] & quadtree::LEAF_MASK)) continue;
        (*ret_val)[index] = res_children[i][4 * j + k] + add_to_merged_tours[i];
      }
    }
  }
}

__global__
void cuda_multiply_quadtree
  (u32 *base_left, int vertex_left,
   u32 *base_right, int vertex_right,
   int level, u32 **ret_val) {

}