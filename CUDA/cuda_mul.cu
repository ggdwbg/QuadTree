#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include "quadtree.h"

using namespace std;

using u32 = uint32_t;
#define cudaPrintError printf("cuda error on line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));

void cuda_test() {
  cudaPrintError
}

u32 *transfer_quadtree_to_gpu(const quadtree &qt) { // sz <-> length of euler your, actual size in bytes is sz * 4 * 4
  size_t sz_alloc = qt.tree_structure_data.size() * 4 * 4;
  printf("size of stuff to be allocated %d\n", (int) sz_alloc);
  cudaPrintError
  u32 *ret;
  cudaMalloc(&ret, sz_alloc);
  cudaPrintError
  cudaMemcpy(ret, qt.tree_structure_data.data(), sz_alloc, cudaMemcpyHostToDevice);
  cudaPrintError
  cudaDeviceSynchronize();
  cudaPrintError
  return ret;
}
// C = A + B
__global__
void add_quadtree(
  u32 *left, u32 u, u32 ussz, // base for left operand, vertex in left operand, subtree size of u-th vertex
  u32 *right, u32 v, u32 vssz, // the same but for right operand
  u32 level, // log2 of size of current block
  u32 *volatile *ret_val, // pointer to where to write the result, allocation is responsibility of this procedure
  volatile u32 *ret_size // answer size in vertices (actual size in bytes is 16 times this value)
) {
  /*
   * TODO: cutoff on the number of threads running simultaneously.
   * Say, there are 2020 threads currently running, then we're not creating new threads
   * but instead proceed with requested addition using this thread only.
   */
  if (level == 3) {
    // If we're in a leaf, just OR what is needed to be ORed and quit.
    *ret_size = 1;
    u32 *ans = new u32[4];
    for (int i = 0; i < 4; i++)
      ans[i] = left[4 * u + i] | right[4 * v + i];
    *ret_val = ans;
    return;
  }
  // Otherwise, we need to recursively add corresponding subtrees
  volatile u32 *sizes = new u32[4]; // sizes of A_k + B_k: MARK SIZES AS VOLATILE
  int *add_to_merged_tours = new int[4];
  volatile int a;

  u32 *volatile *res_children = new u32 *volatile[4]; // euler tours of (A_k + B_k)
  int rnv[4] = {0, 0, 0, 0};
  for (u32 i = 0; i < 4; i++) {
    res_children[i] = 0, sizes[i] = 0;
    // 2: [3, 5, 0, 0] subtree size = 8
    // 0: [1, 3, 0, 0] subtree size = 8
    //     1
    // Euler tour storage format turned out to be extremely convenient:
    // to calculate the size of the subtree of vertex u, we only need to subtract
    // left[u][i] from the left[u][j] with j > i, left[u][j] != 0, j -> min
    // (or if there is none, then that entry is just the size of u)
    // TODO: rewrite this messy code, possibly optimize it in the process.
    int u_next_nnz = i + 1;
    while ((u_next_nnz < 4) && !left[4 * u + u_next_nnz])
      ++u_next_nnz;
    int v_next_nnz = i + 1;
    while ((v_next_nnz < 4) && !right[4 * v + v_next_nnz])
      ++v_next_nnz;
    u32 cur_u_subtree_size, cur_v_subtree_size;
    if (v_next_nnz != 4)
      cur_v_subtree_size = right[4 * v + v_next_nnz] - right[4 * v + i];
    else
      cur_v_subtree_size = vssz - (right[4 * v + i] - v);
    if (u_next_nnz != 4)
      cur_u_subtree_size = left[4 * u + u_next_nnz] - left[4 * u + i];
    else
      cur_u_subtree_size = ussz - (left[4 * u + i] - u);
    if (left[4 * u + i] != 0 && right[4 * v + i] != 0) {
      assert(sizes[i] == 0);
      // If both subtrees are non-zero, then issue another thread recursively.
      add_quadtree << < 1, 1 >> > (left, left[4u * u + i], cur_u_subtree_size,
        right, right[4u * v + i], cur_v_subtree_size,
        level - 1,
        &res_children[i],
        &sizes[i]);
      rnv[i] = 1;
      assert(sizes[i] == 0);
      assert(rnv[i]);
      //cudaDeviceSynchronize();
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
          add_to_merged_tours[i] = -(int) left[4 * u + i];
          res_children[i] = new u32[4 * cur_u_subtree_size];
          memcpy(res_children[i], left + 4 * (left[4 * u + i]), 16 * cur_u_subtree_size);
          //for (u32 j = 0; j < 4*cur_u_subtree_size; j++)
          //  res_children[i][j] = left[4*left[4 * u + i] + j];

        } else {
          assert(right[4 * v + i] && !left[4 * u + i]);
          sizes[i] = cur_v_subtree_size;
          add_to_merged_tours[i] = -(int) right[4 * v + i];
          res_children[i] = new u32[4 * cur_v_subtree_size];
          //for (u32 j = 0; j < 4*cur_v_subtree_size; j++)
          // res_children[i][j] = right[4*right[4 * v + i] + j];
          memcpy(res_children[i], right + 4 * (right[4 * v + i]), 16 * cur_v_subtree_size);
        }
      }
    }
  }
  // Wait for all computations to finish before merging the euler tours.
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("%s <-> %d\n", cudaGetErrorString(err), cudaLimitDevRuntimeSyncDepth);
    assert(false);
  }
  // Merge euler tours:
  int sz_prefix_sums[4] = {1, 0, 0, 0};
  for (int i = 1; i < 4; i++)
    sz_prefix_sums[i] = sz_prefix_sums[i - 1] + sizes[i - 1];
  *ret_size = sz_prefix_sums[3] + sizes[3];
  *ret_val = new u32[4 * (*ret_size)]; // TODO optimize here

  for (u32 i = 0; i < 4; i++) {
    if (!sizes[i])
      (*ret_val)[i] = 0;
    else
      (*ret_val)[i] = sz_prefix_sums[i];
    add_to_merged_tours[i] += (int) sz_prefix_sums[i];
  }
  for (u32 i = 0; i < 4; i++) {
    for (u32 j = 0; j < sizes[i]; j++) {
      for (u32 k = 0; k < 4; k++) {
        u32 index = 4u * (sz_prefix_sums[i] + j) + k;
        (*ret_val)[index] = (int) res_children[i][4 * j + k] +
                            (!res_children[i][4 * j + k]
                             || (res_children[i][4 * j + k] & quadtree::LEAF_MASK) ? 0 : (int) add_to_merged_tours[i]);
      }
    }
  }
  //cudaDeviceSynchronize();
  printf("GPU added left[%u] with right[%u] tour size: %u: elements: ",
         u, v);
  for (int i = 0; i < (int) (4 * (*ret_size)); i++)
    printf("%u ", (*ret_val)[i]);
  printf("\n");
}

__global__ void copy_stuff(u32 *runtime_heap, u32 *cudamalloced_heap, u32 size) {
  memcpy(cudamalloced_heap, runtime_heap, size);
}


vector<array<u32, 4>> test_addition(u32 *l, u32 *r, u32 k, u32 szl, u32 szr) {
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 24);
  cudaPrintError
  cudaDeviceSynchronize();
  cudaPrintError
  // ret_gpu is a pointer in GPU memory to 8 bytes in GPU memory that contain the address in GPU memory of first element of the result
  u32 **ret_gpu, *ret_size;
  cudaMalloc(&ret_gpu, 8);
  cudaMalloc(&ret_size, 8);
  cudaPrintError
  add_quadtree << < 1, 1 >> > (l, 0u, szl, r, 0u, szr, k, (u32 **) ret_gpu, ret_size);
  cudaPrintError
  u32 *alloc_begin;
  cudaDeviceSynchronize();
  cudaPrintError
  // alloc_begin = *ret
  cudaMemcpy(&alloc_begin, ret_gpu, 8, cudaMemcpyDeviceToHost);
  cudaPrintError
  cudaDeviceSynchronize();

  u32 cpu_size;
  cudaMemcpy(&cpu_size, ret_size, 4, cudaMemcpyDeviceToHost);
  cudaPrintError
  cudaDeviceSynchronize();
  printf("CPU SIZE: %d\n", cpu_size);
  vector<array<u32, 4>> ans(cpu_size);
  u32 *cudamalloced;
  cudaMalloc(&cudamalloced, cpu_size * 16);
  cudaPrintError
  cudaDeviceSynchronize();
  copy_stuff << < 1, 1 >> > (alloc_begin, cudamalloced, cpu_size * 16);
  cudaPrintError
  cudaDeviceSynchronize();
  cudaMemcpy(ans.data(), cudamalloced, cpu_size * 16, cudaMemcpyDeviceToHost);
  cudaPrintError
  cudaDeviceSynchronize();
  return ans;
}

__global__
void cuda_multiply_quadtree
  (u32 *base_left, int vertex_left,
   u32 *base_right, int vertex_right,
   int level, u32 **ret_val) {

}