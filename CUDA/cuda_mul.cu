#include <bits/stdc++.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include "quadtree.h"

using namespace std;

using u32 = uint32_t;
#define cudaPrintError printf("cuda error on line %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError()));
#define cudaAssertWorks(expr) { expr; auto r = cudaGetLastError(); if (r != cudaSuccess) { printf("failed on line %d: %s\n", __LINE__, cudaGetErrorString(r)); assert(false); }};

void cuda_test() {
  // always call this function!
#define print_limit(x) { size_t val; cudaDeviceGetLimit(&val, x); printf(#x ": %llu\n", (unsigned long long)val); }
  print_limit(cudaLimitMallocHeapSize)
  print_limit(cudaLimitDevRuntimePendingLaunchCount)
  print_limit(cudaLimitStackSize)
  cudaAssertWorks(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 3072ull * 1024ull * 1024ull));
}

u32 *transfer_quadtree_to_gpu(const quadtree &qt) { // sz <-> length of euler your, actual size in bytes is sz * 4 * 4
  size_t sz_alloc = qt.tree_structure_data.size() * 4 * 4;
  printf("size of stuff to be allocated %d\n", (int) sz_alloc);
  u32 *ret;
  cudaAssertWorks(cudaMalloc(&ret, sz_alloc));
  cudaAssertWorks(cudaMemcpy(ret, qt.tree_structure_data.data(), sz_alloc, cudaMemcpyHostToDevice));
  cudaDeviceSynchronize();
  cudaAssertWorks();
  return ret;
}

__global__ void shift_euler_tours(u32 *base, u32 size, int plus, u32 *copy_from) {
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= size) return;
  base[index] = copy_from[index];
  if (!base[index] || base[index] & quadtree::LEAF_MASK) return;
  base[index] = int(base[index]) + plus;
}

#define DP_THRESHOLD 1024

// C = A + B
__global__
void add_quadtree(
  u32 *left, u32 u, u32 ussz, // base for left operand, vertex in left operand, subtree size of u-th vertex
  u32 *right, u32 v, u32 vssz, // the same but for right operand
  u32 level, // log2 of size of current block
  u32 **ret_val, // pointer to where to write the result, allocation is responsibility of this procedure
  u32 *ret_size // answer size in vertices (actual size in bytes is 16 times this value)
) {
  // TODO: (level <= K) => (do it in this thread only without recursive launches)
  if (level == 3 || (ussz <= DP_THRESHOLD && vssz <= DP_THRESHOLD)) {
    // If we're in a leaf, just OR what is needed to be ORed and quit.
    *ret_size = 1;
    u32 *ans = new u32[4];
    for (int i = 0; i < 4; i++)
      ans[i] = left[4 * u + i] | right[4 * v + i];
    *ret_val = ans;
    return;
  }
  // Otherwise, we need to recursively add corresponding subtrees
  u32 *sizes = new u32[4]; // sizes of A_k + B_k
  int add_to_merged_tours[4];

  u32 **res_children = new u32 *[4]; // euler tours of (A_k + B_k)

  // Euler tour storage format turned out to be extremely convenient:
  // to calculate the size of the subtree of vertex u, we only need to subtract
  // left[u][i] from the left[u][j] with j > i, left[u][j] != 0, j -> min
  // (or if there is none, then that entry is just the size of u)
  u32 sub_u_sizes[4], sub_v_sizes[4];
  u32 last_nnz_u = ussz + u, last_nnz_v = vssz + v;
  for (int i = 3; i >= 0; i--) {
    sub_u_sizes[i] = last_nnz_u - left[4 * u + i];
    sub_v_sizes[i] = last_nnz_v - right[4 * v + i];
    if (left[4 * u + i])
      last_nnz_u = left[4 * u + i];
    if (right[4 * v + i])
      last_nnz_v = right[4 * v + i];
  }
  for (u32 i = 0; i < 4; i++) {
    u32 cur_u_subtree_size = sub_u_sizes[i], cur_v_subtree_size = sub_v_sizes[i];
    if (left[4 * u + i] != 0 && right[4 * v + i] != 0) {
      // If both subtrees are non-zero, then issue another thread recursively.
      add_quadtree << < 1, 1 >> > (left, left[4u * u + i], cur_u_subtree_size,
        right, right[4u * v + i], cur_v_subtree_size,
        level - 1,
        &res_children[i],
        &sizes[i]);
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
        } else {
          //assert(right[4 * v + i] && !left[4 * u + i]);
          sizes[i] = cur_v_subtree_size;
          add_to_merged_tours[i] = -(int) right[4 * v + i];
          res_children[i] = new u32[4 * cur_v_subtree_size];
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
  auto wtr = new u32[4 * (*ret_size)];

  for (u32 i = 0; i < 4; i++) {
    if (!sizes[i])
      wtr[i] = 0;
    else
      wtr[i] = sz_prefix_sums[i];
    add_to_merged_tours[i] += (int) sz_prefix_sums[i];
  }
#define BLOCK_SIZE 128
  // __global__ void shift_euler_tours(u32* base, u32 size, int plus)
  for (int i = 0; i < 4; i++) {
    if (!sizes[i])
      continue;
    // TODO: if size is small, do it in this thread maybe?
    shift_euler_tours << < (4u * sizes[i] + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >> >
                                                                          (wtr + 4u * sz_prefix_sums[i], 4u *
                                                                                                         sizes[i], add_to_merged_tours[i], res_children[i]);
  }
  //cudaAssertWorks(cudaDeviceSynchronize());

  *ret_val = wtr; // TODO optimize here
  /*printf("GPU added left[%u] with right[%u] tour size: %u: elements: ",
         u, v);
  for (int i = 0; i < (int) (4 * (*ret_size)); i++)
    printf("%u ", (*ret_val)[i]);
  printf("\n");*/
  for (int i = 0; i < 4; i++)
    delete[] res_children[i];
  delete[] sizes;
  delete[] res_children;
}

__global__ void copy_stuff(u32 *runtime_heap, u32 *cudamalloced_heap, u32 size) {
  memcpy(cudamalloced_heap, runtime_heap, size);
}

vector<array<u32, 4>> test_addition(u32 *l, u32 *r, u32 k, u32 szl, u32 szr) {
  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 17);

  cudaDeviceSynchronize();
  cudaPrintError
  // ret_gpu is a pointer in GPU memory to 8 bytes in GPU memory that contain the address in GPU memory of first element of the result
  u32 **ret_gpu, *ret_size;
  cudaMalloc(&ret_gpu, 8);
  cudaMalloc(&ret_size, 8);
  cudaDeviceSynchronize();
  cudaPrintError
  //cudaPrintError
  add_quadtree << < 1, 1 >> > (l, 0u, szl, r, 0u, szr, k, (u32 **) ret_gpu, ret_size);
  //cudaPrintError
  u32 *alloc_begin;
  cudaDeviceSynchronize();
  cudaPrintError
  // alloc_begin = *ret
  cudaMemcpy(&alloc_begin, ret_gpu, 8, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  cudaPrintError
  u32 cpu_size;
  cudaMemcpy(&cpu_size, ret_size, 4, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaPrintError
  printf("gpu output size: %d\n", cpu_size);
  vector<array<u32, 4>> ans(cpu_size);
  u32 *cudamalloced;
  cudaMalloc(&cudamalloced, cpu_size * 16);
  copy_stuff << < 1, 1 >> > (alloc_begin, cudamalloced, cpu_size * 16);
  cudaPrintError
  cudaDeviceSynchronize();
  cudaMemcpy(ans.data(), cudamalloced, cpu_size * 16, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaPrintError
  return ans;
}

__global__
void cuda_multiply_quadtree
  (u32 *base_left, int vertex_left,
   u32 *base_right, int vertex_right,
   int level, u32 **ret_val) {

}