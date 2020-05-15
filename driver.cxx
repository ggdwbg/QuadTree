#include <iostream>
#include "CUDA/quadtree.h"
#include "converter.h"
#include <random>
#include <chrono>
#include <bitset>
#include <fstream>
#include <algorithm>
#include <set>
#include <assert.h>
#include <tuple>
#include <cstring>

using namespace std;

std::mt19937 gen;

template<class Callback>
void measure(const std::string &activity, Callback f) {
  auto __start = std::chrono::system_clock::now();
  f();
  auto __end = std::chrono::system_clock::now();
  int ems = std::chrono::duration_cast
    <std::chrono::milliseconds>(__end - __start).count();
  std::cout << "Activity \"" << activity << "\" took " << ems << " ms." << std::endl;
}

using u32 = uint32_t;

void CPU_add_quadtree(
  u32 *left, u32 u, u32 ussz, // base for left operand, vertex in left operand, subtree size of u-th vertex
  u32 *right, u32 v, u32 vssz, // the same but for right operand
  u32 level, // log2 of size of current block
  u32 **ret_val, // pointer to where to write the result, allocation is responsibility of this procedure
  u32 *ret_size // answer size in vertices (actual size in bytes is 16 times this value)
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
    u32 *ans = new u32[4];
    for (int i = 0; i < 4; i++)
      ans[i] = left[4 * u + i] | right[4 * v + i];
    *ret_val = ans;
    return;
  }
  // Otherwise, we need to recursively add corresponding subtrees
  u32 *sizes = new u32[4]; // sizes of A_k + B_k
  int *add_to_merged_tours = new int[4];
  u32 **res_children = new u32 *[4]; // euler tours of (A_k + B_k)
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

    //assert((v_next_nnz != 4 ? right[4 * v + v_next_nnz] : vssz) > right[4 * v + i]);
    //assert((u_next_nnz != 4 ? left[4 * u + u_next_nnz] : ussz) > left[4 * u + i]);
    if (left[4 * u + i] != 0 && right[4 * v + i] != 0) {
      // If both subtrees are non-zero, then issue another thread recursively.
      CPU_add_quadtree(left, left[4u * u + i], cur_u_subtree_size,
                       right, right[4u * v + i], cur_v_subtree_size,
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
          add_to_merged_tours[i] = -(int) left[4 * u + i];
          res_children[i] = new u32[4 * cur_u_subtree_size];
          memcpy(res_children[i], left + 4 * (left[4 * u + i]), 16 * cur_u_subtree_size);
        } else {
          assert(right[4 * v + i] && !left[4 * u + i]);
          sizes[i] = cur_v_subtree_size;
          add_to_merged_tours[i] = -(int) right[4 * v + i];
          res_children[i] = new u32[4 * cur_v_subtree_size];
          memcpy(res_children[i], right + 4 * (right[4 * v + i]), 16 * cur_v_subtree_size);
        }
      }
    }
  }
  // Wait for all computations to finish before merging the euler tours.
  // cudaDeviceSynchronize();
  // Merge euler tours:
  u32 sz_prefix_sums[4] = {1, 0, 0, 0};
  for (int i = 1; i < 4; i++)
    sz_prefix_sums[i] = sz_prefix_sums[i - 1] + sizes[i - 1];
  *ret_size = sz_prefix_sums[3] + sizes[3];
  *ret_val = new u32[4 * (*ret_size)];
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
  printf("CPU added left[%u] with right[%u] tour size: %u: elements: ",
         u, v);
  for (int i = 0; i < (int) (4 * (*ret_size)); i++)
    printf("%u ", (*ret_val)[i]);
  putchar('\n');
}

std::vector<std::pair<int, uint32_t *>> gpu_quadtrees;

u32 *transfer_quadtree_to_gpu(const quadtree &qt);

#define dbg std::cout << __LINE__ << " " << (GLOBAL_DEBUG++) << std::endl;

std::pair<size_t, std::vector<std::array<u32, 4>>>
build_and_register(int k, std::vector<std::pair<int, int>> &coo) {
  auto g = converter::build_quadtree_from_coo(coo, k);
  gpu_quadtrees.
    emplace_back(k,
                 transfer_quadtree_to_gpu(g));
  return {gpu_quadtrees.size() - 1, g.tree_structure_data};
}

extern void cuda_test();

std::vector<std::array<u32, 4>> test_addition(u32 *l, u32 *r, u32 k, u32 szl, u32 szr);

int main(int argc, char **argv) {
#ifdef TEST
  std::cout << "works" << std::endl;
#endif
  int klog = 7;
  int k = 1u << klog;
  cuda_test();
  std::vector<std::pair<int, int>> coo1, coo2;
  for (int i = 0; i < k; i++)
    coo1.emplace_back(i, i);
  for (int i = 0; i < k; i++)
    for (int j = 0; j < 5; j++) {
      coo2.emplace_back(i, gen() % k);
      coo2.emplace_back(i, k - 1 - i);
    }
  coo2.emplace_back(42, 43);
  std::sort(coo2.begin(), coo2.end());
  coo2.erase(std::unique(coo2.begin(), coo2.end()), coo2.end());
  std::vector<std::array<u32, 4>> want1, want2;
  size_t sz1 = 0, sz2 = 0;
  cuda_test();
  measure("building quadtrees", [&]() {
    want1 = build_and_register(klog, coo1).second;
    sz1 = want1.size();
    want2 = build_and_register(klog, coo2).second;
    sz2 = want2.size();
  });
  auto print_euler_tour = [&](std::vector<std::array<u32, 4>> &want, const char *str = "") {
    std::cout << str << '\n';
    for (int i = 0; i < want.size(); i++) {
      std::cout << i << ": [";
      for (int j = 0; j < 4; j++) {
        if (j) std::cout << ", ";
        if (want[i][j] & quadtree::LEAF_MASK)
          std::cout << std::bitset<16>(want[i][j] & ~quadtree::LEAF_MASK);
        else
          std::cout << want[i][j];
      }
      std::cout << "]\n";
    }
  };
  u32 ret_size;
  u32 *ret_val;
  CPU_add_quadtree((u32 *) (want1.data()), 0, want1.size(),
                   (u32 *) want2.data(), 0, want2.size(),
                   klog, &ret_val, &ret_size);
  vector<array<u32, 4>> cpu_quadtree_ret(ret_size);
  for (int i = 0; i < (int) ret_size; i++)
    for (int j = 0; j < 4; j++)
      cpu_quadtree_ret[i][j] = *(ret_val + 4 * i + j);
  auto cpu_points = converter::get_nnz_from_quadtree(quadtree(klog, cpu_quadtree_ret));
  sort(cpu_points.begin(), cpu_points.end());

  cuda_test();
  vector<array<u32, 4>> gpu_ret = test_addition(gpu_quadtrees[0].second, gpu_quadtrees[1].second, klog, want1.size(),
                                                want2.size());
  auto gpu_points = converter::get_nnz_from_quadtree(quadtree(klog, gpu_ret));
  sort(gpu_points.begin(), gpu_points.end());
  cout << "GPU points: ";
  for (auto r : gpu_points)
    cout << "(" << r.first << ", " << r.second << ") ";
  cout << endl;
  cout << "CPU points: ";
  for (auto r : cpu_points)
    cout << "(" << r.first << ", " << r.second << ") ";
  cout << endl;
  return 0;
}