#include <host_defines.h>
#include <utility>
#include <cuda_runtime_api.h>
#include "quadtree.h"

typedef unsigned char byte;
typedef int *device_pint;
typedef short *device_pshort;
typedef std::pair<device_pint, device_pshort> device_qtree;
/*
 * Multiplies two 4x4 tiles.
 */
__device__
int mul_tiles(int a, int b) {
  byte colb[4];
  for (int i = 0; i < 4; i++)
    colb[i] = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (b & (1 << (4 * i + j)))
        colb[j] |= 1 << i;
  int ans = 0;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 4; j++)
      if (((a >> (4 * i)) & 0xff) & colb[j])
        ans |= 1 << (4 * i + j);
  return ans;
}

// transfer cpu to gpu
device_qtree transfer_ctg(quadtree &cpu_qtree) {
  device_pint structure;
  device_pshort tiles;
  size_t structure_sz = cpu_qtree.tree_structure_data.size() * sizeof(int);
  size_t tiles_sz = cpu_qtree.tiles.size() * sizeof(short);
  cudaMalloc(reinterpret_cast<void **>(&structure), structure_sz);
  cudaMalloc(reinterpret_cast<void **>(&tiles), tiles_sz);
  cudaMemcpy(structure, cpu_qtree.tree_structure_data.data(), structure_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(tiles, cpu_qtree.tiles.data(), tiles_sz, cudaMemcpyHostToDevice);
  return std::make_pair(structure, tiles);
}