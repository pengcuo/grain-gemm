/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions
 *    and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific prior written
 *    permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************************/

// Independent Thor 1SM CuTe implementation. CuTe UMMA/TMA primitives and TMEM allocation are adapted
// from CUTLASS tutorials 01_mma_sm100.cu, 02_mma_tma_sm100.cu, and 05_mma_tma_epi_sm100.cu at
// 098de2a652cf8f00fd70b2df54051c7eccbb855a. No CUTLASS collective or GEMM adapter is instantiated.
// The producer warpgroup feeds separate operand, accumulator, and scale rings; eight consumer
// warps promote exact INT32 G256 partials to FP32 while the following UMMA and TMA transfers run.
#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cstdint>

// Supplied by the existing warp-MMA implementation included by the Thor translation unit.
extern "C" int grain_warp_g256(void const*, void const*, void const*, void const*, void*,
                              int, int, int, int, void*);

namespace thor_ws1 {
using namespace cute;

template <class LA, class LB, int BM, int BN, int P> struct Storage {
  // This operand storage becomes the coalesced BF16 exchange after the rings are drained.
  alignas(128) int8_t a[P][cosize_v<LA>];
  alignas(128) int8_t b[P][cosize_v<LB>];
  alignas(16) uint64_t operand_full[P];
  alignas(16) uint64_t operand_empty[P];
  alignas(16) uint64_t accumulator_full[2];
  alignas(16) uint64_t accumulator_empty[2];
  alignas(16) uint64_t scale_full[P];
  alignas(16) uint64_t scale_empty[P];
  alignas(16) float scale_a[P][BM];
  alignas(16) float scale_b[P][BN];
  uint32_t tmem;
};

template <int BM, int BN, int P, class LA, class LB, class MMA>
__global__ __launch_bounds__(384, 1) void kernel(
    float const* SA, float const* SB, bfloat16_t* C, int M, int N, int K,
    LA la, LB lb, MMA mma, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
  static_assert(BM == 128 && (BN == 128 || BN == 256), "Supported Thor 1SM tiles");
  static_assert(P >= 2, "Operand ring requires at least two stages");
  extern __shared__ __align__(128) char raw[];
  auto& shared = *reinterpret_cast<Storage<LA, LB, BM, BN, P>*>(raw);
  // Order the CUDA-generated TMEM allocation completion barrier initialization.
  __syncthreads();
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int tiles_m = M / BM, tiles_n = N / BN;
  int first_m = (blockIdx.x / (8 * tiles_n)) * 8;
  int band_m = min(8, tiles_m - first_m);
  int tile_m = first_m + blockIdx.x % band_m;
  int tile_n = (blockIdx.x % (8 * tiles_n)) / band_m;
  int groups = K / 256;

  auto tiler = make_shape(Int<BM>{}, Int<BN>{}, _256{});
  auto gC = local_tile(make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{})),
                       tiler, make_coord(tile_m, tile_n, _), Step<_1, _1, X>{});
  auto cta = mma.get_slice(_0{});
  auto acc = cta.make_fragment_C(cta.partition_C(gC));
  TMEM::Allocator1Sm allocator;
  if (warp == 0)
    allocator.allocate(2 * BN, &shared.tmem);
  if (threadIdx.x == 0) {
    for (int s = 0; s < P; ++s) {
      cutlass::arch::ClusterBarrier::init(&shared.operand_full[s], 1);
      cutlass::arch::ClusterBarrier::init(&shared.operand_empty[s], 1);
      cutlass::arch::ClusterBarrier::init(&shared.scale_full[s], 32);
      cutlass::arch::ClusterBarrier::init(&shared.scale_empty[s], 256);
    }
    for (int s = 0; s < 2; ++s) {
      cutlass::arch::ClusterBarrier::init(&shared.accumulator_full[s], 1);
      cutlass::arch::ClusterBarrier::init(&shared.accumulator_empty[s], 256);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  acc.data() = shared.tmem;
  if (warp == 0)
    allocator.release_allocation_lock();

  // Two consumer warpgroups each cover all 128 rows and half the N columns. Splitting N reduces
  // the FP32 register fragment to BN/2 per thread while preserving a single final BF16 conversion.
  constexpr int Columns = BN / 2;
  float total[Columns];
  if (threadIdx.x >= 128) {
    CUTE_UNROLL
    for (int i = 0; i < Columns; ++i)
      total[i] = 0.0f;
  }

  if (warp == 2 && lane == 0) {
    for (int g = 0; g < groups; ++g) {
      int stage = g % P;
      if (g >= P)
        cutlass::arch::ClusterBarrier::wait(&shared.operand_empty[stage], ((g / P) - 1) & 1);
      cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared.operand_full[stage], (BM + BN) * 256);
      SM90_TMA_LOAD_2D::copy(&map_a, &shared.operand_full[stage], 0, shared.a[stage],
                            g * 256, tile_m * BM);
      SM90_TMA_LOAD_2D::copy(&map_a, &shared.operand_full[stage], 0,
                            shared.a[stage] + BM * 128, g * 256 + 128, tile_m * BM);
      SM90_TMA_LOAD_2D::copy(&map_b, &shared.operand_full[stage], 0, shared.b[stage],
                            g * 256, tile_n * BN);
      SM90_TMA_LOAD_2D::copy(&map_b, &shared.operand_full[stage], 0,
                            shared.b[stage] + BN * 128, g * 256 + 128, tile_n * BN);
    }
  } else if (warp == 3) {
    for (int g = 0; g < groups; ++g) {
      int stage = g % P;
      if (g >= P)
        cutlass::arch::ClusterBarrier::wait(&shared.scale_empty[stage], ((g / P) - 1) & 1);
      // Exact public SA[M,K/256] K-group-contiguous / SB[K/256,N] N-contiguous layouts.
      CUTE_UNROLL
      for (int row = lane; row < BM; row += 32)
        shared.scale_a[stage][row] = SA[(tile_m * BM + row) * groups + g];
      CUTE_UNROLL
      for (int col = lane; col < BN; col += 32)
        shared.scale_b[stage][col] = SB[g * N + tile_n * BN + col];
      // Every scale writer publishes its own stores; do not delegate them to one lane.
      cutlass::arch::ClusterBarrier::arrive(&shared.scale_full[stage]);
    }
  } else if (warp == 0) {
    for (int g = 0; g < groups; ++g) {
      int stage = g % P, buffer = g % 2;
      cutlass::arch::ClusterBarrier::wait(&shared.operand_full[stage], (g / P) & 1);
      if (g >= 2)
        cutlass::arch::ClusterBarrier::wait(&shared.accumulator_empty[buffer], ((g / 2) - 1) & 1);
      acc.data() = shared.tmem + buffer * BN;
      auto frag_a = cta.make_fragment_A(make_tensor(make_smem_ptr(shared.a[stage]), la));
      auto frag_b = cta.make_fragment_B(make_tensor(make_smem_ptr(shared.b[stage]), lb));
      mma.accumulate_ = UMMA::ScaleOut::Zero;
      CUTE_UNROLL
      for (int kk = 0; kk < size<2>(frag_a); ++kk) {
        gemm(mma, frag_a(_, _, kk), frag_b(_, _, kk), acc);
        mma.accumulate_ = UMMA::ScaleOut::One;
      }
      // Both commits refer to the preceding UMMA work: release its operands and publish its TMEM.
      cutlass::arch::umma_arrive(&shared.operand_empty[stage]);
      cutlass::arch::umma_arrive(&shared.accumulator_full[buffer]);
    }
  } else if (threadIdx.x >= 128) {
    int consumer = threadIdx.x - 128;
    int row = consumer % 128;
    int first_col = (consumer / 128) * Columns;
    for (int g = 0; g < groups; ++g) {
      int stage = g % P, buffer = g % 2;
      cutlass::arch::ClusterBarrier::wait(&shared.accumulator_full[buffer], (g / 2) & 1);
      cutlass::arch::ClusterBarrier::wait(&shared.scale_full[stage], (g / P) & 1);
      float scale_a = shared.scale_a[stage][row];
      uint32_t partial0[16], partial1[16], partial2[16];
      auto load = [&](int sub, auto& v) {
        uint32_t address = shared.tmem + buffer * BN + first_col + sub * 16 +
                           ((row / 32) * 32 << 16);
        SM100_TMEM_LOAD_32dp32b16x::copy(address, v[0], v[1], v[2], v[3], v[4],
                                      v[5], v[6], v[7], v[8], v[9], v[10],
                                      v[11], v[12], v[13], v[14], v[15]);
      };
      load(0, partial0);
      load(1, partial1);
      cutlass::arch::fence_view_async_tmem_load();
      CUTE_UNROLL
      for (int sub = 0; sub < Columns / 16; ++sub) {
        auto& current = sub % 3 == 0 ? partial0 : (sub % 3 == 1 ? partial1 : partial2);
        if (sub + 2 < Columns / 16) {
          auto& next = (sub + 2) % 3 == 0 ? partial0 :
                       ((sub + 2) % 3 == 1 ? partial1 : partial2);
          load(sub + 2, next);
        }
        float scale_b[16];
        uint32_t address_b = cast_smem_ptr_to_uint(&shared.scale_b[stage][first_col + sub * 16]);
        CUTE_UNROLL
        for (int q = 0; q < 4; ++q) {
          asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
              : "=f"(scale_b[q * 4]), "=f"(scale_b[q * 4 + 1]),
                "=f"(scale_b[q * 4 + 2]), "=f"(scale_b[q * 4 + 3])
              : "r"(address_b + q * 16) : "memory");
        }
        CUTE_UNROLL
        for (int j = 0; j < 16; ++j) {
          float scale = scale_a * scale_b[j];
          total[sub * 16 + j] = fmaf(float(int32_t(current[j])), scale, total[sub * 16 + j]);
        }
        // wait::ld drains all pending loads. Each pair of promoted fragments covers two future
        // fragments of load latency; the last wait also drains the ring before slot release.
        if (sub % 2 == 1 || sub + 1 == Columns / 16)
          cutlass::arch::fence_view_async_tmem_load();
      }
      // Every reader has drained tcgen05.ld before freeing the TMEM and scale slots.
      cutlass::arch::ClusterBarrier::arrive(&shared.accumulator_empty[buffer]);
      cutlass::arch::ClusterBarrier::arrive(&shared.scale_empty[stage]);
    }
  }
  __syncthreads();

  auto* exchange = reinterpret_cast<bfloat16_t*>(raw);
  if (threadIdx.x >= 128) {
    int consumer = threadIdx.x - 128;
    int row = consumer % 128;
    int first_col = (consumer / 128) * Columns;
    CUTE_UNROLL
    for (int col = 0; col < Columns; col += 2) {
      int offset = row * BN + first_col + col;
      int swizzled = offset ^ ((row & 7) * 8);
      reinterpret_cast<__nv_bfloat162*>(exchange)[swizzled / 2] =
          __floats2bfloat162_rn(total[col], total[col + 1]);
    }
  }
  __syncthreads();
  for (int v = threadIdx.x; v < BM * BN / 8; v += 384) {
    int offset = v * 8;
    int swizzled = offset ^ (((offset / BN) & 7) * 8);
    *reinterpret_cast<uint4*>(C + (tile_m * BM + offset / BN) * N + tile_n * BN + offset % BN) =
        *reinterpret_cast<uint4*>(exchange + swizzled);
  }
  __syncthreads();
  if (warp == 0)
    allocator.free(shared.tmem, 2 * BN);
}

template <int BM, int BN, int P>
int launch(void const* a, void const* b, void const* sa, void const* sb, void* c,
           int M, int N, int K, cudaStream_t stream) {
  if (!a || !b || !sa || !sb || !c || M <= 0 || N <= 0 || K <= 0 || K % 256 ||
      (reinterpret_cast<uintptr_t>(sa) & 3) ||
      (reinterpret_cast<uintptr_t>(sb) & 3))
    return int(cudaErrorInvalidValue);
  auto fallback = [&]() {
    return grain_warp_g256(a, b, sa, sb, c, M, N, K, 0, static_cast<void*>(stream));
  };
  if (M % BM || N % BN || (reinterpret_cast<uintptr_t>(a) & 15) ||
      (reinterpret_cast<uintptr_t>(b) & 15) || (reinterpret_cast<uintptr_t>(c) & 15))
    return fallback();
  auto mma = make_tiled_mma(SM100_MMA_S8_SS<int8_t, int8_t, int32_t, BM, BN,
                                         UMMA::Major::K, UMMA::Major::K>{});
  auto la = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                                   partition_shape_A(mma, make_shape(Int<BM>{}, _256{})));
  auto lb = UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                                   partition_shape_B(mma, make_shape(Int<BN>{}, _256{})));
  CUtensorMap map_a, map_b;
  uint64_t dims_a[2] = {uint64_t(K), uint64_t(M)}, dims_b[2] = {uint64_t(K), uint64_t(N)};
  uint64_t stride[1] = {uint64_t(K)};
  uint32_t box_a[2] = {128, BM}, box_b[2] = {128, BN}, element_stride[2] = {1, 1};
  auto encode = [&](CUtensorMap* map, void const* ptr, uint64_t* dims, uint32_t* box) {
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void*>(ptr),
                                  dims, stride, box, element_stride,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  if (encode(&map_a, a, dims_a, box_a) != CUDA_SUCCESS ||
      encode(&map_b, b, dims_b, box_b) != CUDA_SUCCESS)
    return fallback();
  auto fn = kernel<BM, BN, P, decltype(la), decltype(lb), decltype(mma)>;
  constexpr int bytes = sizeof(Storage<decltype(la), decltype(lb), BM, BN, P>);
  auto status = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status != cudaSuccess)
    return int(status);
  fn<<<(M / BM) * (N / BN), 384, bytes, stream>>>(static_cast<float const*>(sa),
      static_cast<float const*>(sb), static_cast<bfloat16_t*>(c), M, N, K, la, lb, mma, map_a, map_b);
  return int(cudaGetLastError());
}

} // namespace thor_ws1
