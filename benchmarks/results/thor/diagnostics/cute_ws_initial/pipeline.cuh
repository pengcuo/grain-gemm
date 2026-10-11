/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
// CuTe primitive usage adapted from CUTLASS tutorials
// cute/tutorial/blackwell/{04_mma_tma_2sm_sm100,05_mma_tma_epi_sm100}.cu
// at 098de2a652cf8f00fd70b2df54051c7eccbb855a. The independent G256
// pipeline uses the same 48/216 register split and eight promotion warps
// as the Thor Cosmos pipeline at 8977529aa23f70f02bfc36559599f00d59792996.
#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/config.h>
#include <cutlass/cluster_launch.hpp>
#include <cstdint>

// Defined by the SM110 translation unit's existing warp-MMA implementation.
extern "C" int grain_warp_g256(void const *, void const *, void const *,
                                void const *, void *, int, int, int, int, void *);

namespace thor_ws2 {
using namespace cute;

// A pair of CTAs owns one 256x256 output tile. Every CTA holds its 128 rows
// of A and 128 columns of B. Its local TMEM contains 128x256 accumulators.
template <class LA, class LB, int P> struct Storage {
  static constexpr int EpilogueColumns = 32;
  alignas(128) int8_t a[P][cosize_v<LA>];
  alignas(128) int8_t b[P][cosize_v<LB>];
  alignas(16) uint64_t ab_full[P];
  alignas(16) uint64_t ab_empty[P];
  alignas(16) uint64_t sf_full[P];
  alignas(16) uint64_t sf_empty[P];
  alignas(16) uint64_t acc_full[2];
  alignas(16) uint64_t acc_empty[2];
  alignas(16) float scale_a[P][128];
  alignas(16) float scale_b[P][256];
  // Independent packed BF16 exchanges for the two promotion warpgroups.
  // Keeping these separate from A/B permits producers to prefetch the next
  // output tile while consumers finish the current epilogue.
  alignas(128) bfloat16_t exchange[2][128 * EpilogueColumns];
  uint32_t tmem;
};

CUTE_DEVICE void release_registers() {
#if defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110F_ENABLED)
  asm volatile("setmaxnreg.dec.sync.aligned.u32 48;" ::: "memory");
#endif
}

CUTE_DEVICE void acquire_registers() {
#if defined(CUTLASS_ARCH_MMA_SM110A_ENABLED) || defined(CUTLASS_ARCH_MMA_SM110F_ENABLED)
  asm volatile("setmaxnreg.inc.sync.aligned.u32 216;" ::: "memory");
#endif
}

CUTE_DEVICE void release_local(uint64_t *barrier) {
  cutlass::arch::ClusterBarrier::arrive(barrier);
}

CUTE_DEVICE void release_leader(uint64_t *barrier) {
  cutlass::arch::ClusterBarrier::arrive(barrier, 0, 1);
}

// Eight contiguous M tiles per stripe, with no phantom work in the last
// stripe. All four producer warps and both consumers derive the same tile.
CUTE_DEVICE void tile_coordinates(int tile, int mt, int nt, int &im, int &in) {
  int first = (tile / (8 * nt)) * 8;
  int band = min(8, mt - first);
  int local = tile - first * nt;
  im = first + local % band;
  in = local / band;
}

CUTE_DEVICE void load_partial(uint32_t address, uint32_t (&v)[32]) {
  SM100_TMEM_LOAD_32dp32b32x::copy(
      address, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7],
      v[8], v[9], v[10], v[11], v[12], v[13], v[14], v[15],
      v[16], v[17], v[18], v[19], v[20], v[21], v[22], v[23],
      v[24], v[25], v[26], v[27], v[28], v[29], v[30], v[31]);
}

CUTE_DEVICE void promote(uint32_t const (&partial)[32], float sa,
                         float const *sb, float *total) {
  CUTE_UNROLL
  for (int j = 0; j < 32; j += 4) {
    float b0, b1, b2, b3;
    uint32_t address = cast_smem_ptr_to_uint(sb + j);
    asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(b0), "=f"(b1), "=f"(b2), "=f"(b3)
                 : "r"(address));
    total[j + 0] = fmaf(float(int32_t(partial[j + 0])), sa * b0, total[j + 0]);
    total[j + 1] = fmaf(float(int32_t(partial[j + 1])), sa * b1, total[j + 1]);
    total[j + 2] = fmaf(float(int32_t(partial[j + 2])), sa * b2, total[j + 2]);
    total[j + 3] = fmaf(float(int32_t(partial[j + 3])), sa * b3, total[j + 3]);
  }
}

CUTE_DEVICE uint32_t bf16_pair(float low, float high) {
  uint32_t packed;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;"
      : "=r"(packed) : "f"(high), "f"(low));
  return packed;
}

template <int P, class LA, class LB, class MMA>
__global__ __launch_bounds__(384, 1) void kernel(
    float const *SA, float const *SB, bfloat16_t *C, int M, int N, int K,
    LA la, LB lb, MMA mma, const __grid_constant__ CUtensorMap map_a,
    const __grid_constant__ CUtensorMap map_b) {
  static_assert(P == 2 || P == 3, "The operand ring supports two or three slots");
  static_assert(cosize_v<LA> == 128 * 256 && cosize_v<LB> == 128 * 256);
  extern __shared__ __align__(128) char raw[];
  auto &s = *reinterpret_cast<Storage<LA, LB, P> *>(raw);
  constexpr int EpilogueColumns = Storage<LA, LB, P>::EpilogueColumns;
  constexpr int ExchangeSwizzleShift = EpilogueColumns == 128 ? 4 : 3;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int peer = block_rank_in_cluster();
  int cluster = blockIdx.x / 2, clusters = gridDim.x / 2;
  int mt = M / 256, nt = N / 256 + (N % 256 != 0), groups = K / 256;
  int tiles = mt * nt;
  TMEM::Allocator2Sm allocator;
  if (warp == 0)
    allocator.allocate(512, &s.tmem);
  if (threadIdx.x == 0) {
    CUTE_UNROLL
    for (int i = 0; i < P; ++i) {
      initialize_barrier(s.ab_full[i], 1);
      initialize_barrier(s.ab_empty[i], 1);
      // Each producer lane arrives once normally and once after cp.async.
      initialize_barrier(s.sf_full[i], 64);
      initialize_barrier(s.sf_empty[i], 8);
    }
    CUTE_UNROLL
    for (int i = 0; i < 2; ++i) {
      initialize_barrier(s.acc_full[i], 1);
      // One release from each of the eight consumer warps in both CTAs.
      initialize_barrier(s.acc_empty[i], 16);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  cluster_sync();
  if (warp == 0)
    allocator.release_allocation_lock();

  if (warp < 4) {
    release_registers();
    if (warp == 2) {
      int sequence = 0;
      for (int tile = cluster; tile < tiles; tile += clusters) {
        int im, in;
        tile_coordinates(tile, mt, nt, im, in);
        for (int g = 0; g < groups; ++g, ++sequence) {
          int slot = sequence % P;
          if (sequence >= P)
            wait_barrier(s.ab_empty[slot], ((sequence / P) - 1) & 1);
          if (lane == 0) {
            // The 2SM TMA instruction completes transactions on the leader
            // CTA's barrier even when issued by the peer CTA.
            if (peer == 0)
              set_barrier_transaction_bytes(s.ab_full[slot], 2 * 65536);
            int row_a = im * 256 + peer * 128;
            int row_b = in * 256 + peer * 128;
            SM100_TMA_2SM_LOAD_2D::copy(&map_a, &s.ab_full[slot], 0,
                                       s.a[slot], g * 256, row_a);
            SM100_TMA_2SM_LOAD_2D::copy(&map_a, &s.ab_full[slot], 0,
                                       s.a[slot] + 128 * 128,
                                       g * 256 + 128, row_a);
            SM100_TMA_2SM_LOAD_2D::copy(&map_b, &s.ab_full[slot], 0,
                                       s.b[slot], g * 256, row_b);
            SM100_TMA_2SM_LOAD_2D::copy(&map_b, &s.ab_full[slot], 0,
                                       s.b[slot] + 128 * 128,
                                       g * 256 + 128, row_b);
          }
        }
      }
    } else if (warp == 3) {
      int sequence = 0;
      for (int tile = cluster; tile < tiles; tile += clusters) {
        int im, in;
        tile_coordinates(tile, mt, nt, im, in);
        for (int g = 0; g < groups; ++g, ++sequence) {
          int slot = sequence % P;
          if (sequence >= P)
            wait_barrier(s.sf_empty[slot], ((sequence / P) - 1) & 1);
          CUTE_UNROLL
          for (int row = lane; row < 128; row += 32)
            SM80_CP_ASYNC_CACHEALWAYS<float>::copy(
                SA[(im * 256 + peer * 128 + row) * groups + g],
                s.scale_a[slot][row]);
          CUTE_UNROLL
          for (int col = lane * 4; col < 256; col += 128) {
            // N may end after the first 128 columns of a 2SM tile. The
            // producer still completes every barrier arrival for that tile.
            if (in * 256 + col < N)
              SM80_CP_ASYNC_CACHEALWAYS<uint4>::copy(
                  *reinterpret_cast<uint4 const *>(SB + g * N + in * 256 + col),
                  *reinterpret_cast<uint4 *>(s.scale_b[slot] + col));
            else
              *reinterpret_cast<uint4 *>(s.scale_b[slot] + col) = uint4{0, 0, 0, 0};
          }
          cutlass::arch::cpasync_barrier_arrive_noinc(&s.sf_full[slot]);
          release_local(&s.sf_full[slot]);
          __syncwarp();
        }
      }
    } else if (warp == 0 && peer == 0) {
      auto identity = make_identity_tensor(make_shape(_256{}, _256{}));
      auto cta = mma.get_slice(_0{});
      auto acc = cta.make_fragment_C(cta.partition_C(identity));
      int sequence = 0;
      for (int tile = cluster; tile < tiles; tile += clusters) {
        for (int g = 0; g < groups; ++g, ++sequence) {
          int slot = sequence % P, stage = sequence % 2;
          if (sequence >= 2)
            wait_barrier(s.acc_empty[stage], ((sequence / 2) - 1) & 1);
          wait_barrier(s.ab_full[slot], (sequence / P) & 1);
          acc.data() = s.tmem + stage * 256;
          auto a = make_tensor(make_smem_ptr(s.a[slot]), la);
          auto b = make_tensor(make_smem_ptr(s.b[slot]), lb);
          auto frag_a = cta.make_fragment_A(a);
          auto frag_b = cta.make_fragment_B(b);
          mma.accumulate_ = UMMA::ScaleOut::Zero;
          CUTE_UNROLL
          for (int kk = 0; kk < size<2>(frag_a); ++kk) {
            gemm(mma, frag_a(_, _, kk), frag_b(_, _, kk), acc);
            mma.accumulate_ = UMMA::ScaleOut::One;
          }
          cutlass::arch::umma_arrive_multicast_2x1SM(&s.ab_empty[slot], 3);
          cutlass::arch::umma_arrive_multicast_2x1SM(&s.acc_full[stage], 3);
        }
      }
    }
  } else {
    acquire_registers();
    int part = (warp - 4) / 4;
    int row = (threadIdx.x - 128) % 128;
    int sequence = 0;
    for (int tile = cluster; tile < tiles; tile += clusters) {
      int im, in;
      tile_coordinates(tile, mt, nt, im, in);
      float total[128];
      CUTE_UNROLL
      for (int i = 0; i < 128; ++i)
        total[i] = 0.f;
      for (int g = 0; g < groups; ++g, ++sequence) {
        int slot = sequence % P, stage = sequence % 2;
        wait_barrier(s.acc_full[stage], (sequence / 2) & 1);
        wait_barrier(s.sf_full[slot], (sequence / P) & 1);
        float sa = s.scale_a[slot][row];
        uint32_t address = s.tmem + stage * 256 + part * 128 +
                           ((row / 32) * 32 << 16);
        uint32_t p0[32], p1[32], p2[32];
        load_partial(address, p0);
        load_partial(address + 32, p1);
        cutlass::arch::fence_view_async_tmem_load();
        // Distance-two prefetch, with waits only every two sub-tiles. The
        // current fragment is consumed before its register buffer is reused.
        load_partial(address + 64, p2);
        promote(p0, sa, s.scale_b[slot] + part * 128, total);
        load_partial(address + 96, p0);
        promote(p1, sa, s.scale_b[slot] + part * 128 + 32, total + 32);
        cutlass::arch::fence_view_async_tmem_load();
        promote(p2, sa, s.scale_b[slot] + part * 128 + 64, total + 64);
        promote(p0, sa, s.scale_b[slot] + part * 128 + 96, total + 96);
        // All lanes must finish reading this TMEM stage and its scales before
        // the elected lane allows either producer to reuse a ring slot.
        __syncwarp();
        if (lane == 0) {
          release_local(&s.sf_empty[slot]);
          release_leader(&s.acc_empty[stage]);
        }
      }
      CUTE_UNROLL
      for (int sub = 0; sub < 128 / EpilogueColumns; ++sub) {
        CUTE_UNROLL
        for (int j = 0; j < EpilogueColumns; j += 8) {
          int offset = row * EpilogueColumns + j;
          int q = sub * EpilogueColumns + j;
          uint4 packed{bf16_pair(total[q + 0], total[q + 1]),
                       bf16_pair(total[q + 2], total[q + 3]),
                       bf16_pair(total[q + 4], total[q + 5]),
                       bf16_pair(total[q + 6], total[q + 7])};
          // Eight lanes in each 128-byte shared-store transaction use all
          // 32 banks. Scalar BF16 stores alias banks across the 32 rows.
          *reinterpret_cast<uint4 *>(
              s.exchange[part] +
                  (offset ^ ((offset >> ExchangeSwizzleShift) & 56))) = packed;
        }
        // Each promotion group owns a separate named barrier and exchange.
        cutlass::arch::NamedBarrier::sync(128, uint32_t(part));
        CUTE_UNROLL
        for (int v = row; v < 128 * EpilogueColumns / 8; v += 128) {
          int offset = v * 8;
          int local_row = offset / EpilogueColumns;
          int col = offset % EpilogueColumns;
          int global_row = im * 256 + peer * 128 + local_row;
          int global_col = in * 256 + part * 128 + sub * EpilogueColumns + col;
          if (global_col + 8 <= N)
            *reinterpret_cast<uint4 *>(C + global_row * N + global_col) =
                *reinterpret_cast<uint4 const *>(
                    s.exchange[part] +
                        (offset ^ ((offset >> ExchangeSwizzleShift) & 56)));
        }
        cutlass::arch::NamedBarrier::sync(128, uint32_t(part));
      }
    }
  }
  // Persistent producers may finish ahead of the last consumer. Joining all
  // roles and both CTAs drains async copies, MMAs and TMEM loads before free.
  __syncthreads();
  cluster_sync();
  if (warp == 0)
    allocator.free(s.tmem, 512);
}

template <int P>
int launch(void const *a, void const *b, void const *sa, void const *sb,
           void *c, int M, int N, int K, cudaStream_t stream) {
  auto fallback = [&]() {
    return grain_warp_g256(a, b, sa, sb, c, M, N, K, 0, stream);
  };
  if (!a || !b || !sa || !sb || !c || M <= 0 || N <= 0 || K <= 0 ||
      M % 256 || N % 128 || K % 256 ||
      reinterpret_cast<uintptr_t>(a) % 16 ||
      reinterpret_cast<uintptr_t>(b) % 16 ||
      reinterpret_cast<uintptr_t>(c) % 16 ||
      reinterpret_cast<uintptr_t>(sa) % 4 ||
      reinterpret_cast<uintptr_t>(sb) % 16)
    return fallback();
  auto mma = make_tiled_mma(SM100_MMA_S8_2x1SM_SS<
      int8_t, int8_t, int32_t, 256, 256, UMMA::Major::K, UMMA::Major::K>{});
  auto la = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<int8_t>{},
      partition_shape_A(mma, make_shape(_256{}, _256{})));
  auto lb = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<int8_t>{},
      partition_shape_B(mma, make_shape(_256{}, _256{})));
  CUtensorMap map_a, map_b;
  uint64_t dims_a[2] = {uint64_t(K), uint64_t(M)};
  uint64_t dims_b[2] = {uint64_t(K), uint64_t(N)};
  uint64_t stride[1] = {uint64_t(K)};
  uint32_t box[2] = {128, 128}, element_stride[2] = {1, 1};
  auto encode = [&](CUtensorMap *map, void const *ptr, uint64_t *dims) {
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(ptr), dims,
        stride, box, element_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  if (encode(&map_a, a, dims_a) != CUDA_SUCCESS ||
      encode(&map_b, b, dims_b) != CUDA_SUCCESS)
    return fallback();
  int device, sm_count;
  auto status = cudaGetDevice(&device);
  if (status != cudaSuccess)
    return int(status);
  status = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  if (status != cudaSuccess)
    return int(status);
  int tiles = (M / 256) * (N / 256 + (N % 256 != 0));
  int clusters = tiles < sm_count / 2 ? tiles : sm_count / 2;
  if (clusters <= 0)
    return fallback();
  auto fn = kernel<P, decltype(la), decltype(lb), decltype(mma)>;
  constexpr int bytes = sizeof(Storage<decltype(la), decltype(lb), P>);
  status = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status != cudaSuccess)
    return int(status);
  cutlass::ClusterLaunchParams params{dim3(clusters * 2), dim3(384),
                                     dim3(2, 1, 1), bytes, stream};
  auto result = cutlass::launch_kernel_on_cluster(
      params, reinterpret_cast<void const *>(fn), static_cast<float const *>(sa),
      static_cast<float const *>(sb), static_cast<bfloat16_t *>(c), M, N, K,
      la, lb, mma, map_a, map_b);
  if (result != cutlass::Status::kSuccess)
    return -1000 - int(result);
  return int(cudaGetLastError());
}
} // namespace thor_ws2
