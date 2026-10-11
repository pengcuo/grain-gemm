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

// Adapted from CUTLASS cute/tutorial/blackwell/01_mma_sm100.cu at
// 098de2a652cf8f00fd70b2df54051c7eccbb855a, and its 02_mma_tma tutorial.
// INT8 G256 scaling on Thor, pipelined TMA, two TMEM group buffers,
// a separate scale ring and a vectorized BF16 epilogue.
// Retain the warp MMA candidates for small tiles and independent tuning.
#define grain_cute_g256 grain_warp_g256
#include "../sm12x/int8_g256_cute.cu"
#undef grain_cute_g256
#include <cuda.h>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>

namespace thor {
using namespace cute;
template <class LA, class LB, int P> struct Storage {
  alignas(128) int8_t a[P][cosize_v<LA>];
  alignas(128) int8_t b[P][cosize_v<LB>];
  alignas(16) uint64_t barrier[2];
  uint64_t tma[P];
  float scaleA[P + 1][128];
  float scaleB[P + 1][256];
  uint32_t tmem;
};

template <int BM, int BN, int P, class LA, class LB, class MMA>
__global__ __launch_bounds__(128) void kernel(
    float const *SA, float const *SB, bfloat16_t *C, int M, int N, int K, LA la,
    LB lb, MMA mma, const __grid_constant__ CUtensorMap mapA,
    const __grid_constant__ CUtensorMap mapB) {
  extern __shared__ __align__(128) char raw[];
  auto &shared = *reinterpret_cast<Storage<LA, LB, P> *>(raw);
  // Eight M tiles per stripe, with an exact partial final stripe.
  int tm = M / BM, tn = N / BN, pid = blockIdx.x;
  int first = (pid / (8 * tn)) * 8;
  int band = min(8, tm - first);
  int im = first + pid % band, in = (pid % (8 * tn)) / band;
  auto tiler = make_shape(Int<BM>{}, Int<BN>{}, _256{});
  auto coord = make_coord(im, in, _);
  auto gC = local_tile(
      make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, _1{})),
      tiler, coord, Step<_1, _1, X>{});
  auto cta = mma.get_slice(_0{});
  auto tCgC = cta.partition_C(gC);
  auto acc = cta.make_fragment_C(tCgC);
  TMEM::Allocator1Sm allocator;
  if (threadIdx.x < 32)
    allocator.allocate(2 * BN, &shared.tmem);
  if (threadIdx.x == 0) {
    initialize_barrier(shared.barrier[0], 1);
    initialize_barrier(shared.barrier[1], 1);
    for (int i = 0; i < P; ++i)
      initialize_barrier(shared.tma[i], 1);
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();
  acc.data() = shared.tmem;
  if (threadIdx.x < 32)
    allocator.release_allocation_lock();

  using Load = conditional_t<BM == 64, SM100_TMEM_LOAD_16dp256b1x,
                             SM100_TMEM_LOAD_32dp32b4x>;
  auto t2r = make_tmem_copy(Load{}, acc);
  auto copy_thread = t2r.get_slice(threadIdx.x);
  auto tC = copy_thread.partition_D(tCgC);
  auto partial = make_tensor<int32_t>(shape(tC));
  auto total = make_tensor<float>(shape(tC));
  clear(total);
  auto identity =
      cta.partition_C(make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{})));
  auto positions = copy_thread.partition_D(identity);
  int groups = K / 256;
  auto load = [&](int group, int stage) {
    for (int row = threadIdx.x; row < BM; row += 128)
      shared.scaleA[group % (P + 1)][row] =
          SA[(im * BM + row) * groups + group];
    for (int col = threadIdx.x; col < BN; col += 128)
      shared.scaleB[group % (P + 1)][col] = SB[group * N + in * BN + col];
    if (threadIdx.x == 0) {
      set_barrier_transaction_bytes(shared.tma[stage], (BM + BN) * 256);
      SM90_TMA_LOAD_2D::copy(&mapA, &shared.tma[stage], 0, shared.a[stage],
                             group * 256, im * BM);
      SM90_TMA_LOAD_2D::copy(&mapA, &shared.tma[stage], 0,
                             shared.a[stage] + BM * 128, group * 256 + 128,
                             im * BM);
      SM90_TMA_LOAD_2D::copy(&mapB, &shared.tma[stage], 0, shared.b[stage],
                             group * 256, in * BN);
      SM90_TMA_LOAD_2D::copy(&mapB, &shared.tma[stage], 0,
                             shared.b[stage] + BN * 128, group * 256 + 128,
                             in * BN);
    }
  };
  // Each committed group is bounded, including the minimum K=256 case.
  for (int stage = 0; stage < P - 1; ++stage)
    if (stage < groups)
      load(stage, stage);
  for (int iteration = 0; iteration <= groups; ++iteration) {
    int group = iteration;
    if (iteration < groups) {
      // Reused TMA slots alternate completion parity once per pipeline cycle.
      int stage = group % P;
      acc.data() = shared.tmem + (group % 2) * BN;
      wait_barrier(shared.tma[stage], (group / P) & 1);
      auto sA = make_tensor(make_smem_ptr(shared.a[stage]), la);
      auto sB = make_tensor(make_smem_ptr(shared.b[stage]), lb);
      auto fragA = cta.make_fragment_A(sA);
      auto fragB = cta.make_fragment_B(sB);
      if (threadIdx.x < 32) {
        mma.accumulate_ = UMMA::ScaleOut::Zero;
        CUTE_UNROLL
        for (int kk = 0; kk < size<2>(fragA); ++kk) {
          gemm(mma, fragA(_, _, kk), fragB(_, _, kk), acc);
          mma.accumulate_ = UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive(&shared.barrier[group % 2]);
      }
      // Prefetch a different stage while the current group's MMA runs.
      int next = group + P - 1;
      if (next < groups)
        load(next, next % P);
    }
    if (iteration > 0) {
      group = iteration - 1;
      acc.data() = shared.tmem + (group % 2) * BN;
      auto previous = copy_thread.partition_S(acc);
      if constexpr (BM == 128) {
        // Load and consume sixteen columns at a time: avoid a second BN-sized
        // register fragment and hundreds of simultaneously live scale
        // addresses.
        float scale_a = shared.scaleA[group % (P + 1)][threadIdx.x];
        CUTE_UNROLL
        for (int i = 0; i < BN; i += 16) {
          uint32_t v[16];
          uint32_t address = shared.tmem + (group % 2) * BN +
                             ((threadIdx.x / 32) * 32 << 16) + i;
          SM100_TMEM_LOAD_32dp32b16x::copy(
              address, v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8],
              v[9], v[10], v[11], v[12], v[13], v[14], v[15]);
          cutlass::arch::fence_view_async_tmem_load();
          CUTE_UNROLL
          for (int j = 0; j < 16; ++j)
            total(i + j) = fmaf(float(int32_t(v[j])),
                                scale_a * shared.scaleB[group % (P + 1)][i + j],
                                total(i + j));
        }
      } else {
        copy(t2r, previous, partial);
        cutlass::arch::fence_view_async_tmem_load();
        CUTE_UNROLL
        for (int i = 0; i < size(total); ++i) {
          auto pos = positions(i);
          // Exact existing public layouts: SA[M,K/256], SB[K/256,N].
          float scale = shared.scaleA[group % (P + 1)][get<0>(pos)] *
                        shared.scaleB[group % (P + 1)][get<1>(pos)];
          total(i) = fmaf(float(partial(i)), scale, total(i));
        }
      }
    }
    // Current MMA and previous-group scaling run concurrently in distinct TMEM
    // buffers. Complete the current MMA before reusing its shared operand slot.
    if (iteration < groups)
      wait_barrier(shared.barrier[iteration % 2], (iteration / 2) & 1);
    __syncthreads();
  }
  // All async operand copies are drained at this point. Reuse their storage
  // for a bank-swizzled BF16 exchange and issue coalesced 128-bit global
  // stores.
  auto *exchange = reinterpret_cast<bfloat16_t *>(raw);
  CUTE_UNROLL
  for (int i = 0; i < size(total); ++i) {
    auto pos = positions(i);
    int offset = get<0>(pos) * BN + get<1>(pos);
    exchange[offset ^ ((offset >> 3) & 56)] = bfloat16_t(total(i));
  }
  __syncthreads();
  CUTE_UNROLL
  for (int v = threadIdx.x; v < BM * BN / 8; v += 128) {
    int offset = v * 8;
    int swizzled = offset ^ ((offset >> 3) & 56);
    int row = offset / BN, col = offset % BN;
    *reinterpret_cast<uint4 *>(C + (im * BM + row) * N + in * BN + col) =
        *reinterpret_cast<uint4 *>(exchange + swizzled);
  }
  __syncthreads();
  if (threadIdx.x < 32)
    allocator.free(shared.tmem, 2 * BN);
}

template <int BM, int BN, int P>
int launch(void const *a, void const *b, void const *sa, void const *sb,
           void *c, int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % 256)
    return int(cudaErrorInvalidValue);
  auto mma = make_tiled_mma(SM100_MMA_S8_SS<int8_t, int8_t, int32_t, BM, BN,
                                            UMMA::Major::K, UMMA::Major::K>{});
  auto la = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<int8_t>{},
      partition_shape_A(mma, make_shape(Int<BM>{}, _256{})));
  auto lb = UMMA::tile_to_mma_shape(
      UMMA::Layout_K_SW128_Atom<int8_t>{},
      partition_shape_B(mma, make_shape(Int<BN>{}, _256{})));
  CUtensorMap mapA, mapB;
  uint64_t dimsA[2] = {uint64_t(K), uint64_t(M)},
           dimsB[2] = {uint64_t(K), uint64_t(N)};
  uint64_t stride[1] = {uint64_t(K)};
  uint32_t boxA[2] = {128, BM}, boxB[2] = {128, BN}, elementStride[2] = {1, 1};
  auto encode = [&](CUtensorMap *map, void const *ptr, uint64_t *dims,
                    uint32_t *box) {
    return cuTensorMapEncodeTiled(
        map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<void *>(ptr), dims,
        stride, box, elementStride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  if (encode(&mapA, a, dimsA, boxA) != CUDA_SUCCESS ||
      encode(&mapB, b, dimsB, boxB) != CUDA_SUCCESS)
    return int(cudaErrorInvalidValue);
  auto fn = kernel<BM, BN, P, decltype(la), decltype(lb), decltype(mma)>;
  int bytes = sizeof(Storage<decltype(la), decltype(lb), P>);
  auto status = cudaFuncSetAttribute(
      fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  if (status != cudaSuccess)
    return int(status);
  fn<<<(M / BM) * (N / BN), 128, bytes, stream>>>(
      static_cast<float const *>(sa), static_cast<float const *>(sb),
      static_cast<bfloat16_t *>(c), M, N, K, la, lb, mma, mapA, mapB);
  return int(cudaGetLastError());
}
} // namespace thor
#include "int8_g256_pipeline.cuh"
#include "int8_g256_pipeline_1sm.cuh"
extern "C" int grain_cute_g256(void const *a, void const *b, void const *sa,
                               void const *sb, void *c, int M, int N, int K,
                               int config, void *stream) {
  if (!a || !b || !sa || !sb || !c)
    return int(cudaErrorInvalidValue);
  auto st = static_cast<cudaStream_t>(stream);
  switch (config) {
  case 17:
    return thor::launch<64, 64, 2>(a, b, sa, sb, c, M, N, K, st);
  case 18:
    return thor::launch<64, 128, 2>(a, b, sa, sb, c, M, N, K, st);
  case 19:
    return thor::launch<128, 64, 2>(a, b, sa, sb, c, M, N, K, st);
  case 20:
    return thor::launch<128, 128, 2>(a, b, sa, sb, c, M, N, K, st);
  case 21:
    return thor::launch<128, 256, 2>(a, b, sa, sb, c, M, N, K, st);
  case 22:
    return thor::launch<64, 64, 3>(a, b, sa, sb, c, M, N, K, st);
  case 23:
    return thor::launch<64, 128, 3>(a, b, sa, sb, c, M, N, K, st);
  case 24:
    return thor::launch<128, 64, 3>(a, b, sa, sb, c, M, N, K, st);
  case 25:
    return thor::launch<128, 128, 3>(a, b, sa, sb, c, M, N, K, st);
  case 26:
    return thor_ws2::launch<3>(a, b, sa, sb, c, M, N, K, st);
  case 27:
    return thor_ws2::launch<2>(a, b, sa, sb, c, M, N, K, st);
  case 28:
    return thor_ws1::launch<128, 128, 3>(a, b, sa, sb, c, M, N, K, st);
  case 29:
    return thor_ws1::launch<128, 256, 2>(a, b, sa, sb, c, M, N, K, st);
  default:
    return grain_warp_g256(a, b, sa, sb, c, M, N, K, config, stream);
  }
}
