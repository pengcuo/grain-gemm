/***************************************************************************************************
 * Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once
// GrainGEMM's light BF16 epilogues as CUTLASS collective components.
// These port the tested CuTe store strategies while retaining the CUTLASS
// argument/parameter contract and GemmUniversalAdapter composition.
#include <cuda_bf16.h>
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cute/tensor.hpp"

namespace grain_cutlass {
using namespace cute;

template<int BM, int BN, int StoreBits>
struct Bf16EpilogueStorage {
  cute::array_aligned<cutlass::bfloat16_t, BM * BN> output;
};
template<int BM, int BN>
struct Bf16EpilogueStorage<BM,BN,32> {};

template<class StrideCD, int BM, int BN, int StoreBits>
class Bf16Epilogue : public cutlass::epilogue::collective::DefaultEpilogue<
    cutlass::bfloat16_t,StrideCD,StrideCD,
    cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t,1,float,float,
        cutlass::epilogue::thread::ScaleType::Nothing>,
    cutlass::gemm::EpilogueDefault> {
  using Base = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::bfloat16_t,StrideCD,StrideCD,
      cutlass::epilogue::thread::LinearCombination<cutlass::bfloat16_t,1,float,float,
          cutlass::epilogue::thread::ScaleType::Nothing>,
      cutlass::gemm::EpilogueDefault>;
public:
  using Params = typename Base::Params;
  using SharedStorage = Bf16EpilogueStorage<BM,BN,StoreBits>;
  using TensorStorage = SharedStorage;
  static_assert(StoreBits == 128 || StoreBits == 32);
  Params params;
  CUTLASS_DEVICE Bf16Epilogue(Params const& p) : Base(p), params(p) {}

  template<class ProblemShape, class BlockShape, class BlockCoord,
           class FrgEngine, class FrgLayout, class TiledMma, class Residue>
  CUTLASS_DEVICE void operator()(ProblemShape problem, BlockShape block,
      BlockCoord coord, Tensor<FrgEngine,FrgLayout> const& accum,
      TiledMma mma, Residue, int thread_idx, char* smem_buf) {
    auto [M,N,K,L] = problem;
    auto [tile_m,tile_n,tile_k,tile_l] = coord;
    Tensor mD = make_tensor(make_gmem_ptr(params.ptr_D), make_shape(M,N,L), params.dD);
    Tensor gD = local_tile(mD(_,_,tile_l), take<0,2>(block), make_coord(tile_m,tile_n));
    auto thr_mma = mma.get_thread_slice(thread_idx);
    if constexpr (StoreBits == 128) {
      // The mainloop has already drained cp.async and synchronized the CTA.
      // Convert before the shared-memory exchange, halving that exchange's
      // traffic and capacity compared with the standard FP32 epilogue.
      SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem_buf);
      Tensor sD = make_tensor(make_smem_ptr(storage.output.data()),
          Layout<Shape<Int<BM>,Int<BN>>,Stride<Int<BN>,_1>>{});
      Tensor tDsD = thr_mma.partition_C(sD);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        tDsD(i) = cutlass::bfloat16_t(accum(i));
      }
      __syncthreads();
      CUTLASS_PRAGMA_UNROLL
      for (int v = thread_idx; v < BM * BN / 8; v += int(size(TiledMma{}))) {
        int local = v * 8;
        auto data = reinterpret_cast<uint4 const*>(storage.output.data())[v];
        *reinterpret_cast<uint4*>(&gD(local / BN,local % BN)) = data;
      }
    } else {
      // SM80_16x8x32 accumulator fragments have adjacent N values in each
      // consecutive pair. The aligned/full-tile launcher guarantees that
      // neither member crosses a tensor or tile boundary.
      Tensor tDgD = thr_mma.partition_C(gD);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accum); i += 2) {
        __nv_bfloat162 pair = __floats2bfloat162_rn(accum(i),accum(i + 1));
        *reinterpret_cast<__nv_bfloat162*>(&tDgD(i)) = pair;
      }
    }
  }
};
} // namespace grain_cutlass
