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
// CUTLASS 3.x composition for signed INT8 G256 GEMM on SM121.
// This translation unit does not call the independent grain_cute kernel.
#include <cuda_runtime.h>
#include <cstdint>
#include "cute/tensor.hpp"
#include "cutlass/epilogue/collective/collective_epilogue.hpp"
#include "cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "grain_cutlass_kernel.hpp"
#include "grain_cutlass_epilogue.hpp"

namespace grain_cutlass {
using namespace cute;

template<int BM, int BN, int Stages, int WM, int WN, bool AsyncScale = false, int StoreBits = 0>
int launch(void const* a, void const* b, void const* sa, void const* sb,
           void* c, int M, int N, int K, cudaStream_t stream) {
  if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % 256)
    return int(cudaErrorInvalidValue);
  using StrideAB = Stride<int, _1, _0>;
  using StrideCD = Stride<int, _1, _0>;
  using TileShape = Shape<Int<BM>, Int<BN>, _128>;
  using SmemAtom = decltype(composition(Swizzle<3,4,3>{}, Layout<Shape<_8,_128>, Stride<_128,_1>>{}));
  using GmemCopy = decltype(make_tiled_copy(
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, int8_t>{},
      Layout<Shape<Int<WM*WN*4>,_8>, Stride<_8,_1>>{}, Layout<Shape<_1,_16>>{}));
  using Mma = decltype(make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
      Layout<Shape<Int<WM>,Int<WN>>>{}, Tile<Int<WM*16>,Int<WN*16>,_32>{}));
  using SmemCopy = Copy_Atom<SM75_U32x4_LDSM_N,int8_t>;
  using Mainloop = CollectiveG256<Stages,TileShape,int8_t,StrideAB,int8_t,StrideAB,
      Mma,GmemCopy,SmemAtom,SmemCopy,identity,GmemCopy,SmemAtom,SmemCopy,identity,AsyncScale,(StoreBits != 32)>;
  // Official CUTLASS vectorized epilogue, with an FP32 shared-memory exchange
  // followed by 128-bit BF16 global stores. It reuses mainloop shared memory.
  using Output = cutlass::bfloat16_t;
  using ThreadOp = cutlass::epilogue::thread::LinearCombination<Output,1,float,float,
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using EpiSmem = Layout<Shape<Int<BM>,Int<BN>>,Stride<Int<BN>,_1>>;
  using EpiR2S = Copy_Atom<UniversalCopy<uint64_t>,float>;
  using EpiS2R = decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>,float>{},
      Layout<Shape<Int<(32*WM*WN)/(BN/8)>,Int<BN/8>>,Stride<Int<BN/8>,_1>>{},
      Layout<Shape<_1,_8>>{}));
  using EpiR2G = Copy_Atom<UniversalCopy<uint128_t>,Output>;
  using StandardEpilogue = cutlass::epilogue::collective::Epilogue<
      StrideCD,StrideCD,ThreadOp,EpiSmem,EpiR2S,EpiS2R,EpiR2G>;
  using Epilogue = cute::conditional_t<StoreBits == 0, StandardEpilogue,
      Bf16Epilogue<StrideCD,BM,BN,StoreBits>>;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int>,Mainloop,Epilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  typename Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = make_shape(M,N,K);
  args.mainloop = {static_cast<int8_t const*>(a), make_stride(K,_1{},_0{}),
                   static_cast<int8_t const*>(b), make_stride(K,_1{},_0{}),
                   static_cast<float const*>(sa),static_cast<float const*>(sb),K/256,N};
  args.epilogue.thread.alpha = 1.0f;
  args.epilogue.thread.beta = 0.0f;
  args.epilogue.ptr_C = nullptr;
  args.epilogue.dC = make_stride(N,_1{},_0{});
  args.epilogue.ptr_D = static_cast<Output*>(c);
  args.epilogue.dD = make_stride(N,_1{},_0{});
  Gemm gemm;
  auto status = gemm.can_implement(args);
  if (status != cutlass::Status::kSuccess) return -1000-int(status);
  status = gemm.initialize(args,nullptr,stream);
  if (status != cutlass::Status::kSuccess) return -1000-int(status);
  status = gemm.run(stream);
  if (status != cutlass::Status::kSuccess) return -1000-int(status);
  return int(cudaGetLastError());
}
} // namespace grain_cutlass

extern "C" int grain_cutlass_g256(void const* a, void const* b, void const* sa,
    void const* sb, void* c, int M, int N, int K, int config, void* stream) {
  if (!a || !b || !sa || !sb || !c) return int(cudaErrorInvalidValue);
  cudaStream_t st = static_cast<cudaStream_t>(stream);
  using grain_cutlass::launch;
  switch(config) {
#ifndef GRAIN_CUTLASS_SINGLE_CONFIG
    // 8..15: register scales, convert to BF16 before shared/vector stores.
    case 11: return launch<128,64,3,2,2,false,128>(a,b,sa,sb,c,M,N,K,st);
    case 15: return launch<128,128,3,4,2,false,128>(a,b,sa,sb,c,M,N,K,st);
    // 16..23: cp.async scale ring, direct BF16 pair stores.
    case 23: return launch<128,128,3,4,2,true,32>(a,b,sa,sb,c,M,N,K,st);
    // 24..31: register scales, direct BF16 pair stores (scale-path ablation).
#endif
    default: return int(cudaErrorInvalidValue);
  }
}
extern "C" char const* grain_cutlass_error_string(int code) {
  if (code <= -1000) return cutlassGetStatusString(static_cast<cutlass::Status>(-1000-code));
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
