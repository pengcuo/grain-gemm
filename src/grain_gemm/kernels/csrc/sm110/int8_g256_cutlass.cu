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
// Adapted from nvidia-cosmos/cosmos-int8-gemm at
// 8977529aa23f70f02bfc36559599f00d59792996, Thor blockwise cfg20/cfg21.
// Preserve all 32 warp-MMA configurations and their original C ABI.
#include "../sm12x/cutlass/int8_g256.cu"
#include <cstddef>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

// CUTLASS device assertions extend live ranges across the promotion pipeline
// and cause substantial register spills. Public checks and CUDA status checks
// below remain active in this production build.
#ifndef NDEBUG
#error "Build the Thor CUTLASS pipeline with -DNDEBUG (tools/build_cutlass.py)"
#endif

// Experimental promotion modes change the public scaling semantics.
#if defined(G128_OPT_BIAS) || defined(G128_OPT_PACKED) ||                      \
    defined(G128_EXP_NO_FMA) || defined(G128_EXP_NO_LOAD) ||                   \
    defined(G128_EXP_NO_SFB) || defined(G128_DBG_BIAS_OLDMATH) ||              \
    defined(G128_DBG_BIAS_FORCEMATH) || defined(G128_DBG_REARM_PASSIVE) ||     \
    defined(G128_DBG_REARM_EXIT_NOBAR) || defined(G128_DBG_REARM_EXIT_BAR) ||  \
    defined(G128_DBG_REARM_NOSTORE)
#error "Experimental promotion modes are unsupported by the GrainGEMM API"
#endif

namespace grain_cutlass_sm110 {
using namespace cute;

template <int EpilogueN> struct BlockwiseG256 {
  using Tile = Shape<_256, _256, _256>;
  using Cluster = Shape<_2, _1, _1>;
  using Output = cutlass::bfloat16_t;
  // Directly consume the existing public layouts SA[M,K/256], SB[K/256,N].
  using ScaleConfig =
      cutlass::detail::Sm100BlockwiseScaleConfig<1, 1, 256, UMMA::Major::K,
                                                 UMMA::Major::MN>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Scalar =
      cutlass::epilogue::fusion::Sm90ScalarBroadcast<float, Stride<_0, _0, _0>,
                                                     2, cutlass::multiplies>;
  using Multiply = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, Output, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      Multiply, Scalar, cutlass::epilogue::fusion::Sm90AccFetch>;
  // Preserve Cosmos cfg20's structured auto tiler, not just its dimensions.
  using EpilogueTile = cute::conditional_t<
      EpilogueN == 32, cutlass::epilogue::collective::EpilogueTileAuto,
      Shape<_128, Int<EpilogueN>>>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      EpilogueTile, float, float, void,
      cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::TmaWarpSpecialized2Sm, EVT>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, int8_t,
      tuple<cutlass::layout::RowMajor, LayoutSFA>, 16, int8_t,
      tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 16, int32_t, Tile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                      Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static typename Gemm::Arguments arguments(void const *a, void const *b,
                                            void const *sa, void const *sb,
                                            void *c, int M, int N, int K,
                                            int L = 1) {
    auto stride_a = cutlass::make_cute_packed_stride(typename Kernel::StrideA{},
                                                     make_shape(M, K, L));
    auto stride_b = cutlass::make_cute_packed_stride(typename Kernel::StrideB{},
                                                     make_shape(N, K, L));
    auto stride_d = cutlass::make_cute_packed_stride(typename Kernel::StrideD{},
                                                     make_shape(M, N, L));
    auto layout_sa =
        ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, L));
    auto layout_sb =
        ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, L));
    typename Gemm::Arguments args{
        L == 1 ? cutlass::gemm::GemmUniversalMode::kGemm
               : cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, K, L},
        {static_cast<int8_t const *>(a), stride_a,
         static_cast<int8_t const *>(b), stride_b,
         static_cast<float const *>(sa), layout_sa,
         static_cast<float const *>(sb), layout_sb},
        {{}, nullptr, stride_d, static_cast<Output *>(c), stride_d}};
    args.epilogue.thread = {{{1.f, 1.f}, {nullptr, nullptr}, {}}, {}, {}};
    args.scheduler.max_swizzle_size = int64_t(K) * N > (16 << 20) ? 8 : 1;
    args.scheduler.raster_order =
        cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;
    int device;
    if (cudaGetDevice(&device) == cudaSuccess)
      args.hw_info.device_id = device;
    return args;
  }

  static size_t workspace_bytes(int M, int N, int K, int L = 1) {
    return Gemm::get_workspace_size(
        arguments(nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, L));
  }

  static int launch(void const *a, void const *b, void const *sa,
                    void const *sb, void *c, int M, int N, int K,
                    cudaStream_t stream, void *workspace, size_t bytes) {
    auto args = arguments(a, b, sa, sb, c, M, N, K);
    Gemm gemm;
    // Preserve permissive legacy pointer offsets and sizes via the warp kernel.
    // SFB's vector cp.async requires 16-byte alignment; SFA uses scalar copies.
    if (reinterpret_cast<uintptr_t>(a) % 16 ||
        reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(sa) % alignof(float) ||
        reinterpret_cast<uintptr_t>(sb) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16 ||
        gemm.can_implement(args) != cutlass::Status::kSuccess)
      return grain_cutlass_g256(a, b, sa, sb, c, M, N, K, 0, stream);
    size_t required = Gemm::get_workspace_size(args);
    if ((required && !workspace) || bytes < required)
      return grain_cutlass_g256(a, b, sa, sb, c, M, N, K, 0, stream);
    auto status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
      return -1000 - int(status);
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
      return -1000 - int(status);
    return int(cudaGetLastError());
  }

  static int launch_batched(void const *a, void const *b, void const *sa,
                            void const *sb, void *c, int M, int N, int K,
                            int L, cudaStream_t stream, void *workspace,
                            size_t bytes) {
    // The single-GEMM fallback does not cover other experts. Reject unsupported
    // batched arguments rather than returning an incomplete result.
    if (reinterpret_cast<uintptr_t>(a) % 16 ||
        reinterpret_cast<uintptr_t>(b) % 16 ||
        reinterpret_cast<uintptr_t>(sa) % alignof(float) ||
        reinterpret_cast<uintptr_t>(sb) % 16 ||
        reinterpret_cast<uintptr_t>(c) % 16)
      return int(cudaErrorInvalidValue);
    auto args = arguments(a, b, sa, sb, c, M, N, K, L);
    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess)
      return -1000 - int(status);
    size_t required = Gemm::get_workspace_size(args);
    if ((required && !workspace) || bytes < required)
      return int(cudaErrorInvalidValue);
    // Scheduler state is initialized per invocation on the caller's stream.
    // Workspace is supplied by PyTorch, so capture never calls cudaMalloc.
    status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess)
      return -1000 - int(status);
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess)
      return -1000 - int(status);
    return int(cudaGetLastError());
  }
};

inline bool valid_shape(int M, int N, int K) {
  return M > 0 && N > 0 && K > 0 && M % 64 == 0 && N % 64 == 0 && K % 256 == 0;
}
} // namespace grain_cutlass_sm110

extern "C" size_t grain_cutlass_workspace_bytes(int M, int N, int K,
                                                int config) {
  if (!grain_cutlass_sm110::valid_shape(M, N, K))
    return 0;
  switch (config) {
  case 32:
    return grain_cutlass_sm110::BlockwiseG256<32>::workspace_bytes(M, N, K);
  case 33:
    return grain_cutlass_sm110::BlockwiseG256<16>::workspace_bytes(M, N, K);
  default:
    return 0;
  }
}

extern "C" int grain_cutlass_g256_sm110(void const *a, void const *b,
                                        void const *sa, void const *sb, void *c,
                                        int M, int N, int K, int config,
                                        void *stream, void *workspace,
                                        size_t bytes) {
  if (config >= 0 && config < 32)
    return grain_cutlass_g256(a, b, sa, sb, c, M, N, K, config, stream);
  if (!a || !b || !sa || !sb || !c ||
      !grain_cutlass_sm110::valid_shape(M, N, K))
    return int(cudaErrorInvalidValue);
  auto st = static_cast<cudaStream_t>(stream);
  switch (config) {
  case 32:
    return grain_cutlass_sm110::BlockwiseG256<32>::launch(
        a, b, sa, sb, c, M, N, K, st, workspace, bytes);
  case 33:
    return grain_cutlass_sm110::BlockwiseG256<16>::launch(
        a, b, sa, sb, c, M, N, K, st, workspace, bytes);
  default:
    return int(cudaErrorInvalidValue);
  }
}

extern "C" size_t grain_cutlass_batched_workspace_bytes(int M, int N, int K,
                                                        int L, int config) {
  if (L <= 0 || !grain_cutlass_sm110::valid_shape(M, N, K))
    return 0;
  switch (config) {
  case 32:
    return grain_cutlass_sm110::BlockwiseG256<32>::workspace_bytes(M, N, K, L);
  case 33:
    return grain_cutlass_sm110::BlockwiseG256<16>::workspace_bytes(M, N, K, L);
  default:
    return 0;
  }
}

// A[L,M,K], B[L,N,K], SA[L,M,K/256], SB[L,K/256,N], D[L,M,N].
// All arrays are contiguous. The batch dimension selects the same expert in
// both operands and both scale tensors; no routing is performed here.
extern "C" int grain_cutlass_g256_batched_sm110(
    void const *a, void const *b, void const *sa, void const *sb, void *c,
    int M, int N, int K, int L, int config, void *stream, void *workspace,
    size_t bytes) {
  if (!a || !b || !sa || !sb || !c || L <= 0 ||
      !grain_cutlass_sm110::valid_shape(M, N, K))
    return int(cudaErrorInvalidValue);
  auto st = static_cast<cudaStream_t>(stream);
  switch (config) {
  case 32:
    return grain_cutlass_sm110::BlockwiseG256<32>::launch_batched(
        a, b, sa, sb, c, M, N, K, L, st, workspace, bytes);
  case 33:
    return grain_cutlass_sm110::BlockwiseG256<16>::launch_batched(
        a, b, sa, sb, c, M, N, K, L, st, workspace, bytes);
  default:
    return int(cudaErrorInvalidValue);
  }
}
