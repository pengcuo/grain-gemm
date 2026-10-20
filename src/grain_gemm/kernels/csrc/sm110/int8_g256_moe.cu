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
// MoE-only tile families for the Cosmos-derived Thor INT8 G256 collective.
// The existing single-GEMM translation unit and configurations are independent.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cute/tensor.hpp>
#include <cutlass/detail/blockwise_scale_layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>

#ifndef NDEBUG
#error "Build the Thor MoE collective with -DNDEBUG"
#endif
#if defined(G128_OPT_BIAS) || defined(G128_OPT_PACKED) || \
    defined(G128_EXP_NO_FMA) || defined(G128_EXP_NO_LOAD) || defined(G128_EXP_NO_SFB)
#error "MoE G256 requires the exact scaled INT32-to-FP32 promotion path"
#endif

// Private scheduler tag: only this translation unit's MoE configurations use it.
namespace cutlass::gemm { struct GrainMoeCountsScheduler {}; }

namespace grain_moe_sm110 {
// Preserve the first work tile: the collective enters its producer/consumer
// loops before checking validity. Subsequent work is skipped only for a whole
// cluster whose rows all lie outside the expert's live prefix. Every warp and
// both CTAs of a 2SM cluster follow the same sequence, retaining all barriers.
template <class CtaShape, class ClusterShape>
class CountsStaticScheduler
    : public cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100 {
  using Base = cutlass::gemm::kernel::detail::StaticPersistentTileScheduler100;
 public:
  struct Arguments : Base::Arguments { int32_t const* expert_counts = nullptr; };
  struct Params : Base::Params { int32_t const* expert_counts = nullptr; };
  using WorkTileInfo = typename Base::WorkTileInfo;
  using CLCResponse = typename Base::CLCResponse;

  CUTLASS_DEVICE explicit
  CountsStaticScheduler(CLCResponse* response, Params const& params, dim3 cluster_id)
      : Base(response, static_cast<typename Base::Params const&>(params), cluster_id),
        expert_counts_(params.expert_counts) {}

  template <class ProblemShape, class TileShape, class AtomShape, class Cluster>
  static Params to_underlying_arguments(
      ProblemShape problem, TileShape tile, AtomShape atom, Cluster cluster,
      cutlass::KernelHardwareInfo const& hw, Arguments const& args,
      void* workspace = nullptr, uint32_t epilogue_subtile = 1) {
    Params params;
    static_cast<typename Base::Params&>(params) = Base::to_underlying_arguments(
        problem, tile, atom, cluster, hw,
        static_cast<typename Base::Arguments const&>(args), workspace, epilogue_subtile);
    params.expert_counts = args.expert_counts;
    return params;
  }

  template <class Pipeline, class State>
  CUTLASS_DEVICE auto fetch_next_work(WorkTileInfo work, Pipeline& pipeline, State state) {
    auto next = Base::fetch_next_work(work, pipeline, state);
    while (expert_counts_ && cute::get<0>(next).is_valid()) {
      auto const& candidate = cute::get<0>(next);
      constexpr int cluster_m = cute::size<0>(ClusterShape{});
      constexpr int cta_m = cute::size<0>(CtaShape{});
      int row_start = (candidate.M_idx / cluster_m) * cluster_m * cta_m;
      if (row_start < expert_counts_[candidate.L_idx]) break;
      next = Base::fetch_next_work(candidate, pipeline, state);
    }
    return next;
  }
 private:
  int32_t const* expert_counts_;
};
}  // namespace grain_moe_sm110

namespace cutlass::gemm::kernel::detail {
template <class TileShape, class ClusterShape, uint32_t Stages>
struct TileSchedulerSelector<cutlass::gemm::GrainMoeCountsScheduler,
                             arch::Sm100, TileShape, ClusterShape, Stages> {
  using Scheduler = grain_moe_sm110::CountsStaticScheduler<TileShape, ClusterShape>;
};
}  // namespace cutlass::gemm::kernel::detail

namespace grain_moe_sm110 {
using namespace cute;

template <int TileM, int TileN, bool TwoSm, bool Static = false, bool Counts = false> struct Blockwise {
  using Tile = Shape<Int<TileM>, Int<TileN>, _256>;
  using Cluster = Shape<Int<TwoSm ? 2 : 1>, _1, _1>;
  using Output = cutlass::bfloat16_t;
  using ScaleConfig = cutlass::detail::Sm100BlockwiseScaleConfig<
      1, 1, 256, UMMA::Major::K, UMMA::Major::MN>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());
  using Scalar = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
      float, Stride<_0, _0, _0>, 2, cutlass::multiplies>;
  using Multiply = cutlass::epilogue::fusion::Sm90Compute<
      cutlass::multiplies, Output, float,
      cutlass::FloatRoundStyle::round_to_nearest>;
  using EVT = cutlass::epilogue::fusion::Sm90EVT<
      Multiply, Scalar, cutlass::epilogue::fusion::Sm90AccFetch>;
  using EpilogueSchedule = cute::conditional_t<
      TwoSm, cutlass::epilogue::TmaWarpSpecialized2Sm,
      cutlass::epilogue::TmaWarpSpecialized1Sm>;
  using MainloopSchedule = cute::conditional_t<
      TwoSm, cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100,
      cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float, void,
      cutlass::layout::RowMajor, 8, Output, cutlass::layout::RowMajor, 8,
      EpilogueSchedule, EVT>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, int8_t,
      tuple<cutlass::layout::RowMajor, LayoutSFA>, 16, int8_t,
      tuple<cutlass::layout::ColumnMajor, LayoutSFB>, 16, int32_t, Tile,
      Cluster, cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue,
      cute::conditional_t<Counts, cutlass::gemm::GrainMoeCountsScheduler,
          cute::conditional_t<Static, cutlass::gemm::StaticPersistentScheduler, void>>>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

  static typename Gemm::Arguments arguments(
      void const *a, void const *b, void const *sa, void const *sb, void *out,
      int M, int N, int K, int E, int swizzle, int raster,
      int32_t const* counts = nullptr) {
    auto stride_a = cutlass::make_cute_packed_stride(
        typename Kernel::StrideA{}, make_shape(M, K, E));
    auto stride_b = cutlass::make_cute_packed_stride(
        typename Kernel::StrideB{}, make_shape(N, K, E));
    auto stride_d = cutlass::make_cute_packed_stride(
        typename Kernel::StrideD{}, make_shape(M, N, E));
    auto layout_sa = ScaleConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, E));
    auto layout_sb = ScaleConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, E));
    typename Gemm::Arguments args{
        E == 1 ? cutlass::gemm::GemmUniversalMode::kGemm
               : cutlass::gemm::GemmUniversalMode::kBatched,
        {M, N, K, E},
        {static_cast<int8_t const *>(a), stride_a,
         static_cast<int8_t const *>(b), stride_b,
         static_cast<float const *>(sa), layout_sa,
         static_cast<float const *>(sb), layout_sb},
        {{}, nullptr, stride_d, static_cast<Output *>(out), stride_d}};
    args.epilogue.thread = {{{1.f, 1.f}, {nullptr, nullptr}, {}}, {}, {}};
    if constexpr (Counts) args.scheduler.expert_counts = counts;
    args.scheduler.max_swizzle_size = swizzle;
    using Raster = cutlass::gemm::kernel::detail::RasterOrderOptions;
    args.scheduler.raster_order = raster == 1 ? Raster::AlongN
        : raster == 2 ? Raster::AlongM : Raster::Heuristic;
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      args.hw_info.device_id = device;
      static thread_local int cached_device = -1;
      static thread_local int cached_sms = 0;
      if (cached_device != device) {
        int sms = 0;
        if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) == cudaSuccess) {
          cached_device = device;
          cached_sms = sms;
        }
      }
      args.hw_info.sm_count = cached_sms;
    }
    return args;
  }

  static size_t workspace(int M, int N, int K, int E, int swizzle, int raster) {
    return Gemm::get_workspace_size(arguments(
        nullptr, nullptr, nullptr, nullptr, nullptr, M, N, K, E, swizzle, raster));
  }

  static int launch(void const *a, void const *b, void const *sa, void const *sb,
                    void *out, int M, int N, int K, int E, int swizzle, int raster,
                    int32_t const* counts, cudaStream_t stream, void *workspace, size_t bytes) {
    auto args = arguments(a, b, sa, sb, out, M, N, K, E, swizzle, raster, counts);
    Gemm gemm;
    auto status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return -1000 - int(status);
    auto required = Gemm::get_workspace_size(args);
    if ((required && !workspace) || bytes < required) return int(cudaErrorInvalidValue);
    status = gemm.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) return -1000 - int(status);
    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) return -1000 - int(status);
    return int(cudaGetLastError());
  }
};

inline bool valid(int M, int N, int K, int E, int swizzle, int raster) {
  return M > 0 && N > 0 && K > 0 && E > 0 && M % 64 == 0 &&
      N % 64 == 0 && K % 256 == 0 &&
      (swizzle == 1 || swizzle == 2 || swizzle == 4 || swizzle == 8) &&
      raster >= 0 && raster <= 2;
}
}  // namespace grain_moe_sm110

#define GRAIN_MOE_CONFIGS(OP) \
  OP(34, 128, 256, true, false, false)  \
  OP(35, 128, 128, true, false, false)  \
  OP(36, 256, 128, true, false, false)  \
  OP(37, 128, 256, false, false, false) \
  OP(38, 128, 128, false, false, false) \
  OP(41, 128, 256, true, true, false)   \
  OP(42, 128, 128, false, true, false) \
  OP(43, 128, 256, true, true, true)   \
  OP(44, 128, 128, false, true, true)  \
  OP(45, 256, 256, true, false, false)

extern "C" size_t grain_cutlass_moe_workspace_bytes(
    int M, int N, int K, int E, int config, int swizzle, int raster) {
  if (!grain_moe_sm110::valid(M, N, K, E, swizzle, raster)) return 0;
  switch (config) {
#define WORKSPACE(ID, TM, TN, TWO, STATIC, COUNTS) case ID: return grain_moe_sm110::Blockwise<TM,TN,TWO,STATIC,COUNTS>::workspace(M,N,K,E,swizzle,raster);
    GRAIN_MOE_CONFIGS(WORKSPACE)
#undef WORKSPACE
    default: return 0;
  }
}

extern "C" int grain_cutlass_moe_g256(
    void const *a, void const *b, void const *sa, void const *sb, void *out,
    int M, int N, int K, int E, int config, int swizzle, int raster,
    void const* counts, void *stream, void *workspace, size_t bytes) {
  if (!a || !b || !sa || !sb || !out ||
      !grain_moe_sm110::valid(M,N,K,E,swizzle,raster) ||
      reinterpret_cast<uintptr_t>(a) % 16 || reinterpret_cast<uintptr_t>(b) % 16 ||
      reinterpret_cast<uintptr_t>(sa) % 4 || reinterpret_cast<uintptr_t>(sb) % 16 ||
      reinterpret_cast<uintptr_t>(out) % 16 ||
      reinterpret_cast<uintptr_t>(counts) % 4) return int(cudaErrorInvalidValue);
  auto st = static_cast<cudaStream_t>(stream);
  switch (config) {
#define LAUNCH(ID, TM, TN, TWO, STATIC, COUNTS) case ID: return grain_moe_sm110::Blockwise<TM,TN,TWO,STATIC,COUNTS>::launch(a,b,sa,sb,out,M,N,K,E,swizzle,raster,static_cast<int32_t const*>(counts),st,workspace,bytes);
    GRAIN_MOE_CONFIGS(LAUNCH)
#undef LAUNCH
    default: return int(cudaErrorInvalidValue);
  }
}

extern "C" const char *grain_cutlass_moe_error_string(int code) {
  if (code <= -1000) return cutlassGetStatusString(cutlass::Status(-1000 - code));
  return cudaGetErrorString(cudaError_t(code));
}
