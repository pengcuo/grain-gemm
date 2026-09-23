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
// Small extension of CUTLASS's standard SM70/SM80 GemmUniversal composition.
// Its original shell creates an MMA-typed (INT32) accumulator and does not
// construct the collective with Params. G256 needs an FP32 running sum and
// scale pointers, so only the device composition operator is specialized here.
// Argument lowering, workspace, launch grid and adapter remain upstream code.
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/sm70_gemm.hpp"
#include "collective.hpp"

namespace grain_cutlass {
template<class Mainloop>
struct StandardKernelMainloop : Mainloop {
  using DispatchPolicy = cutlass::gemm::MainloopSm80CpAsyncUnpredicated<Mainloop::DispatchPolicy::Stages>;
};
} // namespace grain_cutlass

namespace cutlass::gemm::kernel {
template<class ProblemShape_, class Mainloop_, class Epilogue_, class Scheduler_>
class GemmUniversal<ProblemShape_, Mainloop_, Epilogue_, Scheduler_,
    cute::enable_if_t<cute::is_same_v<typename Mainloop_::DispatchPolicy::Schedule,
                                    grain_cutlass::KernelG256Multistage>>>
  : public GemmUniversal<ProblemShape_, grain_cutlass::StandardKernelMainloop<Mainloop_>,
                         Epilogue_, Scheduler_> {
  using Base = GemmUniversal<ProblemShape_, grain_cutlass::StandardKernelMainloop<Mainloop_>,
                            Epilogue_, Scheduler_>;
public:
  using CollectiveMainloop = Mainloop_;
  using DispatchPolicy = typename Mainloop_::DispatchPolicy;
  using Params = typename Base::Params;
  CUTLASS_DEVICE void operator()(Params const& params, char* smem_buf) {
    using namespace cute;
    using X = Underscore;
    auto shape_mnkl = append<4>(params.problem_shape, Int<1>{});
    auto [M, N, K, L] = shape_mnkl;
    auto tile = typename Base::TileShape{};
    int m_coord = int(blockIdx.x);
    int n_coord = int(blockIdx.y);
    // Match the existing CuTe traversal: K > 4096 reuses an eight-CTA M stripe.
    if (K > 4096) {
      int tiles_m = M / int(size<0>(tile));
      int tiles_n = N / int(size<1>(tile));
      int pid = m_coord + tiles_m * n_coord;
      int first_m = (pid / (8 * tiles_n)) * 8;
      int group_m = min(8, tiles_m - first_m);
      m_coord = first_m + pid % group_m;
      n_coord = (pid % (8 * tiles_n)) / group_m;
    }
    int l_coord = int(blockIdx.z);
    auto coord = make_coord(m_coord, n_coord, _, l_coord);
    Tensor mA = make_tensor(make_gmem_ptr(params.mainloop.ptr_A), make_shape(M,K,L), params.mainloop.dA);
    Tensor mB = make_tensor(make_gmem_ptr(params.mainloop.ptr_B), make_shape(N,K,L), params.mainloop.dB);
    Tensor gA = local_tile(mA(_,_,l_coord), tile, take<0,3>(coord), Step<_1,X,_1>{});
    Tensor gB = local_tile(mB(_,_,l_coord), tile, take<0,3>(coord), Step<X,_1,_1>{});
    auto residue = make_tuple(M - size<0>(gA) * m_coord,
                              N - size<0>(gB) * n_coord,
                              K - size<1>(gA) * size<2>(gA));
    typename Base::TiledMma mma;
    Tensor int_fragment = partition_fragment_C(mma, take<0,2>(tile));
    Tensor accum = make_fragment_like<float>(int_fragment);
    clear(accum);
    CollectiveMainloop collective(params.mainloop, m_coord, n_coord);
    collective(accum, gA, gB, accum,
               make_coord_iterator(shape<2>(gA)), int(size<2>(gA)),
               residue, int(threadIdx.x), smem_buf);
    Epilogue_ epilogue(params.epilogue);
    epilogue(shape_mnkl, tile, coord, accum, mma, residue, int(threadIdx.x), smem_buf);
  }
};
} // namespace cutlass::gemm::kernel
