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
// Adapted from CUTLASS examples/cute/tutorial/sgemm_sm80.cu at
// commit 098de2a652cf8f00fd70b2df54051c7eccbb855a (https://github.com/NVIDIA/cutlass).
// Changes: signed INT8 operands, K-group INT32-to-FP32 scaling, BF16 output,
// row-major output, native C launcher and multiple CTA configurations.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cstdint>
template <class ElementA,
          class ElementB,
          class SmemLayoutA,
          class SmemLayoutB>
struct SharedStorage
{
  cute::ArrayEngine<ElementA, cute::cosize_v<SmemLayoutA>> A;
  cute::ArrayEngine<ElementB, cute::cosize_v<SmemLayoutB>> B;
};

template <bool Grouped, class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA, class S2RAtomA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB, class S2RAtomB,
          class TC, class CStride, class TiledMma>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a, S2RAtomA s2r_atom_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b, S2RAtomB s2r_atom_b,
            TC      * C, CStride dC, TiledMma mma,
            float const* scale_a, float const* scale_b)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  int tile_m, tile_n;
  if constexpr (Grouped) {
    // Limit the active A-row stripe to eight CTAs for reuse through L2.
    int blocks_m = int(get<0>(shape_MNK)) / int(size<0>(cta_tiler));
    int blocks_n = int(get<1>(shape_MNK)) / int(size<1>(cta_tiler));
    int pid = int(blockIdx.x);
    int group_id = pid / (8 * blocks_n);
    int first_m = group_id * 8;
    int group_m = min(8, blocks_m - first_m);
    tile_m = first_m + pid % group_m;
    tile_n = (pid % (8 * blocks_n)) / group_m;
  } else {
    tile_m = int(blockIdx.x);
    tile_n = int(blockIdx.y);
  }
  auto cta_coord = make_coord(tile_m, tile_n, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  extern __shared__ char shared_memory[];
  using SharedStorage = SharedStorage<TA, TB, ASmemLayout, BSmemLayout>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), sA_layout);   // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), sB_layout);   // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles across the threads
  //

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K,PIPE)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K,PIPE)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<3>(tAsA);

  // Total count of tiles
  int k_tile_count = size<3>(tAgA);
  // Current tile index in gmem to read from
  int k_tile_next = 0;

  // Start async loads for all pipes but the last
  CUTE_UNROLL
  for (int k_pipe = 0; k_pipe < K_PIPE_MAX-1; ++k_pipe) {
    copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,k_pipe));
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,k_pipe));
    cp_async_fence();
    --k_tile_count;
    if (k_tile_count > 0) { ++k_tile_next; }
  }

  //
  // Define A/B partitioning and C accumulators
  //

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate registers for pipelining
  Tensor tCrA = thr_mma.partition_fragment_A(sA(_,_,0));               // (MMA,MMA_M,MMA_K)
  Tensor tCrB = thr_mma.partition_fragment_B(sB(_,_,0));               // (MMA,MMA_N,MMA_K)
  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V((  shape(tCrC) == take<0,3>(shape(tCgC))));     // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V((size<1>(tCgC) == size<1>(tCrA)));              // MMA_M
  CUTE_STATIC_ASSERT_V((size<2>(tCgC) == size<1>(tCrB)));              // MMA_N

  // Clear the accumulators
  clear(tCrC);
  Tensor tCrF = make_fragment_like<float>(tCrC);
  clear(tCrF);
  auto cC = make_identity_tensor(shape(gC));
  auto tCcC = thr_mma.partition_C(cC);
  int processed_tiles = 0;
  int scale_group = 0;

  //
  // Copy Atom retiling
  //

  TiledCopy s2r_copy_a = make_tiled_copy_A(s2r_atom_a, mma);
  ThrCopy   s2r_thr_copy_a = s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);                        // (CPY,MMA_M,MMA_K,PIPE)
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);                         // (CPY,MMA_M,MMA_K)

  TiledCopy s2r_copy_b = make_tiled_copy_B(s2r_atom_b, mma);
  ThrCopy   s2r_thr_copy_b = s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);                        // (CPY,MMA_N,MMA_K,PIPE)
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);                         // (CPY,MMA_N,MMA_K)





  // Current pipe index in smem to read from
  int smem_pipe_read  = 0;
  // Current pipe index in smem to write to
  int smem_pipe_write = K_PIPE_MAX-1;

  // Pipe slice
  Tensor tXsA_p = tXsA(_,_,_,smem_pipe_read);
  Tensor tXsB_p = tXsB(_,_,_,smem_pipe_read);

  // Size of the register pipeline
  auto K_BLOCK_MAX = size<2>(tCrA);
  CUTE_STATIC_ASSERT_V(K_BLOCK_MAX == size<2>(tXrA));

  // PREFETCH register pipeline
  if (K_BLOCK_MAX > 1) {
    // Wait until our first prefetched tile is loaded in
    cp_async_wait<K_PIPE_MAX-2>();
    __syncthreads();

    // Prefetch the first rmem from the first k-tile
    copy(s2r_atom_a, tXsA_p(_,_,Int<0>{}), tXrA(_,_,Int<0>{}));
    copy(s2r_atom_b, tXsB_p(_,_,Int<0>{}), tXrB(_,_,Int<0>{}));
  }

  //
  // PIPELINED MAIN LOOP
  // TUTORIAL: Example of a gemm loop that pipelines shared memory using SM80's cp.async instructions
  //           and explicit pipelines in shared memory.
  //   Data is read from global(k_tile_next) to shared(smem_pipe_write).
  //   Data is read from shared(smem_pipe_read) to registers(k_block_next).
  //   Data is computed on registers(b_block).
  //
  //   This allows all copies and compute to overlap:
  //     Copy from gmem->smem can overlap with copies from smem->rmem and compute on rmem.
  //     Copy from smem->rmem can overlap with compute on rmem.
  //

  CUTE_NO_UNROLL
  while (k_tile_count > -(K_PIPE_MAX-1))
  {
    CUTE_UNROLL
    for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block)
    {
      if (k_block == K_BLOCK_MAX - 1)
      {
        // Slice the smem_pipe_read smem
        tXsA_p = tXsA(_,_,_,smem_pipe_read);
        tXsB_p = tXsB(_,_,_,smem_pipe_read);

        // Commit the smem for smem_pipe_read
        cp_async_wait<K_PIPE_MAX-2>();
        __syncthreads();
      }

      // Load A, B shmem->regs for k_block+1
      auto k_block_next = (k_block + Int<1>{}) % K_BLOCK_MAX;      // static
      copy(s2r_atom_a, tXsA_p(_,_,k_block_next), tXrA(_,_,k_block_next));
      copy(s2r_atom_b, tXsB_p(_,_,k_block_next), tXrB(_,_,k_block_next));
      // Copy gmem to smem before computing gemm on each k-pipe
      if (k_block == 0)
      {
        copy(copy_a, tAgA(_,_,_,k_tile_next), tAsA(_,_,_,smem_pipe_write));
        copy(copy_b, tBgB(_,_,_,k_tile_next), tBsB(_,_,_,smem_pipe_write));
        cp_async_fence();

        // Advance the gmem tile
        --k_tile_count;
        if (k_tile_count > 0) { ++k_tile_next; }

        // Advance the smem pipe
        smem_pipe_write = smem_pipe_read;
        smem_pipe_read = (smem_pipe_read == K_PIPE_MAX-1) ? 0 : smem_pipe_read+1;
      }
      // Thread-level register gemm for k_block
      gemm(mma, tCrA(_,_,k_block), tCrB(_,_,k_block), tCrC);
    }
    ++processed_tiles;
    if (processed_tiles % 2 == 0) {
      CUTE_UNROLL
      for (int i = 0; i < size(tCrC); ++i) {
        auto coord = tCcC(i);
        int m = tile_m * int(size<0>(cta_tiler)) + int(get<0>(coord));
        int n = tile_n * int(size<1>(cta_tiler)) + int(get<1>(coord));
        float sa = scale_a[m * (int(get<2>(shape_MNK)) / 256) + scale_group];
        float sb = scale_b[scale_group * int(get<1>(shape_MNK)) + n];
        tCrF(i) += float(tCrC(i)) * sa * sb;
      }
      clear(tCrC);
      ++scale_group;
    }

  }


  //
  // Epilogue
  //

  CUTE_UNROLL
  for (int i = 0; i < size(tCrF); ++i) tCgC(i) = TC(tCrF(i));
}


template<int BM, int BN, int P, int WM, int WN, bool Grouped>
int launch(void const* a, void const* b, void const* sa, void const* sb, void* c,
           int M, int N, int K, cudaStream_t stream) {
  using namespace cute;
  if (M <= 0 || N <= 0 || K <= 0 || M % BM || N % BN || K % 256) return int(cudaErrorInvalidValue);
  auto shape_mnk=make_shape(M,N,K);
  auto cta=make_shape(Int<BM>{},Int<BN>{},_128{});
  auto dA=make_stride(K,_1{});
  auto dB=make_stride(K,_1{});
  auto dC=make_stride(N,_1{});
  auto swizzle_atom=composition(Swizzle<3,4,3>{}, Layout<Shape<_8,_128>,Stride<_128,_1>>{});
  auto sA=tile_to_shape(swizzle_atom,make_shape(Int<BM>{},_128{},Int<P>{}));
  auto sB=tile_to_shape(swizzle_atom,make_shape(Int<BN>{},_128{},Int<P>{}));
  auto copyA=make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>,int8_t>{},
    Layout<Shape<Int<WM*WN*4>,_8>,Stride<_8,_1>>{}, Layout<Shape<_1,_16>>{});
  auto copyB=copyA;
  auto mma=make_tiled_mma(SM80_16x8x32_S32S8S8S32_TN{},
    Layout<Shape<Int<WM>,Int<WN>>>{}, Tile<Int<WM*16>,Int<WN*16>,_32>{});
  Copy_Atom<SM75_U32x4_LDSM_N,int8_t> s2r;
  using output_t=cute::bfloat16_t;
  auto kernel=gemm_device<Grouped,decltype(shape_mnk),decltype(cta),
    int8_t,decltype(dA),decltype(sA),decltype(copyA),decltype(s2r),
    int8_t,decltype(dB),decltype(sB),decltype(copyB),decltype(s2r),
    output_t,decltype(dC),decltype(mma)>;
  int smem=sizeof(SharedStorage<int8_t,int8_t,decltype(sA),decltype(sB)>);
  cudaError_t status=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
  if(status!=cudaSuccess) return int(status);
  dim3 grid = Grouped ? dim3((M/BM)*(N/BN)) : dim3(M/BM,N/BN);
  kernel<<<grid,32*WM*WN,smem,stream>>>(shape_mnk,cta,
    static_cast<int8_t const*>(a),dA,sA,copyA,s2r,
    static_cast<int8_t const*>(b),dB,sB,copyB,s2r,
    static_cast<output_t*>(c),dC,mma,
    static_cast<float const*>(sa),static_cast<float const*>(sb));
  return int(cudaGetLastError());
}
// Both traversal variants are compiled independently. Small K uses the simpler
// 2D grid; larger K uses a grouped grid to reduce repeated large operand reads.
template<int BM, int BN, int P, int WM, int WN>
int dispatch(void const* a, void const* b, void const* sa, void const* sb, void* c,
             int M, int N, int K, cudaStream_t stream) {
  if (K <= 4096) {
    return launch<BM,BN,P,WM,WN,false>(a,b,sa,sb,c,M,N,K,stream);
  }
  return launch<BM,BN,P,WM,WN,true>(a,b,sa,sb,c,M,N,K,stream);
}

extern "C" int grain_cute_g256(void const* a, void const* b, void const* sa,
  void const* sb,void* c,int M,int N,int K,int config,void* stream) {
  if (!a || !b || !sa || !sb || !c) return int(cudaErrorInvalidValue);
  cudaStream_t st=static_cast<cudaStream_t>(stream);
  switch(config) {
    case 0:return dispatch<64,64,2,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 1:return dispatch<64,64,3,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 2:return dispatch<128,64,2,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 3:return dispatch<128,64,3,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 4:return dispatch<64,128,2,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 5:return dispatch<64,128,3,2,2>(a,b,sa,sb,c,M,N,K,st);
    case 6:return dispatch<128,128,2,4,2>(a,b,sa,sb,c,M,N,K,st);
    case 7:return dispatch<128,128,3,4,2>(a,b,sa,sb,c,M,N,K,st);
    default:return int(cudaErrorInvalidValue);
  }
}

extern "C" char const* grain_cuda_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
