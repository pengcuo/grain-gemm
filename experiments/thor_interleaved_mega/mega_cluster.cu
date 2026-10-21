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
// Experimental single-card Thor INT8 Mega MoE. A persistent 2SM cluster owns
// an entire expert M128 tile: adjacent-pair G1 -> exact SwiGLU/G256 -> G2.
// Quantized hidden is cluster-local shared memory; scales are retained by
// quantizer registers and published into dead G1 A scratch before G2.
// No global intermediate activation/scales or cross-cluster ready queue.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/arch/simd_sm100.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cstdint>
#include <cstddef>


// Caller invariant: scale=RN(max(amax(abs(value)),1e-12f)/127),
// hence every finite value satisfies |value/scale|<128; finite scale and
// reciprocal are normal. Reuse one correctly-rounded reciprocal per G256 row.
// Three fp32 RN errors bound |approx-RN(value/scale)| below 2.29e-5.
// Guard a wider 1e-4 interval around every half-integer; only there (or for
// nonfinite approximate results) retain the original correctly-rounded divide.
__device__ __forceinline__ float grain_quant_nearest_integer_exact(
    float value, float scale, float reciprocal) {
  float approximate=__fmul_rn(value,reciprocal);
  float rounded=nearbyintf(approximate);
  float boundary_distance=fabsf(fabsf(approximate-rounded)-0.5f);
  if (!isfinite(approximate) || boundary_distance<=1.0e-4f)
    rounded=nearbyintf(__fdiv_rn(value,scale));
  return rounded;
}

#ifdef GRAIN_MOE_EXACT_SILU_LUT
#include "exact_silu_lut.cuh"
#endif

__device__ __forceinline__ float grain_moe_silu(float gate) {
#ifdef GRAIN_MOE_EXACT_SILU_LUT
  return grain_moe_silu_lookup::evaluate(gate);
#else
  return __fdiv_rn(gate,1.f+expf(-gate));
#endif
}

namespace grain_moe_mega_cluster {
using namespace cute;
constexpr int BM=64,BN=256,ClusterBM=128,BLocal=128,TmemN=128,P=2,ProducerThreads=128;
constexpr int MaxGroupsI=4;
using Mma=decltype(make_tiled_mma(SM100_MMA_S8_2x1SM_SS<int8_t,int8_t,int32_t,ClusterBM,BN,
                                  UMMA::Major::K,UMMA::Major::K>{}));
using LayoutA=decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                             partition_shape_A(Mma{},make_shape(_128{},_256{}))));
using LayoutB=decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                             partition_shape_B(Mma{},make_shape(_256{},_256{}))));
template<int CW> struct Storage {
  alignas(128) int8_t a[6][cosize_v<LayoutA>];
  alignas(128) int8_t b[P][cosize_v<LayoutB>];
  alignas(128) int8_t qh[MaxGroupsI][cosize_v<LayoutA>];
  alignas(16) uint64_t operand_full[P],operand_empty[P];
  alignas(16) uint64_t accumulator_full[4],accumulator_empty[4];
  alignas(16) uint64_t scale_full[P],scale_empty[P];
  alignas(16) float scale_a[P][BM],scale_b[P][BN];
  alignas(128) float maximum[BM];
  uint32_t tmem;
};
__device__ __forceinline__ uint32_t pack_bf16(float x,float y){
  union {__nv_bfloat162 bf;uint32_t u;} value;
  value.bf=__floats2bfloat162_rn(x,y);return value.u;
}
__device__ __forceinline__ float2 unpack_bf16(uint32_t bits){
  union {__nv_bfloat162 bf;uint32_t u;} value;
  value.u=bits;return __bfloat1622float2(value.bf);
}
__device__ __forceinline__ int token_for(int64_t assignment,int R){return R==4?int(assignment)>>2:int(assignment)/R;}
__device__ __forceinline__ int a_offset(int row,int k){
  return (k/128)*BM*128+row*128+((k%128)^((row&7)*16));
}

template<int CW>
__device__ __forceinline__ void promote(Storage<CW>& shared,int& serial,int groups,
                                        int row,int cw,int first_col,float* total){
  constexpr int Columns=BM*BN/(CW*32);
  CUTE_UNROLL
  for(int index=0;index<Columns;++index)total[index]=0.f;
  for(int kg=0;kg<groups;++kg,++serial){
    int stage=serial%P,buffer=serial%4;
    cutlass::arch::ClusterBarrier::wait(&shared.accumulator_full[buffer],(serial/4)&1);
    cutlass::arch::ClusterBarrier::wait(&shared.scale_full[stage],(serial/P)&1);
    float scale_a=shared.scale_a[stage][row];
    CUTE_UNROLL
    for(int sub=0;sub<Columns/8;++sub){
      uint32_t partial[8];
      uint32_t address=shared.tmem+buffer*TmemN+(first_col%(BN/2))+sub*8+
        (((cw%2)*32+(first_col/(BN/2))*64)<<16);
      SM100_TMEM_LOAD_32dp32b8x::copy(address,partial[0],partial[1],partial[2],partial[3],
                                   partial[4],partial[5],partial[6],partial[7]);
      cutlass::arch::fence_view_async_tmem_load();
      CUTE_UNROLL
      for(int q=0;q<2;++q){
        float scale_b[4];
        uint32_t address_b=cast_smem_ptr_to_uint(&shared.scale_b[stage][first_col+sub*8+q*4]);
        asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3},[%4];\n"
          :"=f"(scale_b[0]),"=f"(scale_b[1]),"=f"(scale_b[2]),"=f"(scale_b[3]):"r"(address_b):"memory");
        CUTE_UNROLL
        for(int j=0;j<4;j+=2){
          int index=sub*8+q*4+j;
          float2 scales,aa=make_float2(scale_a,scale_a),bb=make_float2(scale_b[j],scale_b[j+1]);
          cute::mul(scales,aa,bb);
          float2 pv=make_float2(float(int32_t(partial[q*4+j])),float(int32_t(partial[q*4+j+1])));
          float2 running=make_float2(total[index],total[index+1]);
          cute::fma(running,pv,scales,running);total[index]=running.x;total[index+1]=running.y;
        }
      }
    }
    __syncwarp();
    if((threadIdx.x%32)==0){
      cutlass::arch::ClusterBarrier::arrive(&shared.accumulator_empty[buffer],0,1);
      cutlass::arch::ClusterBarrier::arrive(&shared.scale_empty[stage]);
    }
  }
}

template<int CW,bool Down>
__device__ __forceinline__ void phase(Storage<CW>& shared,int& serial,
  int8_t const* X,float const* SX,float const* SB,int32_t const* OFFSETS,int64_t const* ORDER,
  __nv_bfloat16* Y,int8_t* QDBG,float* SDBG,
  int e,int tile_m,int count,int C,int I,int K,int R,
  CUtensorMap const& map_b,float (&resident_scales)[MaxGroupsI][BM/CW]){
  constexpr int ConsumerThreads=CW*32;
  constexpr int Columns=BM*BN/ConsumerThreads,Chunks=CW/2,NPerChunk=BN/Chunks;
  int warp=threadIdx.x/32,lane=threadIdx.x%32,peer=int(cute::block_rank_in_cluster());
  int groups=Down?I/256:K/256,ntiles=Down?K/256:I/128;
  if(warp==1&&lane==0){
    for(int nt=0;nt<ntiles;++nt){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P;
        if(serial>=P)cutlass::arch::ClusterBarrier::wait(&shared.operand_empty[stage],((serial/P)-1)&1);
        if(peer==0)cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(&shared.operand_full[stage],2*BLocal*256);
        int col=nt*256+peer*BLocal;
        SM100_TMA_2SM_LOAD_3D::copy(&map_b,&shared.operand_full[stage],0,shared.b[stage],kg*256,col,e);
        SM100_TMA_2SM_LOAD_3D::copy(&map_b,&shared.operand_full[stage],0,shared.b[stage]+BLocal*128,kg*256+128,col,e);
      }
    }
  }else if(warp==3){
    int start=OFFSETS[e],m0=tile_m*ClusterBM+peer*BM+lane,m1=m0+32;
    int token0=m0<count?token_for(ORDER[start+m0],R):0,token1=m1<count?token_for(ORDER[start+m1],R):0;
    for(int nt=0;nt<ntiles;++nt){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P;
        if(lane==0&&serial>=P)cutlass::arch::ClusterBarrier::wait(&shared.operand_empty[stage],((serial/P)-1)&1);
        __syncwarp();
        if constexpr(!Down){
          if(K!=1536||nt==0){
          CUTE_UNROLL
          for(int index=lane;index<BM*16;index+=32){
            int row=index/16,k=(index%16)*16,cached=row<32?token0:token1;
            int token=__shfl_sync(0xffffffff,cached,row%32),m=tile_m*ClusterBM+peer*BM+row;
            int valid=m<count?16:0;
            uint32_t dst=cast_smem_ptr_to_uint(shared.a[K==1536?kg:stage]+a_offset(row,k));
            auto src=X+token*K+kg*256+k;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(dst),"l"(src),"r"(valid):"memory");
          }
          asm volatile("cp.async.commit_group; cp.async.wait_group 0;":::"memory");
          asm volatile("fence.proxy.async.shared::cta;":::"memory");
          }
        }
        __syncwarp();
        if(lane==0)cutlass::arch::ClusterBarrier::arrive(&shared.operand_full[stage],0,1);
      }
    }
  }else if(warp==2){
    int route_start=OFFSETS[e],m0=tile_m*ClusterBM+peer*BM+lane,m1=m0+32;
    int token0=m0<count?token_for(ORDER[route_start+m0],R):0,token1=m1<count?token_for(ORDER[route_start+m1],R):0;
    for(int nt=0;nt<ntiles;++nt){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P;
        if(serial>=P)cutlass::arch::ClusterBarrier::wait(&shared.scale_empty[stage],((serial/P)-1)&1);
        CUTE_UNROLL
        for(int row=lane;row<BM;row+=32){
          int m=tile_m*ClusterBM+peer*BM+row;
          if constexpr(Down){shared.scale_a[stage][row]=reinterpret_cast<float*>(shared.a)[kg*BM+row];}
          else{int token=row<32?token0:token1;shared.scale_a[stage][row]=m<C&&m<count?SX[token*groups+kg]:0.f;}
        }
        int output_n=Down?K:2*I;
        CUTE_UNROLL
        for(int col=lane;col<BN;col+=32)shared.scale_b[stage][col]=SB[(e*groups+kg)*output_n+nt*256+col];
        __syncwarp();
        if(lane==0)cutlass::arch::ClusterBarrier::arrive(&shared.scale_full[stage]);
      }
    }
  }else if(warp==0&&peer==0){
    auto cta=Mma{}.get_slice(_0{});
    auto gC=make_tensor(make_gmem_ptr(static_cast<bfloat16_t*>(nullptr)),make_shape(_128{},_256{}),make_stride(_256{},_1{}));
    auto acc=cta.make_fragment_C(cta.partition_C(gC));Mma mma;
    for(int nt=0;nt<ntiles;++nt){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P,buffer=serial%4;
        cutlass::arch::ClusterBarrier::wait(&shared.operand_full[stage],(serial/P)&1);
        if(serial>=4)cutlass::arch::ClusterBarrier::wait(&shared.accumulator_empty[buffer],((serial/4)-1)&1);
        acc.data()=shared.tmem+buffer*TmemN;
        int8_t* a=Down?shared.qh[kg]:shared.a[K==1536?kg:stage];
        auto frag_a=cta.make_fragment_A(make_tensor(make_smem_ptr(a),LayoutA{}));
        auto frag_b=cta.make_fragment_B(make_tensor(make_smem_ptr(shared.b[stage]),LayoutB{}));
        mma.accumulate_=UMMA::ScaleOut::Zero;
        CUTE_UNROLL
        for(int kk=0;kk<size<2>(frag_a);++kk){gemm(mma,frag_a(_,_,kk),frag_b(_,_,kk),acc);mma.accumulate_=UMMA::ScaleOut::One;}
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared.operand_empty[stage],uint16_t(3));
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared.accumulator_full[buffer],uint16_t(3));
      }
    }
  }else if(threadIdx.x>=ProducerThreads){
    int consumer=threadIdx.x-ProducerThreads,cw=consumer/32,row=(cw%2)*32+lane;
    int first_col=((cw%4)/2)*(BN/2)+(cw/4)*NPerChunk;
    int m=tile_m*ClusterBM+peer*BM+row;float total[Columns];
    float first_hidden[Columns/2];
    for(int nt=0;nt<ntiles;++nt){
      promote<CW>(shared,serial,groups,row,cw,first_col,total);
      if constexpr(Down){
        // G1's six cached A groups are dead. The first1KiB holds SH, so
        // reserve A+32KiB for a 32KiB BF16 output tile. Row-dependent 16-byte
        // XOR stripes let the register-fragment writers use all shared banks.
        auto scratch=reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<int8_t*>(shared.a)+32768);
        CUTE_UNROLL
        for(int j=0;j<Columns;j+=8){
          uint4 bits;
          bits.x=pack_bf16(m<count?total[j]:0.f,m<count?total[j+1]:0.f);
          bits.y=pack_bf16(m<count?total[j+2]:0.f,m<count?total[j+3]:0.f);
          bits.z=pack_bf16(m<count?total[j+4]:0.f,m<count?total[j+5]:0.f);
          bits.w=pack_bf16(m<count?total[j+6]:0.f,m<count?total[j+7]:0.f);
          int offset=row*BN+((first_col+j)^((row&7)*8));
          *reinterpret_cast<uint4*>(scratch+offset)=bits;
        }
        asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
        // In this mapping every warp writes one complete contiguous row.
        for(int v=consumer;v<BM*BN/8;v+=ConsumerThreads){
          int qr=v/(BN/8),col=(v%(BN/8))*8;
          int qm=tile_m*ClusterBM+peer*BM+qr;
          if(qm<C){
            int source=qr*BN+(col^((qr&7)*8));
            uint4 bits=*reinterpret_cast<uint4 const*>(scratch+source);
            *reinterpret_cast<uint4*>(Y+(int64_t(e)*C+qm)*K+nt*BN+col)=bits;
          }
        }
        asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
      }else{
        int half=nt%2,gi=nt/2;
        CUTE_UNROLL
        for(int j=0;j<Columns/2;j+=4){
          float2 gu0=unpack_bf16(pack_bf16(total[2*j],total[2*j+1]));
          float2 gu1=unpack_bf16(pack_bf16(total[2*j+2],total[2*j+3]));
          float2 gu2=unpack_bf16(pack_bf16(total[2*j+4],total[2*j+5]));
          float2 gu3=unpack_bf16(pack_bf16(total[2*j+6],total[2*j+7]));
          float h0=m<count?grain_moe_silu(gu0.x)*gu0.y:0.f;
          float h1=m<count?grain_moe_silu(gu1.x)*gu1.y:0.f;
          float h2=m<count?grain_moe_silu(gu2.x)*gu2.y:0.f;
          float h3=m<count?grain_moe_silu(gu3.x)*gu3.y:0.f;
          if(half==0){first_hidden[j]=h0;first_hidden[j+1]=h1;first_hidden[j+2]=h2;first_hidden[j+3]=h3;}
          else{total[j]=h0;total[j+1]=h1;total[j+2]=h2;total[j+3]=h3;}
        }
        if(half==1){
          // Each lane owns one row, each consumer warp one N stripe. Shared
          // positive-float atomicMax reduces the stripes with only 256 bytes.
          if(first_col==0)shared.maximum[row]=1.e-12f;
          asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
          float maximum=1.e-12f;
          CUTE_UNROLL
          for(int j=0;j<Columns/2;++j){maximum=fmaxf(maximum,fabsf(first_hidden[j]));maximum=fmaxf(maximum,fabsf(total[j]));}
          atomicMax(reinterpret_cast<unsigned int*>(shared.maximum+row),__float_as_uint(maximum));
          asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
          float scale=__fdiv_rn(shared.maximum[row],127.f);
          float reciprocal=__frcp_rn(scale);
          CUTE_UNROLL
          for(int j=0;j<Columns/2;j+=4){
            uint32_t pack0=0,pack1=0;
            CUTE_UNROLL
            for(int q=0;q<4;++q){
              int v0=int(fminf(127.f,fmaxf(-127.f,grain_quant_nearest_integer_exact(first_hidden[j+q],scale,reciprocal))));
              int v1=int(fminf(127.f,fmaxf(-127.f,grain_quant_nearest_integer_exact(total[j+q],scale,reciprocal))));
              pack0|=uint32_t(uint8_t(v0))<<(q*8);pack1|=uint32_t(uint8_t(v1))<<(q*8);
              int col=first_col/2+j+q;
              if(QDBG&&m<C){QDBG[(int64_t(e)*C+m)*I+gi*256+col]=int8_t(v0);QDBG[(int64_t(e)*C+m)*I+gi*256+128+col]=int8_t(v1);}
            }
            int col=first_col/2+j;
            *reinterpret_cast<uint32_t*>(shared.qh[gi]+a_offset(row,col))=pack0;
            *reinterpret_cast<uint32_t*>(shared.qh[gi]+a_offset(row,128+col))=pack1;
          }
          if(first_col==0){
            if(gi==0)resident_scales[0][0]=scale;
            if(gi==1)resident_scales[1][0]=scale;
            if(gi==2)resident_scales[2][0]=scale;
            if(gi==3)resident_scales[3][0]=scale;
            if(SDBG&&m<C)SDBG[(e*C+m)*(I/256)+gi]=scale;
          }
          asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
        }
      }
    }
  }
}

template<int CW>
__global__ __launch_bounds__(ProducerThreads+CW*32,1) void kernel(
  int8_t const* X,float const* SX,float const* SB1,float const* SB2,int32_t const* COUNTS,
  int32_t const* OFFSETS,int64_t const* ORDER,__nv_bfloat16* Y,int8_t* QDBG,float* SDBG,
  int C,int I,int K,int E,int T,int R,
  const __grid_constant__ CUtensorMap map_b1,const __grid_constant__ CUtensorMap map_b2){
  static_assert(CW==8||CW==16);
  extern __shared__ __align__(128) char raw[];auto& shared=*reinterpret_cast<Storage<CW>*>(raw);
  int warp=threadIdx.x/32,lane=threadIdx.x%32,peer=int(cute::block_rank_in_cluster());
  int cluster_id=int(blockIdx.x)/2,cluster_count=int(gridDim.x)/2,tiles_m=(C+127)/128;
  __syncthreads();TMEM::Allocator2Sm allocator;
  if(warp==0)allocator.allocate(512,&shared.tmem);
  if(threadIdx.x==0){
    for(int s=0;s<P;++s){
      cutlass::arch::ClusterBarrier::init(&shared.operand_full[s],3);
      cutlass::arch::ClusterBarrier::init(&shared.operand_empty[s],1);
      cutlass::arch::ClusterBarrier::init(&shared.scale_full[s],1);
      cutlass::arch::ClusterBarrier::init(&shared.scale_empty[s],CW);
    }
    for(int s=0;s<4;++s){
      cutlass::arch::ClusterBarrier::init(&shared.accumulator_full[s],1);
      cutlass::arch::ClusterBarrier::init(&shared.accumulator_empty[s],2*CW);
    }
    cutlass::arch::fence_barrier_init();
  }
  __syncthreads();cute::cluster_sync();if(warp==0)allocator.release_allocation_lock();
  int serial=0;
  float resident_scales[MaxGroupsI][BM/CW];
  for(int work=cluster_id;work<E*tiles_m;work+=cluster_count){
    int e=work/tiles_m,tm=work%tiles_m,count=COUNTS[e];
    if(tm*ClusterBM>=count){
      if(threadIdx.x>=ProducerThreads){
        int consumer=threadIdx.x-ProducerThreads;
        for(int index=consumer;index<BM*K;index+=CW*32){int row=index/K,col=index%K,m=tm*ClusterBM+peer*BM+row;
          if(m<C)Y[(int64_t(e)*C+m)*K+col]=__float2bfloat16_rn(0.f);}
        if(QDBG)for(int index=consumer;index<BM*I;index+=CW*32){int row=index/I,col=index%I,m=tm*ClusterBM+peer*BM+row;if(m<C)QDBG[(int64_t(e)*C+m)*I+col]=0;}
        if(SDBG)for(int index=consumer;index<BM*(I/256);index+=CW*32){int row=index/(I/256),col=index%(I/256),m=tm*ClusterBM+peer*BM+row;if(m<C)SDBG[(e*C+m)*(I/256)+col]=__fdiv_rn(1.e-12f,127.f);}
      }
      continue;
    }
    phase<CW,false>(shared,serial,X,SX,SB1,OFFSETS,ORDER,Y,QDBG,SDBG,e,tm,count,C,I,K,R,map_b1,resident_scales);
    __syncthreads();cute::cluster_sync();
    if(threadIdx.x>=ProducerThreads){
      int cw=(threadIdx.x-ProducerThreads)/32;
      int first_col=((cw%4)/2)*(BN/2)+(cw/4)*(BN/(CW/2));
      int row=(cw%2)*32+lane;
      if(first_col==0){
        CUTE_UNROLL
        for(int gi=0;gi<MaxGroupsI;++gi)if(gi<I/256)
          reinterpret_cast<float*>(shared.a)[gi*BM+row]=resident_scales[gi][0];
      }
    }
    __syncthreads();asm volatile("fence.proxy.async.shared::cta;":::"memory");cute::cluster_sync();
    phase<CW,true>(shared,serial,X,SX,SB2,OFFSETS,ORDER,Y,QDBG,SDBG,e,tm,count,C,I,K,R,map_b2,resident_scales);
    __syncthreads();cute::cluster_sync();
  }
  __syncthreads();cute::cluster_sync();if(warp==0)allocator.free(shared.tmem,512);
}
} // namespace grain_moe_mega_cluster

extern "C" size_t grain_moe_mega_cluster_smem_bytes(int cw){
  using namespace grain_moe_mega_cluster;return cw==8?sizeof(Storage<8>):cw==16?sizeof(Storage<16>):0;
}
extern "C" int grain_moe_mega_cluster_launch(
  void const* a,void const* sa,void const* b1,void const* sb1,void const* b2,void const* sb2,
  void const* counts,void const* offsets,void const* order,void* y,void* qdebug,void* sdebug,
  int C,int I,int K,int E,int T,int R,int cw,int requested_blocks,void* stream){
  using namespace grain_moe_mega_cluster;
  if(!a||!sa||!b1||!sb1||!b2||!sb2||!counts||!offsets||!order||!y||T<=0||R<=0||R>E||
      C<=0||I<=0||I>MaxGroupsI*256||K<=0||E<=0||C%64||I%256||K%256||(cw!=8&&cw!=16))return int(cudaErrorInvalidValue);
  CUtensorMap maps[2];
  for(int projection=0;projection<2;++projection){
    int inner=projection?I:K,outer=projection?K:2*I;
    uint64_t dims[3]={uint64_t(inner),uint64_t(outer),uint64_t(E)};
    uint64_t strides[2]={uint64_t(inner),uint64_t(inner)*outer};
    uint32_t box[3]={128,128,1},element[3]={1,1,1};
    CUresult encoded=cuTensorMapEncodeTiled(maps+projection,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,
      const_cast<void*>(projection?b2:b1),dims,strides,box,element,CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(encoded!=CUDA_SUCCESS)return -10000-int(encoded);
  }
  int work_count=E*((C+127)/128),blocks=requested_blocks;cudaError_t status;
  if(blocks<=0){int device;status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
    status=cudaDeviceGetAttribute(&blocks,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return int(status);}
  int clusters=blocks/2+blocks%2;clusters=clusters<work_count?clusters:work_count;blocks=2*clusters;
  cudaLaunchAttribute attribute{};attribute.id=cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x=2;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
  cudaLaunchConfig_t config{};config.gridDim=dim3(blocks,1,1);config.stream=static_cast<cudaStream_t>(stream);
  config.attrs=&attribute;config.numAttrs=1;
  if(cw==16){auto fn=kernel<16>;status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage<16>));
    if(status!=cudaSuccess)return int(status);config.blockDim=dim3(ProducerThreads+16*32,1,1);config.dynamicSmemBytes=sizeof(Storage<16>);
    status=cudaLaunchKernelEx(&config,fn,static_cast<int8_t const*>(a),static_cast<float const*>(sa),
      static_cast<float const*>(sb1),static_cast<float const*>(sb2),static_cast<int32_t const*>(counts),
      static_cast<int32_t const*>(offsets),static_cast<int64_t const*>(order),static_cast<__nv_bfloat16*>(y),
      static_cast<int8_t*>(qdebug),static_cast<float*>(sdebug),C,I,K,E,T,R,maps[0],maps[1]);
  }else{auto fn=kernel<8>;status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage<8>));
    if(status!=cudaSuccess)return int(status);config.blockDim=dim3(ProducerThreads+8*32,1,1);config.dynamicSmemBytes=sizeof(Storage<8>);
    status=cudaLaunchKernelEx(&config,fn,static_cast<int8_t const*>(a),static_cast<float const*>(sa),
      static_cast<float const*>(sb1),static_cast<float const*>(sb2),static_cast<int32_t const*>(counts),
      static_cast<int32_t const*>(offsets),static_cast<int64_t const*>(order),static_cast<__nv_bfloat16*>(y),
      static_cast<int8_t*>(qdebug),static_cast<float*>(sdebug),C,I,K,E,T,R,maps[0],maps[1]);
  }
  return int(status==cudaSuccess?cudaGetLastError():status);
}
extern "C" const char* grain_moe_mega_cluster_error_string(int code){
  if(code<=-10000){const char* msg=nullptr;cuGetErrorString(CUresult(-10000-code),&msg);return msg?msg:"TMA encode error";}
  return cudaGetErrorString(cudaError_t(code));
}
