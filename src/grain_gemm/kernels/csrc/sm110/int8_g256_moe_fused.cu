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
// Single-card Thor G256 INT8 fused gate/up + exact SwiGLU quantization.
// CuTe UMMA/TMA/TMEM primitives adapted from the existing grain Thor CuTe pipeline
// and CUTLASS tutorials; persistent work/epilogue overlap inspired by DeepInfer
// Mega MoE. Two-SM paired projections share input A; H1536 also caches A across
// the two I128 halves. Each projection preserves its BF16 rounding. SwiGLU stays
// FP32 in shared memory until G256 quantization, without a complete gate/up
// workspace in global memory. Public weight and scale layouts are unchanged.
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

namespace grain_moe_fused {
using namespace cute;
constexpr int BM=64, BN=128, ClusterBM=128, BLocal=64, TmemN=64, P=2, ProducerThreads=128;
using Mma = decltype(make_tiled_mma(SM100_MMA_S8_2x1SM_SS<int8_t,int8_t,int32_t,ClusterBM,BN,
                                  UMMA::Major::K,UMMA::Major::K>{}));
using LayoutA = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                             partition_shape_A(Mma{},make_shape(_128{},_256{}))));
using LayoutB = decltype(UMMA::tile_to_mma_shape(UMMA::Layout_K_SW128_Atom<int8_t>{},
                             partition_shape_B(Mma{},make_shape(_128{},_256{}))));

template<int CW> struct Storage {
  // H1536 has six G256 blocks. Keep its full M64 input in shared so the
  // second I128 half can reuse A without another global TMA load. Other K
  // values keep using the original two cyclic slots of this same allocation.
  alignas(128) int8_t a[6][cosize_v<LayoutA>];
  alignas(128) int8_t b[P][2][cosize_v<LayoutB>];
  alignas(16) uint64_t operand_full[P],operand_empty[P];
  alignas(16) uint64_t accumulator_full[4],accumulator_empty[4];
  alignas(16) uint64_t scale_full[P],scale_empty[P];
  alignas(16) float scale_a[P][BM],scale_b[P][2][BN];
  alignas(128) float hidden[BM][256];
  uint32_t tmem;
};

struct Tile { int m,i,e,count; };
__device__ __forceinline__ Tile decode(int work,int tiles_m,int groups_i,int32_t const* counts){
  Tile tile;
  tile.e=work/(tiles_m*groups_i);
  // Eight M tiles per I-band, including the final partial band.
  int local=work%(tiles_m*groups_i);
  int first=(local/(8*groups_i))*8;
  int band=min(8,tiles_m-first),within=local-first*groups_i;
  tile.m=first+within%band;
  tile.i=within/band;
  tile.count=counts[tile.e];
  return tile;
}

__device__ __forceinline__ uint32_t pack_bf16(float x,float y){
  union { __nv_bfloat162 bf; uint32_t u; } value;
  value.bf=__floats2bfloat162_rn(x,y);
  return value.u;
}
__device__ __forceinline__ float2 unpack_bf16(uint32_t bits){
  union { __nv_bfloat162 bf; uint32_t u; } value;
  value.u=bits;
  return __bfloat1622float2(value.bf);
}

template<int CW>
__global__ __launch_bounds__(ProducerThreads+CW*32,1) void kernel(
  float const* SA,float const* SB,int32_t const* COUNTS,int8_t* Q,float* S,
  int C,int I,int K,int E,
  const __grid_constant__ CUtensorMap map_a,const __grid_constant__ CUtensorMap map_b){
  static_assert(CW==8||CW==16);
  constexpr int ConsumerThreads=CW*32,Threads=ProducerThreads+ConsumerThreads;
  constexpr int Columns=BM*BN/ConsumerThreads,Chunks=CW/2,NPerChunk=BN/Chunks;
  extern __shared__ __align__(128) char raw[];
  auto& shared=*reinterpret_cast<Storage<CW>*>(raw);
  int warp=threadIdx.x/32,lane=threadIdx.x%32;
  int peer=int(cute::block_rank_in_cluster());
  int cluster_id=int(blockIdx.x)/2,cluster_count=int(gridDim.x)/2;
  int tiles_m=(C+ClusterBM-1)/ClusterBM,groups_i=I/256,groups=K/256,work_count=E*tiles_m*groups_i;
  bool const cache_a=K==1536;
  __syncthreads();
  TMEM::Allocator2Sm allocator;
  if(warp==0) allocator.allocate(512,&shared.tmem);
  if(threadIdx.x==0){
    for(int s=0;s<P;++s){
      cutlass::arch::ClusterBarrier::init(&shared.operand_full[s],1);
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
  __syncthreads();
  cute::cluster_sync();
  if(warp==0) allocator.release_allocation_lock();

  if(warp==1&&lane==0){
    int serial=0;
    for(int work=cluster_id;work<work_count;work+=cluster_count){
      Tile tile=decode(work,tiles_m,groups_i,COUNTS);
      if(tile.m*ClusterBM>=tile.count) continue;
      for(int half=0;half<2;++half){
      int col_base=tile.i*256+half*128;
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P,a_stage=cache_a?kg:stage;
        if(serial>=P) cutlass::arch::ClusterBarrier::wait(&shared.operand_empty[stage],((serial/P)-1)&1);
        bool const load_a=!cache_a||half==0;
        if(peer==0) cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(
            &shared.operand_full[stage],2*((load_a?BM:0)+2*BLocal)*256);
        if(load_a){
          SM100_TMA_2SM_LOAD_3D::copy(&map_a,&shared.operand_full[stage],0,shared.a[a_stage],kg*256,tile.m*ClusterBM+peer*BM,tile.e);
          SM100_TMA_2SM_LOAD_3D::copy(&map_a,&shared.operand_full[stage],0,shared.a[a_stage]+BM*128,kg*256+128,tile.m*ClusterBM+peer*BM,tile.e);
        }
        // Cache lifetime proof for G=6 and P=2: loading A[k] for the next
        // live task waits for operand_empty at serial 2*G+k-P, no earlier
        // than the previous task's last read of A[k] at serial G+k.
        // Thus the existing B-stage empty barrier protects A reuse as well.
        CUTE_UNROLL
        for(int projection=0;projection<2;++projection){
          int col=col_base+projection*I+peer*BLocal;
          SM100_TMA_2SM_LOAD_3D::copy(&map_b,&shared.operand_full[stage],0,shared.b[stage][projection],kg*256,col,tile.e);
          SM100_TMA_2SM_LOAD_3D::copy(&map_b,&shared.operand_full[stage],0,shared.b[stage][projection]+BLocal*128,kg*256+128,col,tile.e);
        }
      }
      }
    }
  }else if(warp==2){
    int serial=0;
    for(int work=cluster_id;work<work_count;work+=cluster_count){
      Tile tile=decode(work,tiles_m,groups_i,COUNTS);
      if(tile.m*ClusterBM>=tile.count) continue;
      for(int half=0;half<2;++half){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P;
        if(serial>=P) cutlass::arch::ClusterBarrier::wait(&shared.scale_empty[stage],((serial/P)-1)&1);
        CUTE_UNROLL
        for(int row=lane;row<BM;row+=32){
          int m=tile.m*ClusterBM+peer*BM+row;
          shared.scale_a[stage][row]=m<C&&m<tile.count?SA[(tile.e*C+m)*groups+kg]:0.f;
        }
        CUTE_UNROLL
        for(int projection=0;projection<2;++projection){
          int col_base=tile.i*256+half*128+projection*I;
          CUTE_UNROLL
          for(int col=lane;col<BN;col+=32)
            shared.scale_b[stage][projection][col]=SB[(tile.e*groups+kg)*(2*I)+col_base+col];
        }
        __syncwarp();
        if(lane==0) cutlass::arch::ClusterBarrier::arrive(&shared.scale_full[stage]);
      }
      }
    }
  }else if(warp==0&&peer==0){
    auto cta=Mma{}.get_slice(_0{});
    auto gC=make_tensor(make_gmem_ptr(static_cast<bfloat16_t*>(nullptr)),make_shape(_128{},_128{}),make_stride(_128{},_1{}));
    auto acc_gate=cta.make_fragment_C(cta.partition_C(gC));
    auto acc_up=cta.make_fragment_C(cta.partition_C(gC));
    Mma mma;
    int serial=0;
    for(int work=cluster_id;work<work_count;work+=cluster_count){
      Tile tile=decode(work,tiles_m,groups_i,COUNTS);
      if(tile.m*ClusterBM>=tile.count) continue;
      for(int half=0;half<2;++half){
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P,buffer=serial%4;
        cutlass::arch::ClusterBarrier::wait(&shared.operand_full[stage],(serial/P)&1);
        if(serial>=4) cutlass::arch::ClusterBarrier::wait(&shared.accumulator_empty[buffer],((serial/4)-1)&1);
        acc_gate.data()=shared.tmem+buffer*2*TmemN;
        acc_up.data()=shared.tmem+buffer*2*TmemN+TmemN;
        auto frag_a=cta.make_fragment_A(make_tensor(make_smem_ptr(shared.a[cache_a?kg:stage]),LayoutA{}));
        auto frag_bg=cta.make_fragment_B(make_tensor(make_smem_ptr(shared.b[stage][0]),LayoutB{}));
        auto frag_bu=cta.make_fragment_B(make_tensor(make_smem_ptr(shared.b[stage][1]),LayoutB{}));
        mma.accumulate_=UMMA::ScaleOut::Zero;
        CUTE_UNROLL
        for(int kk=0;kk<size<2>(frag_a);++kk){
          gemm(mma,frag_a(_,_,kk),frag_bg(_,_,kk),acc_gate);
          gemm(mma,frag_a(_,_,kk),frag_bu(_,_,kk),acc_up);
          mma.accumulate_=UMMA::ScaleOut::One;
        }
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared.operand_empty[stage],uint16_t(3));
        cutlass::arch::umma_arrive_multicast_2x1SM(&shared.accumulator_full[buffer],uint16_t(3));
      }
      }
    }
  }else if(threadIdx.x>=ProducerThreads){
    int consumer=threadIdx.x-ProducerThreads,cw=consumer/32;
    int row=(cw%2)*32+lane;
    int first_col=((cw%4)/2)*64+(cw/4)*NPerChunk;
    float total[Columns],gate_total[Columns];
    int serial=0;
    for(int work=cluster_id;work<work_count;work+=cluster_count){
      Tile tile=decode(work,tiles_m,groups_i,COUNTS);
      int m=tile.m*ClusterBM+peer*BM+row;
      if(tile.m*ClusterBM>=tile.count){
        for(int index=consumer;index<BM*256;index+=ConsumerThreads){
          int qr=index/256,col=index%256;
          int qm=tile.m*ClusterBM+peer*BM+qr;
          if(qm<C) Q[(tile.e*C+qm)*I+tile.i*256+col]=0;
        }
        if(consumer<BM){
          int qm=tile.m*ClusterBM+peer*BM+consumer;
          if(qm<C) S[(tile.e*C+qm)*groups_i+tile.i]=__fdiv_rn(1.e-12f,127.f);
        }
        continue;
      }
      for(int half=0;half<2;++half){
      CUTE_UNROLL
      for(int index=0;index<Columns;++index){total[index]=0.f;gate_total[index]=0.f;}
      for(int kg=0;kg<groups;++kg,++serial){
        int stage=serial%P,buffer=serial%4;
        cutlass::arch::ClusterBarrier::wait(&shared.accumulator_full[buffer],(serial/4)&1);
        cutlass::arch::ClusterBarrier::wait(&shared.scale_full[stage],(serial/P)&1);
        float scale_a=shared.scale_a[stage][row];
        CUTE_UNROLL
        for(int sub=0;sub<Columns/8;++sub){
          uint32_t partial_gate[8],partial_up[8];
          uint32_t address=shared.tmem+buffer*2*TmemN+(first_col%64)+sub*8+(((cw%2)*32+(first_col/64)*64)<<16);
          SM100_TMEM_LOAD_32dp32b8x::copy(address,
              partial_gate[0],partial_gate[1],partial_gate[2],partial_gate[3],partial_gate[4],partial_gate[5],partial_gate[6],partial_gate[7]);
          SM100_TMEM_LOAD_32dp32b8x::copy(address+TmemN,
              partial_up[0],partial_up[1],partial_up[2],partial_up[3],partial_up[4],partial_up[5],partial_up[6],partial_up[7]);
          cutlass::arch::fence_view_async_tmem_load();
          // Keep one four-scale fragment live and reuse it for G then U.
          CUTE_UNROLL
          for(int q=0;q<2;++q){
            CUTE_UNROLL
            for(int projection=0;projection<2;++projection){
              float scale_b[4];
              uint32_t address_b=cast_smem_ptr_to_uint(&shared.scale_b[stage][projection][first_col+sub*8+q*4]);
              asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3},[%4];\n"
                :"=f"(scale_b[0]),"=f"(scale_b[1]),"=f"(scale_b[2]),"=f"(scale_b[3]):"r"(address_b):"memory");
              CUTE_UNROLL
              for(int j=0;j<4;j+=2){
                int index=sub*8+q*4+j;
                float2 scales,aa=make_float2(scale_a,scale_a),bb=make_float2(scale_b[j],scale_b[j+1]);
                cute::mul(scales,aa,bb);
                float2 pv=projection==0?make_float2(float(int32_t(partial_gate[q*4+j])),float(int32_t(partial_gate[q*4+j+1]))):
                                       make_float2(float(int32_t(partial_up[q*4+j])),float(int32_t(partial_up[q*4+j+1])));
                float2 running=projection==0?make_float2(gate_total[index],gate_total[index+1]):make_float2(total[index],total[index+1]);
                cute::fma(running,pv,scales,running);
                if(projection==0){gate_total[index]=running.x;gate_total[index+1]=running.y;}
                else {total[index]=running.x;total[index+1]=running.y;}
              }
            }
          }
        }
        __syncwarp();
        if(lane==0){
          cutlass::arch::ClusterBarrier::arrive(&shared.accumulator_empty[buffer],0,1);
          cutlass::arch::ClusterBarrier::arrive(&shared.scale_empty[stage]);
        }
      }
      CUTE_UNROLL
      for(int j=0;j<Columns/2;++j){
        float2 g=unpack_bf16(pack_bf16(gate_total[2*j],gate_total[2*j+1]));
        float2 u=unpack_bf16(pack_bf16(total[2*j],total[2*j+1]));
        float h0=m<tile.count?__fdiv_rn(g.x,1.f+expf(-g.x))*u.x:0.f;
        float h1=m<tile.count?__fdiv_rn(g.y,1.f+expf(-g.y))*u.y:0.f;
        int col=half*128+first_col+2*j;
        shared.hidden[row][col^(row&31)]=h0;
        shared.hidden[row][(col+1)^(row&31)]=h1;
      }
      }
      // The complete G256 hidden group remains FP32. Every warp then owns
      // whole rows so scale reduction and output stores are N-contiguous.
      asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
      for(int qr=cw;qr<BM;qr+=CW){
        float values[8];
        float maximum=1.e-12f;
        CUTE_UNROLL
        for(int j=0;j<8;++j){
          int col=j*32+lane;
          float v=shared.hidden[qr][col^(qr&31)];
          values[j]=v;maximum=fmaxf(maximum,fabsf(v));
        }
        CUTE_UNROLL
        for(int d=16;d>0;d/=2) maximum=fmaxf(maximum,__shfl_down_sync(0xffffffff,maximum,d));
        maximum=__shfl_sync(0xffffffff,maximum,0);
        float scale=__fdiv_rn(maximum,127.f);
        int qm=tile.m*ClusterBM+peer*BM+qr;
        if(qm<C){
          CUTE_UNROLL
          for(int j=0;j<8;++j){
            int q=int(fminf(127.f,fmaxf(-127.f,nearbyintf(__fdiv_rn(values[j],scale)))));
            Q[(tile.e*C+qm)*I+tile.i*256+j*32+lane]=int8_t(q);
          }
          if(lane==0) S[(tile.e*C+qm)*groups_i+tile.i]=scale;
        }
      }
      // No following task overwrites hidden before all quantizer reads end.
      asm volatile("bar.sync 1,%0;\n"::"n"(ConsumerThreads):"memory");
    }
  }
  __syncthreads();
  cute::cluster_sync();
  if(warp==0) allocator.free(shared.tmem,512);
}
} // namespace grain_moe_fused

extern "C" size_t grain_moe_fused_smem_bytes(int cw){
  using namespace grain_moe_fused;
  return cw==8?sizeof(Storage<8>):cw==16?sizeof(Storage<16>):0;
}
extern "C" int grain_moe_fused_launch(
  void const* a,void const* b,void const* sa,void const* sb,void const* counts,
  void* q,void* s,int C,int I,int K,int E,int cw,int requested_blocks,void* stream){
  using namespace grain_moe_fused;
  if(!a||!b||!sa||!sb||!counts||!q||!s||C<=0||I<=0||K<=0||E<=0||C%64||I%256||K%256||(cw!=8&&cw!=16))
    return int(cudaErrorInvalidValue);
  CUtensorMap map_a,map_b;
  uint64_t dims_a[3]={uint64_t(K),uint64_t(C),uint64_t(E)},dims_b[3]={uint64_t(K),uint64_t(2*I),uint64_t(E)};
  uint64_t strides_a[2]={uint64_t(K),uint64_t(K)*C},strides_b[2]={uint64_t(K),uint64_t(K)*(2*I)};
  uint32_t box_a[3]={128,64,1},box_b[3]={128,64,1},element_stride[3]={1,1,1};
  auto encode=[&](CUtensorMap* map,void const* ptr,uint64_t* dims,uint64_t* strides,uint32_t* box){
    return cuTensorMapEncodeTiled(map,CU_TENSOR_MAP_DATA_TYPE_UINT8,3,const_cast<void*>(ptr),dims,strides,box,element_stride,
      CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  };
  CUresult encoded=encode(&map_a,a,dims_a,strides_a,box_a);
  if(encoded!=CUDA_SUCCESS) return -10000-int(encoded);
  encoded=encode(&map_b,b,dims_b,strides_b,box_b);
  if(encoded!=CUDA_SUCCESS) return -10000-int(encoded);
  int work_count=E*((C+127)/128)*(I/256),blocks=requested_blocks;
  cudaError_t status;
  if(blocks<=0){
    int device;
    status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
    status=cudaDeviceGetAttribute(&blocks,cudaDevAttrMultiProcessorCount,device);if(status!=cudaSuccess)return int(status);
  }
  int clusters=blocks/2+blocks%2;
  clusters=clusters<work_count?clusters:work_count;
  blocks=2*clusters;
  cudaLaunchAttribute attribute{};
  attribute.id=cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x=2;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
  cudaLaunchConfig_t launch_config{};
  launch_config.gridDim=dim3(blocks,1,1);
  launch_config.dynamicSmemBytes=0;
  launch_config.stream=static_cast<cudaStream_t>(stream);
  launch_config.attrs=&attribute;launch_config.numAttrs=1;
  if(cw==16){
    auto fn=kernel<16>;
    status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage<16>));
    if(status!=cudaSuccess)return int(status);
    launch_config.blockDim=dim3(ProducerThreads+16*32,1,1);
    launch_config.dynamicSmemBytes=sizeof(Storage<16>);
    status=cudaLaunchKernelEx(&launch_config,fn,
      static_cast<float const*>(sa),static_cast<float const*>(sb),static_cast<int32_t const*>(counts),
      static_cast<int8_t*>(q),static_cast<float*>(s),C,I,K,E,map_a,map_b);
    if(status!=cudaSuccess)return int(status);
  }else{
    auto fn=kernel<8>;
    status=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Storage<8>));
    if(status!=cudaSuccess)return int(status);
    launch_config.blockDim=dim3(ProducerThreads+8*32,1,1);
    launch_config.dynamicSmemBytes=sizeof(Storage<8>);
    status=cudaLaunchKernelEx(&launch_config,fn,
      static_cast<float const*>(sa),static_cast<float const*>(sb),static_cast<int32_t const*>(counts),
      static_cast<int8_t*>(q),static_cast<float*>(s),C,I,K,E,map_a,map_b);
    if(status!=cudaSuccess)return int(status);
  }
  return int(cudaGetLastError());
}
extern "C" const char* grain_moe_fused_error_string(int code){
  if(code<=-10000){const char* message=nullptr;cuGetErrorString(CUresult(-10000-code),&message);return message?message:"TMA encode error";}
  return cudaGetErrorString(cudaError_t(code));
}
