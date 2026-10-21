// Exact SiLU lookup indexed by an already-rounded BF16 gate. Runtime-generated
// once per CUDA device using the same FP32 expression as the original epilogue.
namespace grain_moe_silu_lookup {
__device__ float table[65536];
__global__ void initialize(){
  unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index<65536){float gate=__uint_as_float(index<<16);table[index]=__fdiv_rn(gate,1.f+expf(-gate));}
}
__device__ __forceinline__ float evaluate(float gate){
  unsigned index=__float_as_uint(gate)>>16;
  return __ldg(table+index);
}
__global__ void verify(float* reference,float* looked_up){
  unsigned index=blockIdx.x*blockDim.x+threadIdx.x;
  if(index<65536){
    float gate=__uint_as_float(index<<16);
    reference[index]=__fdiv_rn(gate,1.f+expf(-gate));
    looked_up[index]=evaluate(gate);
  }
}
}
extern "C" int grain_moe_silu_lut_init(void* stream){
  grain_moe_silu_lookup::initialize<<<256,256,0,static_cast<cudaStream_t>(stream)>>>();
  return int(cudaGetLastError());
}
extern "C" int grain_moe_mega_cluster_initialize(void* stream){return grain_moe_silu_lut_init(stream);}
extern "C" int grain_moe_interleaved_initialize(void* stream){return grain_moe_silu_lut_init(stream);}
extern "C" int grain_moe_silu_lut_copy(void* destination,void* stream){
  return int(cudaMemcpyFromSymbolAsync(destination,grain_moe_silu_lookup::table,65536*sizeof(float),0,
              cudaMemcpyDeviceToDevice,static_cast<cudaStream_t>(stream)));
}
extern "C" int grain_moe_silu_lut_verify(void* reference,void* looked_up,void* stream){
  grain_moe_silu_lookup::verify<<<256,256,0,static_cast<cudaStream_t>(stream)>>>(
    static_cast<float*>(reference),static_cast<float*>(looked_up));
  return int(cudaGetLastError());
}
