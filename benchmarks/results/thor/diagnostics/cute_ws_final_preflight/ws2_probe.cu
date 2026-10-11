#include "/tmp/thor-ws2-race-syncsf/int8_g256_pipeline.cuh"
extern "C" int grain_warp_g256(void const*,void const*,void const*,void const*,void*,int,int,int,int,void*) { return 1; }
extern "C" int probe(void const*a,void const*b,void const*sa,void const*sb,void*c,int M,int N,int K,void*stream,int p) {
 if (p==2) return thor_ws2::launch<2>(a,b,sa,sb,c,M,N,K,(cudaStream_t)stream);
 return thor_ws2::launch<3>(a,b,sa,sb,c,M,N,K,(cudaStream_t)stream);
}
