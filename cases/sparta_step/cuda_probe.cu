// Verify the allocated CUDA device and kernel execution before building SPARTA.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

static void check(cudaError_t e) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA_PROBE_FAILED: %s\n", cudaGetErrorString(e));
    std::exit(1);
  }
}
__global__ void probe(int *p) { *p = 42; }
int main() {
  check(cudaSetDevice(0));
  cudaDeviceProp d; check(cudaGetDeviceProperties(&d, 0));
  int *p, value=0;
  check(cudaMalloc(&p, sizeof(int)));
  probe<<<1,1>>>(p);
  check(cudaGetLastError()); check(cudaDeviceSynchronize());
  check(cudaMemcpy(&value, p, sizeof(int), cudaMemcpyDeviceToHost));
  check(cudaFree(p));
  if (value!=42) return 2;
  size_t available, total;
  check(cudaMemGetInfo(&available, &total));
  std::printf("{\"name\":\"%s\",\"compute_capability\":\"%d.%d\",", d.name,d.major,d.minor);
  std::printf("\"uuid\":\"GPU-");
  for (int i=0; i<16; ++i) {
    if (i==4 || i==6 || i==8 || i==10) std::printf("-");
    std::printf("%02x", static_cast<unsigned char>(d.uuid.bytes[i]));
  }
  std::printf("\",\"total_bytes\":%zu,\"free_bytes\":%zu,\"kernel_pass\":true}\n",total,available);
}
