#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

constexpr int WARP     = 32;
constexpr int STRIDE_B = 32;
constexpr int NBYTES   = WARP * STRIDE_B;   // 32 * 32 = 1024 bytes

__device__ __forceinline__ uint8_t ld_global_ca_u8(const uint8_t* p) {
    unsigned int v;
    asm volatile(
        "ld.global.ca.u8 %0, [%1];\n"
        : "=r"(v)
        : "l"(p)
    );
    return static_cast<uint8_t>(v);
}

// In iteration k, thread i reads the k-th byte of the i-th 32-byte block.
__global__ void bench_kernel(const uint8_t* in, volatile unsigned int* sink) {
    int lane = threadIdx.x;
    if (lane >= WARP) return;

    size_t base = static_cast<size_t>(lane) * STRIDE_B;
    unsigned int acc = 0;

    #pragma unroll 1
    for (int k = 0; k < STRIDE_B; ++k) {
        uint8_t v = ld_global_ca_u8(in + base + k);

        // Do a simple operation so the load is not optimized away.
        acc = acc * 33u + static_cast<unsigned int>(v);
    }

    sink[lane] = acc;
}

int main() {
    uint8_t h_in[NBYTES];
    for (int i = 0; i < NBYTES; ++i) {
        h_in[i] = static_cast<uint8_t>((i & 0xff) + 1);
    }

    uint8_t* d_in = nullptr;
    unsigned int* d_sink = nullptr;
    unsigned int h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_in, NBYTES));
    CHECK_CUDA(cudaMalloc(&d_sink, WARP * sizeof(unsigned int)));

    CHECK_CUDA(cudaMemcpy(d_in, h_in, NBYTES, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sink, 0, WARP * sizeof(unsigned int)));

    bench_kernel<<<1, WARP>>>(d_in, d_sink);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, WARP * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < WARP; ++i) {
        printf("lane %2d: %u\n", i, h_sink[i]);
    }

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
