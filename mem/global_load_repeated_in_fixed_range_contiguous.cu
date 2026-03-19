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

constexpr int WARP        = 32;
constexpr int INNER_ITERS = 1024;
constexpr int OUTER_LOOPS = 10;

__device__ __forceinline__ uint8_t ld_global_ca_u8(const uint8_t* p) {
    unsigned int v;
    asm volatile(
        "ld.global.ca.u8 %0, [%1];\n"
        : "=r"(v)
        : "l"(p)
    );
    return static_cast<uint8_t>(v);
}

__global__ void load_u8_contiguous_same_base_10x1024_kernel(
    const uint8_t* in,
    volatile unsigned long long* sink) {

    int lane = threadIdx.x;
    unsigned long long acc = 0;

    #pragma unroll 1
    for (int outer = 0; outer < OUTER_LOOPS; ++outer) {
        #pragma unroll 1
        for (int t = 0; t < INNER_ITERS; ++t) {
            size_t idx = static_cast<size_t>(t) * WARP + static_cast<size_t>(lane);
            uint8_t v = ld_global_ca_u8(in + idx);
            acc += static_cast<unsigned long long>(v);
        }
    }

    sink[lane] = acc;
}

uint8_t* make_device_input(size_t n_elems) {
    size_t bytes = n_elems * sizeof(uint8_t);

    uint8_t* h_in = static_cast<uint8_t*>(std::malloc(bytes));
    if (!h_in) {
        std::fprintf(stderr, "Host malloc failed for %zu bytes\n", bytes);
        std::exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n_elems; ++i) {
        h_in[i] = static_cast<uint8_t>((i & 0xff) + 1);
    }

    uint8_t* d_in = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    std::free(h_in);
    return d_in;
}

int main() {
    constexpr size_t n_elems   = static_cast<size_t>(INNER_ITERS) * WARP;
    constexpr size_t out_bytes = WARP * sizeof(unsigned long long);

    uint8_t* d_in = make_device_input(n_elems);

    unsigned long long* d_sink = nullptr;
    unsigned long long h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_sink, out_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, out_bytes));

    load_u8_contiguous_same_base_10x1024_kernel<<<1, WARP>>>(d_in, d_sink);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, out_bytes, cudaMemcpyDeviceToHost));

    std::printf("u8 contiguous same-base 10x1024 done, checksum=%llu\n",
                static_cast<unsigned long long>(h_sink[0]));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
    return 0;
}
