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

constexpr int WARP  = 32;
constexpr int ITERS = 1 << 20;

// -------------------- inline PTX global load --------------------
__device__ __forceinline__ uint8_t ld_global_ca_u8(const uint8_t* p) {
    unsigned int v;
    asm volatile(
        "ld.global.ca.u8 %0, [%1];\n"
        : "=r"(v)
        : "l"(p)
    );
    return static_cast<uint8_t>(v);
}

// -------------------- kernel --------------------
// For each t:
//   1) 32 threads first access the first 16B of the current 32B chunk
//   2) perform one computation
//   3) 32 threads then access the second 16B of the current 32B chunk
//   4) perform one more computation
// Then proceed to the next t
__global__ void load_contiguous_split16_kernel(const uint8_t* in,
                                               volatile unsigned long long* sink,
                                               int iters) {
    int lane   = threadIdx.x;     // 0..31
    int lane16 = lane & 15;       // 0..15, keeps all 32 threads active
    unsigned long long acc = 0;

    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        size_t base = static_cast<size_t>(t) * 32;

        // First load: first 16B
        uint8_t v0 = ld_global_ca_u8(in + base + static_cast<size_t>(lane16));  // [0, 15]
        acc += static_cast<unsigned long long>(v0);  // perform one computation

        // Second load: second 16B
        uint8_t v1 = ld_global_ca_u8(in + base + 16 + static_cast<size_t>(lane16)); // [16, 31]
        acc += static_cast<unsigned long long>(v1);  // perform one more computation
    }

    sink[lane] = acc;
}

// -------------------- host helpers --------------------

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

void run_contiguous_split16() {
    size_t n_elems   = static_cast<size_t>(ITERS) * 32;
    size_t out_bytes = WARP * sizeof(unsigned long long);

    uint8_t* d_in = make_device_input(n_elems);

    unsigned long long* d_sink = nullptr;
    unsigned long long h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_sink, out_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, out_bytes));

    load_contiguous_split16_kernel<<<1, WARP>>>(d_in, d_sink, ITERS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, out_bytes, cudaMemcpyDeviceToHost));

    printf("contiguous u8 split16 done, checksum=%llu\n",
           static_cast<unsigned long long>(h_sink[0]));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
}

int main() {
    run_contiguous_split16();
    return 0;
}
