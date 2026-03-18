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

constexpr int WARP      = 32;
constexpr int ITERS     = 1 << 20;
constexpr int CHUNK_B   = 32;
constexpr int SEG_BYTES = 16;   // allowed: 32, 16, 8, 4, 2, 1

static_assert(
    SEG_BYTES == 32 || SEG_BYTES == 16 || SEG_BYTES == 8 ||
    SEG_BYTES == 4  || SEG_BYTES == 2  || SEG_BYTES == 1,
    "SEG_BYTES must be one of: 32, 16, 8, 4, 2, 1"
);

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
//   - process one 32B chunk
//   - split it into 32 / SEG_BYTES segments
//   - for each segment: load SEG_BYTES data pattern, then perform one computation
// lane_in_seg keeps all 32 threads active, but only SEG_BYTES unique byte
// positions are touched in each segment.
template <int SEG>
__global__ void load_contiguous_segmented_kernel(const uint8_t* in,
                                                 volatile unsigned long long* sink,
                                                 int iters) {
    static_assert(
        SEG == 32 || SEG == 16 || SEG == 8 ||
        SEG == 4  || SEG == 2  || SEG == 1,
        "SEG must be one of: 32, 16, 8, 4, 2, 1"
    );

    int lane = threadIdx.x;
    int lane_in_seg = lane & (SEG - 1);
    unsigned long long acc = 0;

    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        size_t base = static_cast<size_t>(t) * CHUNK_B;

        #pragma unroll
        for (int seg = 0; seg < CHUNK_B; seg += SEG) {
            size_t idx = base + static_cast<size_t>(seg + lane_in_seg);
            uint8_t v = ld_global_ca_u8(in + idx);
            acc += static_cast<unsigned long long>(v);
        }
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

template <int SEG>
void run_contiguous_segmented() {
    size_t n_elems   = static_cast<size_t>(ITERS) * CHUNK_B;
    size_t out_bytes = WARP * sizeof(unsigned long long);

    uint8_t* d_in = make_device_input(n_elems);

    unsigned long long* d_sink = nullptr;
    unsigned long long h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_sink, out_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, out_bytes));

    load_contiguous_segmented_kernel<SEG><<<1, WARP>>>(d_in, d_sink, ITERS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, out_bytes, cudaMemcpyDeviceToHost));

    printf("contiguous u8 segmented (SEG_BYTES=%d) done, checksum=%llu\n",
           SEG,
           static_cast<unsigned long long>(h_sink[0]));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
}

int main() {
    run_contiguous_segmented<SEG_BYTES>();
    return 0;
}
