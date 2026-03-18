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

constexpr int WARP              = 32;
constexpr int STRIDE_B          = 32;
constexpr int COLUMN_NBYTES     = WARP * STRIDE_B;         // 32 * 32 = 1024 bytes
constexpr int EVEN_ODD_BLOCKS   = 64;
constexpr int EVEN_ODD_NBYTES   = EVEN_ODD_BLOCKS * STRIDE_B; // 64 * 32 = 2048 bytes

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

// Thread i first reads the first byte of block 2*i,
// then reads the first byte of block 2*i+1.
__global__ void even_odd_first_byte_kernel(const uint8_t* in,
                                           volatile unsigned int* sink) {
    int lane = threadIdx.x;
    if (lane >= WARP) return;

    size_t even_block_base = static_cast<size_t>(2 * lane) * STRIDE_B;
    size_t odd_block_base  = static_cast<size_t>(2 * lane + 1) * STRIDE_B;

    uint8_t v0 = ld_global_ca_u8(in + even_block_base);
    uint8_t v1 = ld_global_ca_u8(in + odd_block_base);

    // Do a simple operation so both loads contribute to the final result.
    unsigned int acc = static_cast<unsigned int>(v0);
    acc = acc * 33u + static_cast<unsigned int>(v1);

    sink[lane] = acc;
}

int main() {
    {
        uint8_t h_in[COLUMN_NBYTES];
        for (int i = 0; i < COLUMN_NBYTES; ++i) {
            h_in[i] = static_cast<uint8_t>((i & 0xff) + 1);
        }

        uint8_t* d_in = nullptr;
        unsigned int* d_sink = nullptr;
        unsigned int h_sink[WARP];

        CHECK_CUDA(cudaMalloc(&d_in, COLUMN_NBYTES));
        CHECK_CUDA(cudaMalloc(&d_sink, WARP * sizeof(unsigned int)));

        CHECK_CUDA(cudaMemcpy(d_in, h_in, COLUMN_NBYTES, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_sink, 0, WARP * sizeof(unsigned int)));

        bench_kernel<<<1, WARP>>>(d_in, d_sink);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_sink, d_sink, WARP * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        printf("bench_kernel\n");
        for (int i = 0; i < WARP; ++i) {
            printf("lane %2d: %u\n", i, h_sink[i]);
        }

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_sink));
    }

    {
        uint8_t h_in[EVEN_ODD_NBYTES];
        for (int i = 0; i < EVEN_ODD_NBYTES; ++i) {
            h_in[i] = static_cast<uint8_t>((i & 0xff) + 1);
        }

        uint8_t* d_in = nullptr;
        unsigned int* d_sink = nullptr;
        unsigned int h_sink[WARP];

        CHECK_CUDA(cudaMalloc(&d_in, EVEN_ODD_NBYTES));
        CHECK_CUDA(cudaMalloc(&d_sink, WARP * sizeof(unsigned int)));

        CHECK_CUDA(cudaMemcpy(d_in, h_in, EVEN_ODD_NBYTES, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_sink, 0, WARP * sizeof(unsigned int)));

        even_odd_first_byte_kernel<<<1, WARP>>>(d_in, d_sink);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_sink, d_sink, WARP * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        printf("even_odd_first_byte_kernel\n");
        for (int i = 0; i < WARP; ++i) {
            printf("lane %2d: %u\n", i, h_sink[i]);
        }

        CHECK_CUDA(cudaFree(d_in));
        CHECK_CUDA(cudaFree(d_sink));
    }

    return 0;
}
