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

template <typename T>
__global__ void load_contiguous_kernel(const T* __restrict__ in,
                                       volatile unsigned long long* __restrict__ sink,
                                       int iters) {
    int lane = threadIdx.x;
    unsigned long long acc = 0;

    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        size_t base = static_cast<size_t>(t) * WARP;
        T v = in[base + lane];
        acc += static_cast<unsigned long long>(v);
    }

    sink[lane] = acc;
}

template <typename T>
__global__ void load_stride_kernel(const T* __restrict__ in,
                                   volatile unsigned long long* __restrict__ sink,
                                   int iters,
                                   size_t stride_elems) {
    int lane = threadIdx.x;
    unsigned long long acc = 0;

    #pragma unroll 1
    for (int t = 0; t < iters; ++t) {
        size_t base = static_cast<size_t>(t) * WARP * stride_elems;
        T v = in[base + lane * stride_elems];
        acc += static_cast<unsigned long long>(v);
    }

    sink[lane] = acc;
}

template <typename T>
T* make_device_input(size_t n_elems) {
    size_t bytes = n_elems * sizeof(T);

    T* h_in = static_cast<T*>(std::malloc(bytes));
    if (!h_in) {
        std::fprintf(stderr, "Host malloc failed for %zu bytes\n", bytes);
        std::exit(EXIT_FAILURE);
    }

    for (size_t i = 0; i < n_elems; ++i) {
        h_in[i] = static_cast<T>((i & 0xff) + 1);
    }

    T* d_in = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    std::free(h_in);
    return d_in;
}

template <typename T>
void run_contiguous(const char* label) {
    size_t n_elems = static_cast<size_t>(ITERS) * WARP;
    size_t out_bytes = WARP * sizeof(unsigned long long);

    T* d_in = make_device_input<T>(n_elems);

    unsigned long long* d_sink = nullptr;
    unsigned long long h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_sink, out_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, out_bytes));

    load_contiguous_kernel<T><<<1, WARP>>>(d_in, d_sink, ITERS);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, out_bytes, cudaMemcpyDeviceToHost));

    printf("%s (%zu-bit) done, checksum=%llu\n",
           label,
           sizeof(T) * 8,
           static_cast<unsigned long long>(h_sink[0]));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
}

template <typename T>
void run_stride(const char* label, size_t stride_bytes) {
    if (stride_bytes % sizeof(T) != 0) {
        std::fprintf(stderr,
                     "Invalid stride_bytes=%zu for element size %zu\n",
                     stride_bytes, sizeof(T));
        std::exit(EXIT_FAILURE);
    }

    size_t stride_elems = stride_bytes / sizeof(T);
    size_t n_elems = static_cast<size_t>(ITERS) * WARP * stride_elems;
    size_t out_bytes = WARP * sizeof(unsigned long long);

    T* d_in = make_device_input<T>(n_elems);

    unsigned long long* d_sink = nullptr;
    unsigned long long h_sink[WARP];

    CHECK_CUDA(cudaMalloc(&d_sink, out_bytes));
    CHECK_CUDA(cudaMemset(d_sink, 0, out_bytes));

    load_stride_kernel<T><<<1, WARP>>>(d_in, d_sink, ITERS, stride_elems);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_sink, d_sink, out_bytes, cudaMemcpyDeviceToHost));

    printf("%s (%zu-bit, stride=%zuB) done, checksum=%llu\n",
           label,
           sizeof(T) * 8,
           stride_bytes,
           static_cast<unsigned long long>(h_sink[0]));

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_sink));
}

int main() {
    run_contiguous<uint8_t>("contiguous");
    run_contiguous<uint16_t>("contiguous");
    run_contiguous<uint32_t>("contiguous");
    run_contiguous<uint64_t>("contiguous");

    run_stride<uint8_t>("stride", 32);
    run_stride<uint16_t>("stride", 32);
    run_stride<uint32_t>("stride", 32);
    run_stride<uint64_t>("stride", 32);

    return 0;
}
