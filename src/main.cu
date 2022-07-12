#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "fast_prng.cuh"

#define THREADS_PER_WARP 32
#define TABLE_SIZE ((size_t)THREADS_PER_WARP * sizeof(uint16_t))

struct warp_stats {
    int idx;
    int offset;
    int base;
};

__device__ warp_stats get_warp_stats()
{
    warp_stats w;
    w.idx = threadIdx.x / THREADS_PER_WARP;
    w.offset = threadIdx.x % THREADS_PER_WARP;
    w.base = THREADS_PER_WARP * w.idx;
    return w;
}

template <typename T> void bit_print(T data, bool spacing = true)
{
    size_t typewidth_m1 = sizeof(T) * 8 - 1;
    for (int i = typewidth_m1; i >= 0; i--) {
        printf("%c", (data >> i) & 0b1 ? '1' : '0');
        if (spacing && i < typewidth_m1 && i > 0 && i % 8 == 0) {
            printf(" ");
        }
    }
}

int main()
{
    printf("DONE\n");
}
