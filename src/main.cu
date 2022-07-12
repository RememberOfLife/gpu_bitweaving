#include "cuda_try.cuh"
#include "cuda_time.cuh"
#include <cstdint>
#include <cstddef>
#include "fast_prng.cuh"

#define THREADS_PER_WARP 32

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

// all elem counts are a multiple of the weaving width, i.e. worst case bits -> multiple of 32

template <bool (*COMPARE_OP)(uint32_t left, uint32_t right)>
__global__ void kernel_classic_compare(uint32_t* data_left, uint32_t* data_right, size_t elems, uint32_t* mask_out)
{
    warp_stats w = get_warp_stats();
    // most basic, unoptimized readin / writeout, just 32 threads comparing their neighorbing elements and one writes out
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; tid < elems; tid += stride) {
        bool bit = COMPARE_OP(data_left[tid], data_right[tid]);
        __syncwarp();
        uint32_t rv = __ballot_sync(0xFFFFFFFF, bit);
        if (w.offset == 0) {
            mask_out[tid / 32] = rv;
        }
    }
}

__global__ void kernel_byteweaving_in(uint32_t* data, size_t elems, uint32_t* data_woven)
{
    const size_t elems_per_thread = 4; // byte weaving
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x * elems_per_thread;
    size_t stride = elems_per_thread * blockDim.x * gridDim.x;
    uint32_t elem_cache[elems_per_thread] = {0, 0, 0, 0}; // msbytes at idx 0, left data elements go to the right of the woven el
    for (; tid < elems; tid += stride) {
        for (int i = 0; i < elems_per_thread; i++) {
            uint32_t in_el = data[tid + i];
            if (threadIdx.x == 0) {
                printf("%u\n", in_el);
            }
            elem_cache[3] |= ((in_el)&0xFF) << i * 8;
            elem_cache[2] |= ((in_el >> 8) & 0xFF) << i * 8;
            elem_cache[1] |= ((in_el >> 16) & 0xFF) << i * 8;
            elem_cache[0] |= ((in_el >> 24) & 0xFF) << i * 8;
        }
        for (int i = 0; i < elems_per_thread; i++) {
            data_woven[(tid / elems_per_thread) * i] = elem_cache[i];
            elem_cache[i] = 0;
        }
    }
}

// TODO weaving_out

template <bool (*COMPARE_OP)(uint32_t left, uint32_t right)>
__global__ void kernel_byteweaving_op(uint32_t* data_left, uint32_t* data_right, size_t elems, uint32_t* mask)
{
    // TODO
}

__host__ __device__ bool compare_equal(uint32_t left, uint32_t right)
{
    return left == right;
}

int main()
{
    size_t elems = 32;
    uint32_t* h_dl = (uint32_t*)malloc(sizeof(uint32_t) * elems);
    uint32_t* h_dr = (uint32_t*)malloc(sizeof(uint32_t) * elems);
    uint32_t* h_res1 = (uint32_t*)malloc(sizeof(uint32_t) * elems / 32);
    uint32_t* h_res2 = (uint32_t*)malloc(sizeof(uint32_t) * elems / 32);

    // generate data
    fast_prng rng(42);
    for (size_t i = 0; i < elems; i++) {
        uint32_t r = rng.rand();
        h_dl[i] = r;
        if (r % 8 == 0) {
            h_dr[i] = r;
        }
        else {
            h_dr[i] = rng.rand();
        }
    }

    uint32_t* d_dl;
    uint32_t* d_dr;
    uint32_t* d_dl_woven;
    uint32_t* d_dr_woven;
    uint32_t* d_res1;
    uint32_t* d_res2;
    CUDA_TRY(cudaMalloc(&d_dl, sizeof(uint32_t) * elems));
    CUDA_TRY(cudaMalloc(&d_dr, sizeof(uint32_t) * elems));
    CUDA_TRY(cudaMalloc(&d_dl_woven, sizeof(uint32_t) * elems));
    CUDA_TRY(cudaMalloc(&d_dr_woven, sizeof(uint32_t) * elems));
    CUDA_TRY(cudaMalloc(&d_res1, sizeof(uint32_t) * elems / 32));
    CUDA_TRY(cudaMalloc(&d_res2, sizeof(uint32_t) * elems / 32));
    CUDA_TRY(cudaMemcpy(d_dl, h_dl, sizeof(uint32_t) * elems, cudaMemcpyHostToDevice));
    CUDA_TRY(cudaMemcpy(d_dr, h_dr, sizeof(uint32_t) * elems, cudaMemcpyHostToDevice));

    // run kernels
    printf("%zu elems\n", elems);
    float time;
    CUDA_QUICKTIME(&time, kernel_classic_compare<&compare_equal><<<1, 32>>>(d_dl, d_dr, elems, d_res1));
    printf("classic: %.3f ms\n", time);
    CUDA_QUICKTIME(&time, {
        kernel_byteweaving_in<<<1, 32>>>(d_dl, elems, d_dl_woven);
        kernel_byteweaving_in<<<1, 32>>>(d_dr, elems, d_dr_woven);
    });
    printf("byteweave-ingest: %.3f ms\n", time);
    CUDA_QUICKTIME(&time, kernel_byteweaving_op<&compare_equal><<<1, 32>>>(d_dl_woven, d_dr_woven, elems, d_res2));
    printf("byteweave-compare: %.3f ms\n", time);

    CUDA_TRY(cudaMemcpy(h_res1, d_res1, sizeof(uint32_t) * elems / 32, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_res2, d_res2, sizeof(uint32_t) * elems / 32, cudaMemcpyDeviceToHost));

    for (int i = 0; i < elems; i++) {
        printf("DL #%02u : ", i);
        bit_print(h_dl[i]);
        printf("\n");
    }
    uint32_t* h_dl_woven = (uint32_t*)malloc(sizeof(uint32_t) * elems);
    CUDA_TRY(cudaMemcpy(h_dl_woven, d_dl_woven, sizeof(uint32_t) * elems, cudaMemcpyDeviceToHost));
    for (int i = 0; i < elems; i++) {
        printf("WL #%02u : ", i);
        bit_print(h_dl_woven[i]);
        printf("\n");
    }

    // compare results
    // due to ballot sync, element idx 0 gets the lsb in the corresponding mask, i.e. mask >> idx
    for (int i = 0; i < elems; i++) {
        if (h_res1[i] != h_res2[i]) {
            printf("FAIL #%d\n", i);
            exit(1);
        }
    }

    CUDA_TRY(cudaFree(d_dl));
    CUDA_TRY(cudaFree(d_dr));
    CUDA_TRY(cudaFree(d_dl_woven));
    CUDA_TRY(cudaFree(d_dr_woven));
    CUDA_TRY(cudaFree(d_res1));
    CUDA_TRY(cudaFree(d_res2));

    free(h_dl);
    free(h_dr);
    free(h_res1);
    free(h_res2);
    printf("DONE\n");
}
