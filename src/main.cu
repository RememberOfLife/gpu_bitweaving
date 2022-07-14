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
            elem_cache[3] |= ((in_el)&0xFF) << (i * 8);
            elem_cache[2] |= ((in_el >> 8) & 0xFF) << (i * 8);
            elem_cache[1] |= ((in_el >> 16) & 0xFF) << (i * 8);
            elem_cache[0] |= ((in_el >> 24) & 0xFF) << (i * 8);
        }
        for (int i = 0; i < elems_per_thread; i++) {
            data_woven[(tid / elems_per_thread) + (elems / elems_per_thread) * i] = elem_cache[i];
            elem_cache[i] = 0;
        }
    }
}

// TODO weaving_out

template <int (*WEAVE_COMPARE_OP)(uint32_t left, uint32_t right)>
__global__ void kernel_byteweaving_op(uint32_t* data_left, uint32_t* data_right, size_t elems, uint32_t* mask)
{
    // TODO
    const size_t elems_per_thread = 4; // byte weaving
    const size_t element_width = 32;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x * element_width;
    size_t stride = element_width * blockDim.x * gridDim.x;
    int compare_cache[element_width];
    uint32_t writeout_bits = 0;
    for (; tid < elems; tid += stride) {
        for (int j = 0; j < element_width; j++) {
            compare_cache[j] = -1;
        }
        // w is the idx into the continuous uint32 of the woven input per chunk
        for (int w = 0; w < element_width / elems_per_thread; w++) { // byteweaving processes 4 elemens partial compare per uint32
            int i = 0; // which byte of the weaving is being processes right now
            bool need_info = true;
            while (need_info) {
                need_info = false;
                // load in woven left and right
                uint32_t woven_in_el_left = data_left[(tid / elems_per_thread) + (elems / elems_per_thread) * i + w];
                uint32_t woven_in_el_right = data_right[(tid / elems_per_thread) + (elems / elems_per_thread) * i + w];
                // if all elements are resolved, go next
                for (int j = 0; j < elems_per_thread; j++) {
                    int comp = WEAVE_COMPARE_OP((woven_in_el_left >> (j * 8)) & 0xFF, (woven_in_el_right >> (j * 8)) & 0xFF);
                    // if the comparison for this data piece was true for earlier pieces of data, AND the new compare information to it
                    if (comp == 1) {
                        compare_cache[j + w * elems_per_thread] = (compare_cache[j + w * elems_per_thread] == 1 ? 1 : 0);
                    }
                    else {
                        compare_cache[j + w * elems_per_thread] = comp;
                    }
                    // load more info if: weave compare requests it, compare is true but this is not the last data piece
                    need_info |=
                        (compare_cache[j + w * elems_per_thread] == -1 || (i < elems_per_thread - 1 && compare_cache[j + w * elems_per_thread] == 1));
                }
                i++;
            }
        }
        // no more info required and whole writeout cache is full, store results from compare cache
        writeout_bits = 0;
        for (int i = 0; i < element_width; i++) {
            writeout_bits |= ((compare_cache[i] ? 1 : 0) << i);
        }
        mask[tid / element_width] = writeout_bits;
    }
}

// WARNING in general weave compare functions, e.g. >= return -1 if they need more information
//  i.e. if both first bytes are 0 then >= can not be determined, so it will return 0
//  for all cases where the supplied data is enough, 0 means compare false and 1 means compare true
// TODO bool for more data available? or general spec to make 0 sticky?, or use a template in the kernel to specify which result is sticky
__host__ __device__ int weave_compare_equal(uint32_t left, uint32_t right)
{
    return left == right ? 1 : 0;
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
        // debugging numbers every byte uniquely, up to 64 elems
        static uint8_t a = 1;
        h_dl[i] = (a << 24) | ((a + 1) << 16) | ((a + 2) << 8) | (a + 3);
        a += 4;
        h_dr[i] = h_dl[i] + 1; //(r % 4 == 0 ? h_dl[i] : r);
        if (i == 0) {
            h_dr[i] = h_dl[i];
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
    CUDA_QUICKTIME(&time, kernel_byteweaving_op<&weave_compare_equal><<<1, 32>>>(d_dl_woven, d_dr_woven, elems, d_res2));
    printf("byteweave-compare: %.3f ms\n", time);

    CUDA_TRY(cudaMemcpy(h_res1, d_res1, sizeof(uint32_t) * elems / 32, cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaMemcpy(h_res2, d_res2, sizeof(uint32_t) * elems / 32, cudaMemcpyDeviceToHost));

    printf("\n");
    for (int i = 0; i < elems; i++) {
        printf("DL #%02u : ", i);
        bit_print(h_dl[i]);
        printf(" %s ", h_dl[i] == h_dr[i] ? "==" : "!=");
        bit_print(h_dr[i]);
        printf("\n");
    }
    printf("\n");
    uint32_t* h_dl_woven = (uint32_t*)malloc(sizeof(uint32_t) * elems);
    CUDA_TRY(cudaMemcpy(h_dl_woven, d_dl_woven, sizeof(uint32_t) * elems, cudaMemcpyDeviceToHost));
    for (int i = 0; i < elems; i++) {
        printf("WL #%02u : ", i);
        bit_print(h_dl_woven[i]);
        printf("\n");
    }
    printf("\n");
    printf("RES C  : ");
    bit_print(h_res1[0]);
    printf("\n");
    printf("RES W  : ");
    bit_print(h_res2[0]);
    printf("\n");

    // compare results
    // due to ballot sync, element idx 0 gets the lsb in the corresponding mask, i.e. mask >> idx
    for (int i = 0; i < elems / 32; i++) {
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
