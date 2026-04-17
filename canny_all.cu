/*
 * canny_all.cu — Self-contained Canny kernels + CPU hysteresis in one file.
 *
 * Pipeline order:
 *   1. gaussian_coarse_4x4      — GPU kernel: fused separable 5x5 Gaussian
 *   2. SobelNMSFusedKernelCorrected — GPU kernel: fused Sobel + NMS + threshold
 *   3. remap_and_border         — GPU kernel: remap NMS values + add 1px border
 *   4. hysteresis_omp_numa      — CPU (OpenMP): NUMA-aware BFS hysteresis
 *   5. final_output             — CPU: map interior (2 -> 255) to binary
 *
 * Map convention (shared by Sobel/NMS output and hysteresis input):
 *   0 = weak candidate (after remap), 1 = suppressed / border, 2 = strong
 *
 * Hysteresis buffer layout:
 *   (W+2) x (H+2), 1-pixel border of 1s so 8-neighbor reads are in-bounds.
 */

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#if defined(_MSC_VER)
  #include <intrin.h>
#endif

/* ===========================================================================
 * 1. Gaussian 5x5, sigma=1.4, fused separable, thread-coarsened 4x4.
 * ===========================================================================
 * Block : 16x16 = 256 threads, each writes 4x4 strided output pixels.
 * Tile  : 68x68 unsigned char shared (4624 B)
 * s_hz  : 68x64 float shared (17408 B, stride+1 padded for bank conflicts)
 * Grid  : ((W+63)/64, (H+63)/64)
 * =========================================================================== */

#define GAUSSIAN_KSIZE 5
#define GAUSSIAN_KRAD  2

__constant__ float c_gauss_1d[5] = {
    0.11023f, 0.23680f, 0.30594f, 0.23680f, 0.11023f
};
__global__ __launch_bounds__(512)
__global__ void gaussian_coarse_4x4(const unsigned char* __restrict__ in,
                                    unsigned char*       __restrict__ out,
                                    int W, int H)
{
    const int TPX   = 2;
    const int TPY   = 4;
    const int BDX   = 32;
    const int BDY   = 16;
    const int BX    = BDX * TPX;              /* 64 */
    const int BY    = BDY * TPY;              /* 64 */
    const int TX    = BX + 2 * GAUSSIAN_KRAD; /* 68 */
    const int TY    = BY + 2 * GAUSSIAN_KRAD; /* 68 */
    const int SHZ_S = BX;                 /* 64 padded stride */

    __shared__ unsigned char s_in[TY * TX];
    __shared__ float         s_hz[TY * SHZ_S];

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int bx  = blockIdx.x * BX;
    const int by  = blockIdx.y * BY;
    const int tid = ty * BDX + tx;
    const int bs  = BDX * BDY;

    /* Step 1: cooperative 68x68 tile load (clamp-to-edge). */
    for (int i = tid; i < TX * TY; i += bs) {
        int tr = i / TX;
        int tc = i - tr * TX;
        int ir = by + tr - GAUSSIAN_KRAD;
        int ic = bx + tc - GAUSSIAN_KRAD;
        ir = max(0, min(ir, H - 1));
        ic = max(0, min(ic, W - 1));
        s_in[i] = __ldg(&in[ir * W + ic]);
    }
    __syncthreads();

    /* Step 2: cooperative horizontal blur -> s_hz (68 rows x 64 cols). */
    for (int i = tid; i < TY * BX; i += bs) {
        int hr = i / BX;
        int hc = i - hr * BX;
        float s = 0.0f;
        #pragma unroll
        for (int kx = 0; kx < GAUSSIAN_KSIZE; kx++)
            s += (float)s_in[hr * TX + hc + kx] * c_gauss_1d[kx];
        s_hz[hr * SHZ_S + hc] = s;
    }
    __syncthreads();

    /* Step 3: each thread writes 4x4 strided output pixels. */
    #pragma unroll
    for (int k = 0; k < TPY; k++) {
        int or_ = by + ty + k * BDY;
        if (or_ >= H) continue;

        #pragma unroll
        for (int l = 0; l < TPX; l++) {
            int oc = bx + tx + l * BDX;
            if (oc >= W) continue;

            float s = 0.0f;
            #pragma unroll
            for (int ky = 0; ky < GAUSSIAN_KSIZE; ky++)
                s += s_hz[(ty + k * BDY + ky) * SHZ_S + tx + l * BDX]
                     * c_gauss_1d[ky];

            out[or_ * W + oc] = (unsigned char)(s + 0.5f);
        }
    }
}

/* ===========================================================================
 * 2. Fused Sobel + NMS + double-threshold.
 * ===========================================================================
 * Block : 16x16 = 256 threads, each produces 4x4 output pixels (64x64/block).
 * Tile  : 68 rows x 20 uchar4 cols in shared (5440 B).
 *         Vectorized uchar4 loads from global; integer Sobel (no sqrtf);
 *         squared-threshold comparison avoids the sqrt entirely.
 *         Register sliding window: top/mid/bot rows kept in registers,
 *         shifted down each iteration instead of re-reading shared.
 * Output convention: 0 = suppressed, 1 = weak, 2 = strong
 * Grid  : ((W+63)/64, (H+63)/64)
 * =========================================================================== */

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)
#define OUT_TILE 64
#define IN_TILE 68
#define IN_UCHAR4_COLS 17
#define SHMEM_UCHAR4_COLS 20

__device__ __constant__ signed char nms_prev_row[4] = {0, -1, -1, -1};
__device__ __constant__ signed char nms_prev_col[4] = {-1, 1, 0, -1};
__device__ __constant__ signed char nms_next_row[4] = {0, 1, 1, 1};
__device__ __constant__ signed char nms_next_col[4] = {1, -1, 0, 1};

__device__ inline uchar4 load_clamped_uchar4(const unsigned char* src, int width, int gx, int gy)
{
    const unsigned char* row = src + gy * width;
    const int cx0 = gx < 0 ? 0 : (gx >= width ? width - 1 : gx);
    const int cx1 = (gx + 1) < 0 ? 0 : ((gx + 1) >= width ? width - 1 : (gx + 1));
    const int cx2 = (gx + 2) < 0 ? 0 : ((gx + 2) >= width ? width - 1 : (gx + 2));
    const int cx3 = (gx + 3) < 0 ? 0 : ((gx + 3) >= width ? width - 1 : (gx + 3));
    return make_uchar4(row[cx0], row[cx1], row[cx2], row[cx3]);
}

__device__ inline unsigned char compute_direction_code(int gx, int gy)
{
    const float ax = fabsf(static_cast<float>(gx));
    const float ay = fabsf(static_cast<float>(gy));
    const unsigned int is_horizontal = static_cast<unsigned int>(ay < ax * 0.41421356237f);
    const unsigned int is_vertical = static_cast<unsigned int>(ay > ax * 2.41421356237f);
    const unsigned int is_diagonal = 1u ^ (is_horizontal | is_vertical);
    const unsigned int same_sign = static_cast<unsigned int>((gx ^ gy) >= 0);
    return static_cast<unsigned char>((is_vertical << 1) | (is_diagonal * (same_sign ? 3u : 1u)));
}

__device__ inline int ring_row(int row_idx, signed char delta)
{
    return (row_idx + delta + 3) % 3;
}

__device__ inline unsigned int in_bounds_x(int x, int width)
{
    return static_cast<unsigned int>(x) < static_cast<unsigned int>(width);
}

__global__ __launch_bounds__(256)
void SobelNMSFusedKernelCorrected(const unsigned char* src,
                                  int width,
                                  int height,
                                  int high_thresh,
                                  int low_thresh,
                                  unsigned char* output)
{
    __shared__ uchar4 tile[IN_TILE * SHMEM_UCHAR4_COLS];

    const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
    const int block_input_x = blockIdx.x * OUT_TILE - 2;
    const int block_input_y = blockIdx.y * OUT_TILE - 2;

    for (int i = tid; i < IN_TILE * IN_UCHAR4_COLS; i += BLOCK_SIZE) {
        const int row = i / IN_UCHAR4_COLS;
        const int col = i % IN_UCHAR4_COLS;
        const int gy = max(0, min(block_input_y + row, height - 1));
        const int gx = block_input_x + (col << 2);

        if (gx >= 0 && (gx + 3) < width) {
            const unsigned char* row_ptr = src + gy * width + gx;
            tile[row * SHMEM_UCHAR4_COLS + col] = make_uchar4(row_ptr[0], row_ptr[1], row_ptr[2], row_ptr[3]);
        } else {
            tile[row * SHMEM_UCHAR4_COLS + col] = load_clamped_uchar4(src, width, gx, gy);
        }
    }
    __syncthreads();

    int mags[18];
    unsigned char prev_dirs[4];
    unsigned char curr_dirs[4];

    const int base_row = threadIdx.y << 2;
    const int sobel_base_x = blockIdx.x * OUT_TILE + (threadIdx.x << 2) - 1;
    const int sobel_base_y = blockIdx.y * OUT_TILE + base_row - 1;
    const int out_base_x = sobel_base_x + 1;

    const uchar4* shmem = tile;
    uchar4 top1 = shmem[base_row * SHMEM_UCHAR4_COLS + threadIdx.x];
    uchar4 top2 = shmem[base_row * SHMEM_UCHAR4_COLS + threadIdx.x + 1];
    uchar4 mid1 = shmem[(base_row + 1) * SHMEM_UCHAR4_COLS + threadIdx.x];
    uchar4 mid2 = shmem[(base_row + 1) * SHMEM_UCHAR4_COLS + threadIdx.x + 1];

    #pragma unroll
    for (int i = 0; i < 6; ++i) {
        const uchar4 bot1 = shmem[(base_row + i + 2) * SHMEM_UCHAR4_COLS + threadIdx.x];
        const uchar4 bot2 = shmem[(base_row + i + 2) * SHMEM_UCHAR4_COLS + threadIdx.x + 1];
        const int sobel_y = sobel_base_y + i;
        const bool valid_y = (sobel_y >= 0 && sobel_y < height);
        const int mag_row = (i % 3) * 6;
        int gx;
        int gy;

        gx = -int(top1.x) + int(top1.z) - (int(mid1.x) << 1) + (int(mid1.z) << 1) - int(bot1.x) + int(bot1.z);
        gy = -int(top1.x) - (int(top1.y) << 1) - int(top1.z) + int(bot1.x) + (int(bot1.y) << 1) + int(bot1.z);
        mags[mag_row + 0] = (valid_y & in_bounds_x(sobel_base_x + 0, width)) ? (gx * gx + gy * gy) : 0;

        gx = -int(top1.y) + int(top1.w) - (int(mid1.y) << 1) + (int(mid1.w) << 1) - int(bot1.y) + int(bot1.w);
        gy = -int(top1.y) - (int(top1.z) << 1) - int(top1.w) + int(bot1.y) + (int(bot1.z) << 1) + int(bot1.w);
        mags[mag_row + 1] = (valid_y & in_bounds_x(sobel_base_x + 1, width)) ? (gx * gx + gy * gy) : 0;
        if (i > 0 && i < 5) {
            curr_dirs[0] = compute_direction_code(gx, gy);
        }

        gx = -int(top1.z) + int(top2.x) - (int(mid1.z) << 1) + (int(mid2.x) << 1) - int(bot1.z) + int(bot2.x);
        gy = -int(top1.z) - (int(top1.w) << 1) - int(top2.x) + int(bot1.z) + (int(bot1.w) << 1) + int(bot2.x);
        mags[mag_row + 2] = (valid_y & in_bounds_x(sobel_base_x + 2, width)) ? (gx * gx + gy * gy) : 0;
        if (i > 0 && i < 5) {
            curr_dirs[1] = compute_direction_code(gx, gy);
        }

        gx = -int(top1.w) + int(top2.y) - (int(mid1.w) << 1) + (int(mid2.y) << 1) - int(bot1.w) + int(bot2.y);
        gy = -int(top1.w) - (int(top2.x) << 1) - int(top2.y) + int(bot1.w) + (int(bot2.x) << 1) + int(bot2.y);
        mags[mag_row + 3] = (valid_y & in_bounds_x(sobel_base_x + 3, width)) ? (gx * gx + gy * gy) : 0;
        if (i > 0 && i < 5) {
            curr_dirs[2] = compute_direction_code(gx, gy);
        }

        gx = -int(top2.x) + int(top2.z) - (int(mid2.x) << 1) + (int(mid2.z) << 1) - int(bot2.x) + int(bot2.z);
        gy = -int(top2.x) - (int(top2.y) << 1) - int(top2.z) + int(bot2.x) + (int(bot2.y) << 1) + int(bot2.z);
        mags[mag_row + 4] = (valid_y & in_bounds_x(sobel_base_x + 4, width)) ? (gx * gx + gy * gy) : 0;
        if (i > 0 && i < 5) {
            curr_dirs[3] = compute_direction_code(gx, gy);
        }

        gx = -int(top2.y) + int(top2.w) - (int(mid2.y) << 1) + (int(mid2.w) << 1) - int(bot2.y) + int(bot2.w);
        gy = -int(top2.y) - (int(top2.z) << 1) - int(top2.w) + int(bot2.y) + (int(bot2.z) << 1) + int(bot2.w);
        mags[mag_row + 5] = (valid_y & in_bounds_x(sobel_base_x + 5, width)) ? (gx * gx + gy * gy) : 0;

        if (i >= 2) {
            const int out_y = sobel_base_y + (i - 1);
            if (out_y > 0 && out_y < (height - 1)) {
                const int center_row = (i - 1) % 3;
                unsigned char* dst = output + out_y * width + out_base_x;
                unsigned char dir;
                int mag, prev_mag, next_mag;
                unsigned char is_max;
                unsigned char keep_low, keep_high;

                if (out_base_x > 0 && out_base_x < (width - 1)) {
                    dir = prev_dirs[0];
                    mag = mags[center_row * 6 + 1];
                    prev_mag =
                        mags[ring_row(center_row, nms_prev_row[dir]) * 6 +
                             1 + nms_prev_col[dir]];
                    next_mag =
                        mags[ring_row(center_row, nms_next_row[dir]) * 6 +
                             1 + nms_next_col[dir]];
                    is_max = (mag > prev_mag) && (mag >= next_mag);
                    keep_low = is_max & (mag >= low_thresh);
                    keep_high = is_max & (mag >= high_thresh);
                    dst[0] = keep_low + keep_high;
                }

                if ((out_base_x + 1) > 0 && (out_base_x + 1) < (width - 1)) {
                    dir = prev_dirs[1];
                    mag = mags[center_row * 6 + 2];
                    prev_mag =
                        mags[ring_row(center_row, nms_prev_row[dir]) * 6 +
                             2 + nms_prev_col[dir]];
                    next_mag =
                        mags[ring_row(center_row, nms_next_row[dir]) * 6 +
                             2 + nms_next_col[dir]];
                    is_max = (mag > prev_mag) && (mag >= next_mag);
                    keep_low = is_max & (mag >= low_thresh);
                    keep_high = is_max & (mag >= high_thresh);
                    dst[1] = keep_low + keep_high;
                }

                if ((out_base_x + 2) > 0 && (out_base_x + 2) < (width - 1)) {
                    dir = prev_dirs[2];
                    mag = mags[center_row * 6 + 3];
                    prev_mag =
                        mags[ring_row(center_row, nms_prev_row[dir]) * 6 +
                             3 + nms_prev_col[dir]];
                    next_mag =
                        mags[ring_row(center_row, nms_next_row[dir]) * 6 +
                             3 + nms_next_col[dir]];
                    is_max = (mag > prev_mag) && (mag >= next_mag);
                    keep_low = is_max & (mag >= low_thresh);
                    keep_high = is_max & (mag >= high_thresh);
                    dst[2] = keep_low + keep_high;
                }

                if ((out_base_x + 3) > 0 && (out_base_x + 3) < (width - 1)) {
                    dir = prev_dirs[3];
                    mag = mags[center_row * 6 + 4];
                    prev_mag =
                        mags[ring_row(center_row, nms_prev_row[dir]) * 6 +
                             4 + nms_prev_col[dir]];
                    next_mag =
                        mags[ring_row(center_row, nms_next_row[dir]) * 6 +
                             4 + nms_next_col[dir]];
                    is_max = (mag > prev_mag) && (mag >= next_mag);
                    keep_low = is_max & (mag >= low_thresh);
                    keep_high = is_max & (mag >= high_thresh);
                    dst[3] = keep_low + keep_high;
                }
            }
        }

        prev_dirs[0] = curr_dirs[0];
        prev_dirs[1] = curr_dirs[1];
        prev_dirs[2] = curr_dirs[2];
        prev_dirs[3] = curr_dirs[3];
        top1 = mid1;
        top2 = mid2;
        mid1 = bot1;
        mid2 = bot2;
    }
}

/* ===========================================================================
 * 3. Remap NMS output -> hysteresis map convention + 1-pixel border of 1s.
 *
 *   NMS out:  0 = suppressed, 1 = weak,      2 = strong
 *   Map:      0 = weak,       1 = suppressed/border, 2 = strong
 *
 * Map buffer is (W+2) x (H+2).
 * =========================================================================== */

__global__ void remap_and_border(const unsigned char* __restrict__ nms,
                                 int W, int H,
                                 unsigned char* __restrict__ map,
                                 int mapW, int mapH)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= mapW || y >= mapH) return;

    if (x == 0 || x == mapW - 1 || y == 0 || y == mapH - 1) {
        map[y * mapW + x] = 1;
        return;
    }
    unsigned char v = nms[(y - 1) * W + (x - 1)];
    map[y * mapW + x] = (v == 2) ? (unsigned char)2 : (unsigned char)(1 - v);
}

/* ===========================================================================
 * 4. CPU NUMA-aware hysteresis (OpenMP).
 * ===========================================================================
 * Per-thread local queues + CAS-dedup enqueue. proc_bind(close) packs the
 * team on one socket so the shared map stays hot in that socket's L3.
 * The only map transition is 0 -> 2; single-byte stores are atomic on x86,
 * so the CAS is only used to deduplicate frontier enqueues.
 * =========================================================================== */

static inline bool cas_u8(unsigned char* p, unsigned char expected, unsigned char desired)
{
#if defined(_MSC_VER)
    char prev = _InterlockedCompareExchange8((char*)p, (char)desired, (char)expected);
    return (unsigned char)prev == expected;
#else
    return __atomic_compare_exchange_n(p, &expected, desired, false,
                                       __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#endif
}

void hysteresis_omp_numa(unsigned char* map, int mapW, int mapH, int num_threads)
{
    if (num_threads > 0) omp_set_num_threads(num_threads);
    const int s = mapW;

    #pragma omp parallel proc_bind(close)
    {
        /* Step 1: per-thread queue allocation inside the parallel region so
           OS first-touch places the pages on the executing thread's NUMA node. */
        std::vector<unsigned char*> q;
        q.reserve(1 << 14);

        /* Step 2: seed phase — row-striped scan for strong pixels (==2),
           pushed into the thread's local frontier. */
        #pragma omp for schedule(static) nowait
        for (int y = 1; y < mapH - 1; ++y) {
            unsigned char* row = map + y * mapW;
            for (int x = 1; x < mapW - 1; ++x)
                if (row[x] == 2) q.push_back(row + x);
        }
        #pragma omp barrier

        /* Step 3: drain phase — LIFO BFS. For each popped strong pixel, try
           to promote the 8 neighbors 0 -> 2 via CAS; winners are enqueued. */
        while (!q.empty()) {
            unsigned char* m = q.back(); q.pop_back();
            if (m[-s - 1] == 0 && cas_u8(m - s - 1, 0, 2)) q.push_back(m - s - 1);
            if (m[-s    ] == 0 && cas_u8(m - s,     0, 2)) q.push_back(m - s    );
            if (m[-s + 1] == 0 && cas_u8(m - s + 1, 0, 2)) q.push_back(m - s + 1);
            if (m[-1    ] == 0 && cas_u8(m - 1,     0, 2)) q.push_back(m - 1    );
            if (m[ 1    ] == 0 && cas_u8(m + 1,     0, 2)) q.push_back(m + 1    );
            if (m[ s - 1] == 0 && cas_u8(m + s - 1, 0, 2)) q.push_back(m + s - 1);
            if (m[ s    ] == 0 && cas_u8(m + s,     0, 2)) q.push_back(m + s    );
            if (m[ s + 1] == 0 && cas_u8(m + s + 1, 0, 2)) q.push_back(m + s + 1);
        }
    }
}

/* ===========================================================================
 * 5. Final output: map interior (W+2)x(H+2) -> binary WxH. 2 -> 255, else 0.
 * =========================================================================== */

void final_output(const unsigned char* map, int mapW,
                  unsigned char* dst, int W, int H)
{
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; ++y) {
        const unsigned char* pmap = map + (y + 1) * mapW + 1;
        unsigned char* pdst = dst + y * W;
        for (int x = 0; x < W; ++x)
            pdst[x] = (pmap[x] == 2) ? (unsigned char)255 : (unsigned char)0;
    }
}

/* ===========================================================================
 * main — single-image Canny pipeline with per-stage timing.
 *
 * Usage:  canny_all.x  <image_path>  <low>  <high>  <threads>
 *
 *   Step 1: load image from disk (stb_image, grayscale).
 *   Step 2: GPU warmup (pipeline + OpenMP pool priming).
 *   Step 3: timed pipeline
 *             H2D -> Gaussian -> Sobel+NMS -> D2H -> Hysteresis -> Final
 *   Then:    write per-stage table + total to canny_timing.txt.
 * =========================================================================== */

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <image_path> <low> <high> <threads>\n", argv[0]);
        return 1;
    }
    const char* img_path    = argv[1];
    float       low_thresh  = (float)std::atof(argv[2]);
    float       high_thresh = (float)std::atof(argv[3]);
    int         h_thresh    = ((int)high_thresh) * ((int)high_thresh);
    int         l_thresh    = ((int)low_thresh)  * ((int)low_thresh);
    int         num_threads = std::atoi(argv[4]);

    /* =================================================================
     *  Step 1: load image
     * ================================================================= */
    int W = 0, H = 0, channels = 0;
    unsigned char* stb_pixels = stbi_load(img_path, &W, &H, &channels, 1);
    if (!stb_pixels) {
        std::fprintf(stderr, "failed to load image '%s': %s\n",
                     img_path, stbi_failure_reason());
        return 1;
    }

    size_t N    = (size_t)W * H;
    int    mapW = W + 2, mapH = H + 2;
    size_t mapN = (size_t)mapW * mapH;

    /* --- pinned host buffers --- */
    unsigned char *h_in = nullptr, *h_map = nullptr, *h_out = nullptr;
    cudaMallocHost(&h_in,  N);
    cudaMallocHost(&h_map, mapN);
    cudaMallocHost(&h_out, N);
    std::memcpy(h_in, stb_pixels, N);
    stbi_image_free(stb_pixels);

    /* --- device buffers --- */
    unsigned char *d_in = nullptr, *d_blur = nullptr,
                  *d_nms = nullptr, *d_map  = nullptr;
    cudaMalloc(&d_in,   N);
    cudaMalloc(&d_blur, N);
    cudaMalloc(&d_nms,  N);
    cudaMalloc(&d_map,  mapN);

    /* --- grid/block configs --- */
    dim3 gauss_blk(32, 16);
    dim3 gauss_grd((W + 63) / 64, (H + 63) / 64);
    dim3 nms_blk(BLOCK_X, BLOCK_Y);
    dim3 nms_grd((W + OUT_TILE - 1) / OUT_TILE,
                 (H + OUT_TILE - 1) / OUT_TILE);
    dim3 map_blk(16, 16);
    dim3 map_grd((mapW + 15) / 16, (mapH + 15) / 16);

    /* =================================================================
     *  Step 2: warmup
     * ================================================================= */
    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);
    gaussian_coarse_4x4<<<gauss_grd, gauss_blk>>>(d_in, d_blur, W, H);
    SobelNMSFusedKernelCorrected<<<nms_grd, nms_blk>>>(d_blur, W, H,
                                              h_thresh, l_thresh, d_nms);
    remap_and_border<<<map_grd, map_blk>>>(d_nms, W, H, d_map, mapW, mapH);
    cudaDeviceSynchronize();

    /* Warm up the OpenMP thread pool so we don't pay pool creation in timing. */
    {
        for (size_t i = 0; i < mapN; ++i) h_map[i] = 1;
        hysteresis_omp_numa(h_map, mapW, mapH, num_threads);
    }

    /* =================================================================
     *  Step 3: timed pipeline — averaged over NREPS runs.
     *  Each rep re-runs the full pipeline (H2D rewrites d_in, Sobel+NMS+
     *  remap rewrites d_map, D2H rewrites h_map → hysteresis sees a fresh
     *  map every iteration so 0→2 propagation is real work each time).
     * ================================================================= */
    const int NREPS = 10;

    cudaEvent_t ev_h2d0, ev_h2d1, ev_g1, ev_nms1, ev_d2h0, ev_d2h1;
    cudaEventCreate(&ev_h2d0); cudaEventCreate(&ev_h2d1);
    cudaEventCreate(&ev_g1);   cudaEventCreate(&ev_nms1);
    cudaEventCreate(&ev_d2h0); cudaEventCreate(&ev_d2h1);

    float  h2d_sum = 0.0f, gauss_sum = 0.0f, nms_sum = 0.0f, d2h_sum = 0.0f;
    double hyst_sum = 0.0, final_sum = 0.0;

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    for (int rep = 0; rep < NREPS; ++rep) {
        /* H2D */
        cudaEventRecord(ev_h2d0);
        cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);
        cudaEventRecord(ev_h2d1);

        /* Gaussian */
        gaussian_coarse_4x4<<<gauss_grd, gauss_blk>>>(d_in, d_blur, W, H);
        cudaEventRecord(ev_g1);

        /* Sobel + NMS (remap folded in — must finish before D2H) */
        SobelNMSFusedKernelCorrected<<<nms_grd, nms_blk>>>(d_blur, W, H,
                                                  h_thresh, l_thresh, d_nms);
        remap_and_border<<<map_grd, map_blk>>>(d_nms, W, H, d_map, mapW, mapH);
        cudaEventRecord(ev_nms1);

        /* D2H */
        cudaEventRecord(ev_d2h0);
        cudaMemcpy(h_map, d_map, mapN, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_d2h1);
        cudaEventSynchronize(ev_d2h1);

        float h2d_ms_r, gauss_ms_r, nms_ms_r, d2h_ms_r;
        cudaEventElapsedTime(&h2d_ms_r,   ev_h2d0, ev_h2d1);
        cudaEventElapsedTime(&gauss_ms_r, ev_h2d1, ev_g1);
        cudaEventElapsedTime(&nms_ms_r,   ev_g1,   ev_nms1);
        cudaEventElapsedTime(&d2h_ms_r,   ev_d2h0, ev_d2h1);
        h2d_sum   += h2d_ms_r;
        gauss_sum += gauss_ms_r;
        nms_sum   += nms_ms_r;
        d2h_sum   += d2h_ms_r;

        /* Hysteresis (CPU) */
        auto t_h0 = std::chrono::high_resolution_clock::now();
        hysteresis_omp_numa(h_map, mapW, mapH, num_threads);
        auto t_h1 = std::chrono::high_resolution_clock::now();
        hyst_sum +=
            std::chrono::duration<double, std::milli>(t_h1 - t_h0).count();

        /* Final output (CPU) */
        auto t_f0 = std::chrono::high_resolution_clock::now();
        final_output(h_map, mapW, h_out, W, H);
        auto t_f1 = std::chrono::high_resolution_clock::now();
        final_sum +=
            std::chrono::duration<double, std::milli>(t_f1 - t_f0).count();
    }

    auto wall_t1 = std::chrono::high_resolution_clock::now();

    float  h2d_ms   = h2d_sum   / NREPS;
    float  gauss_ms = gauss_sum / NREPS;
    float  nms_ms   = nms_sum   / NREPS;
    float  d2h_ms   = d2h_sum   / NREPS;
    double hyst_ms  = hyst_sum  / NREPS;
    double final_ms = final_sum / NREPS;
    double total_wall_ms =
        std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count()
        / NREPS;
    double total_ms = h2d_ms + gauss_ms + nms_ms + d2h_ms + hyst_ms + final_ms;

    /* =================================================================
     *  Write timing table
     * ================================================================= */
    const char* out_path = "canny_timing.txt";
    FILE* fp = std::fopen(out_path, "w");
    if (!fp) {
        std::fprintf(stderr, "failed to open %s for writing\n", out_path);
    } else {
        std::fprintf(fp, "========================================================\n");
        std::fprintf(fp, "  Canny pipeline timing — %s\n", img_path);
        std::fprintf(fp, "========================================================\n");
        std::fprintf(fp, "  Size        : %d x %d  (%.2f MP)\n",
                     W, H, (double)N / 1e6);
        std::fprintf(fp, "  Thresholds  : low=%.1f  high=%.1f\n",
                     low_thresh, high_thresh);
        std::fprintf(fp, "  CPU threads : %d\n", num_threads);
        std::fprintf(fp, "  Repetitions : %d  (timings below are per-run averages)\n",
                     NREPS);
        std::fprintf(fp, "--------------------------------------------------------\n");
        std::fprintf(fp, "  %-20s %12s\n", "Stage", "Time (ms)");
        std::fprintf(fp, "--------------------------------------------------------\n");
        std::fprintf(fp, "  %-20s %12.4f\n", "H2D",          h2d_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "Gaussian",     gauss_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "Sobel + NMS",  nms_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "D2H",          d2h_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "Hysteresis",   hyst_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "Final output", final_ms);
        std::fprintf(fp, "--------------------------------------------------------\n");
        std::fprintf(fp, "  %-20s %12.4f\n", "TOTAL (sum)",  total_ms);
        std::fprintf(fp, "  %-20s %12.4f\n", "Wall (start->end)", total_wall_ms);
        std::fprintf(fp, "========================================================\n");
        std::fclose(fp);
        std::fprintf(stdout, "Timing table written to %s\n", out_path);
    }

    /* --- write edge image for correctness comparison --- */
    const char* png_path = "canny_output.png";
    if (!stbi_write_png(png_path, W, H, 1, h_out, W)) {
        std::fprintf(stderr, "failed to write %s\n", png_path);
    } else {
        std::fprintf(stdout, "Edge image written to %s\n", png_path);
    }

    /* --- cleanup --- */
    cudaEventDestroy(ev_h2d0); cudaEventDestroy(ev_h2d1);
    cudaEventDestroy(ev_g1);   cudaEventDestroy(ev_nms1);
    cudaEventDestroy(ev_d2h0); cudaEventDestroy(ev_d2h1);
    cudaFree(d_in);   cudaFree(d_blur); cudaFree(d_nms); cudaFree(d_map);
    cudaFreeHost(h_in); cudaFreeHost(h_map); cudaFreeHost(h_out);
    return 0;
}
