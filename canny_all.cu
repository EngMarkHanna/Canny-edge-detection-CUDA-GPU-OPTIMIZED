/*
 * canny_all.cu — Self-contained Canny kernels + CPU hysteresis in one file.
 *
 * Pipeline order:
 *   1. gaussian_coarse_4x4      — GPU kernel: fused separable 5x5 Gaussian
 *   2. SobelNMSFixedKernel      — GPU kernel: fused Sobel + NMS + threshold
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
 * s_hz  : 68x65 float shared (17680 B, stride+1 padded for bank conflicts)
 * Grid  : ((W+63)/64, (H+63)/64)
 * =========================================================================== */

#define GAUSSIAN_KSIZE 5
#define GAUSSIAN_KRAD  2

__constant__ float c_gauss_1d[5] = {
    0.11023f, 0.23680f, 0.30594f, 0.23680f, 0.11023f
};

__global__ void gaussian_coarse_4x4(const unsigned char* __restrict__ in,
                                    unsigned char*       __restrict__ out,
                                    int W, int H)
{
    const int TPX   = 4;
    const int TPY   = 4;
    const int BDX   = 16;
    const int BDY   = 16;
    const int BX    = BDX * TPX;              /* 64 */
    const int BY    = BDY * TPY;              /* 64 */
    const int TX    = BX + 2 * GAUSSIAN_KRAD; /* 68 */
    const int TY    = BY + 2 * GAUSSIAN_KRAD; /* 68 */
    const int SHZ_S = BX + 1;                 /* 65 padded stride */

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
 * Tile  : 68x68 unsigned char shared (4624 B)
 *         (2-pixel halo: 1 for Sobel + 1 for NMS neighbor's Sobel.)
 * Output convention: 0 = suppressed, 1 = weak, 2 = strong
 * Grid  : ((W+63)/64, (H+63)/64)
 * =========================================================================== */

#define FIXED_BLOCK_X 16
#define FIXED_BLOCK_Y 16
#define FIXED_TILE    68
#define FIXED_OUT     64

__global__ void SobelNMSFixedKernel(
        const unsigned char* __restrict__ src, int width, int height,
        float high_thresh, float low_thresh,
        unsigned char* __restrict__ output)
{
    __shared__ unsigned char tile[FIXED_TILE * FIXED_TILE];

    const int bx  = blockIdx.x * FIXED_OUT;
    const int by  = blockIdx.y * FIXED_OUT;
    const int tid = threadIdx.y * FIXED_BLOCK_X + threadIdx.x;

    /* Tile load: byte-level, clamp-to-edge replicate. */
    for (int i = tid; i < FIXED_TILE * FIXED_TILE; i += 256) {
        int tr = i / FIXED_TILE;
        int tc = i - tr * FIXED_TILE;
        int gy = by + tr - 2;
        int gx = bx + tc - 2;
        gy = max(0, min(gy, height - 1));
        gx = max(0, min(gx, width  - 1));
        tile[i] = src[gy * width + gx];
    }
    __syncthreads();

    const int base_tc = threadIdx.x * 4 + 2;   /* first output col in tile coords */
    const int base_tr = threadIdx.y * 4 + 2;   /* first output row in tile coords */

    float mag[18];                   /* 3 rows x 6 cols of magnitudes (ring). */
    int   save_dr[2][4], save_dc[2][4];

    /* Slide a 3-row window over 6 rows (ri=0..5). Output rows = ri 2..5. */
    for (int ri = 0; ri < 6; ri++) {
        const int tr   = base_tr - 1 + ri;
        const int ring = ri % 3;

        /* Sobel for 6 columns. */
        for (int ci = 0; ci < 6; ci++) {
            const int tc = base_tc - 1 + ci;

            float p00 = tile[(tr - 1) * FIXED_TILE + tc - 1];
            float p01 = tile[(tr - 1) * FIXED_TILE + tc    ];
            float p02 = tile[(tr - 1) * FIXED_TILE + tc + 1];
            float p10 = tile[ tr      * FIXED_TILE + tc - 1];
            float p12 = tile[ tr      * FIXED_TILE + tc + 1];
            float p20 = tile[(tr + 1) * FIXED_TILE + tc - 1];
            float p21 = tile[(tr + 1) * FIXED_TILE + tc    ];
            float p22 = tile[(tr + 1) * FIXED_TILE + tc + 1];

            float gx = -p00 + p02 - 2.0f*p10 + 2.0f*p12 - p20 + p22;
            float gy = -p00 - 2.0f*p01 - p02 + p20 + 2.0f*p21 + p22;

            mag[ring * 6 + ci] = sqrtf(gx * gx + gy * gy);

            if (ci >= 1 && ci <= 4 && ri >= 1 && ri <= 4) {
                float agx = fabsf(gx), agy = fabsf(gy);
                int dr, dc;
                if (agy < 0.41421356f * agx) {
                    dr = 0; dc = 1;
                } else if (agy > 2.41421356f * agx) {
                    dr = 1; dc = 0;
                } else if (gx * gy > 0.0f) {
                    dr = 1; dc = 1;
                } else {
                    dr = 1; dc = -1;
                }
                save_dr[ri & 1][ci - 1] = dr;
                save_dc[ri & 1][ci - 1] = dc;
            }
        }

        /* NMS on previous row (ri-1), once ring has 3 rows loaded. */
        if (ri >= 2) {
            const int nms_ring   = (ri - 1) % 3;
            const int ring_above = (nms_ring + 2) % 3;
            const int ring_below = ri % 3;
            const int buf        = (ri - 1) & 1;

            const int out_row  = ri - 2;
            const int global_y = by + threadIdx.y * 4 + out_row;
            const int global_x = bx + threadIdx.x * 4;

            uint32_t out_buf = 0;

            for (int ci = 1; ci <= 4; ci++) {
                int dr = save_dr[buf][ci - 1];
                int dc = save_dc[buf][ci - 1];

                float m = mag[nms_ring * 6 + ci];

                int n1_row = global_y - dr;
                int n1_col = global_x + (ci - 1) - dc;
                int n2_row = global_y + dr;
                int n2_col = global_x + (ci - 1) + dc;

                int r1 = (dr == 0 || (n1_row >= 0 && n1_row < height))
                         ? ((dr == 0) ? nms_ring : ring_above) : nms_ring;
                int c1 = (n1_col >= 0 && n1_col < width) ? (ci - dc) : ci;

                int r2 = (dr == 0 || (n2_row >= 0 && n2_row < height))
                         ? ((dr == 0) ? nms_ring : ring_below) : nms_ring;
                int c2 = (n2_col >= 0 && n2_col < width) ? (ci + dc) : ci;

                float n1 = mag[r1 * 6 + c1];
                float n2 = mag[r2 * 6 + c2];

                unsigned char val = 0;
                if (m >= n1 && m >= n2) {
                    if      (m > high_thresh) val = 2;
                    else if (m > low_thresh)  val = 1;
                }
                out_buf |= ((uint32_t)val) << ((ci - 1) * 8);
            }

            if (global_y < height) {
                int addr = global_y * width + global_x;
                if (global_x + 3 < width && (addr & 3) == 0) {
                    *reinterpret_cast<uint32_t*>(&output[addr]) = out_buf;
                } else {
                    for (int k = 0; k < 4; k++) {
                        if (global_x + k < width)
                            output[addr + k] = (unsigned char)((out_buf >> (k * 8)) & 0xFF);
                    }
                }
            }
        }
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
    dim3 gauss_blk(16, 16);
    dim3 gauss_grd((W + 63) / 64, (H + 63) / 64);
    dim3 nms_blk(FIXED_BLOCK_X, FIXED_BLOCK_Y);
    dim3 nms_grd((W + FIXED_OUT - 1) / FIXED_OUT,
                 (H + FIXED_OUT - 1) / FIXED_OUT);
    dim3 map_blk(16, 16);
    dim3 map_grd((mapW + 15) / 16, (mapH + 15) / 16);

    /* =================================================================
     *  Step 2: warmup
     * ================================================================= */
    cudaMemcpy(d_in, h_in, N, cudaMemcpyHostToDevice);
    gaussian_coarse_4x4<<<gauss_grd, gauss_blk>>>(d_in, d_blur, W, H);
    SobelNMSFixedKernel<<<nms_grd, nms_blk>>>(d_blur, W, H,
                                              high_thresh, low_thresh, d_nms);
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
        SobelNMSFixedKernel<<<nms_grd, nms_blk>>>(d_blur, W, H,
                                                  high_thresh, low_thresh, d_nms);
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
