#include <cstdint>
#include <cuda_runtime.h>

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
                int mag, prev_mag,next_mag;
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
