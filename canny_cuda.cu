#include "canny_cuda.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>

namespace cv
{

#define BLOCK_X 16
#define BLOCK_Y 16

__global__ void SobelMagDirNMSKernel(const unsigned char* src, int width, int height, float low_thresh, float high_thresh, unsigned char* out, bool L2gradient)
{
    __shared__ unsigned char shared_src[BLOCK_Y + 4][BLOCK_X + 4];
    __shared__ float shared_mag[BLOCK_Y + 2][BLOCK_X + 2];
    __shared__ unsigned char shared_dir[BLOCK_Y + 2][BLOCK_X + 2];

    int thread_x = threadIdx.x;
    int thread_y = threadIdx.y;

    int block_start_x = blockIdx.x * BLOCK_X;
    int block_start_y = blockIdx.y * BLOCK_Y;

    int pixel_x = block_start_x + thread_x;
    int pixel_y = block_start_y + thread_y;

    for (int tile_y = thread_y; tile_y < BLOCK_Y + 4; tile_y += BLOCK_Y)
    {
        for (int tile_x = thread_x; tile_x < BLOCK_X + 4; tile_x += BLOCK_X)
        {
            int src_x = block_start_x + tile_x - 2;
            int src_y = block_start_y + tile_y - 2;

            if (src_x < 0) src_x = 0;
            if (src_x >= width) src_x = width - 1;
            if (src_y < 0) src_y = 0;
            if (src_y >= height) src_y = height - 1;

            shared_src[tile_y][tile_x] = src[src_y * width + src_x];
        }
    }

    __syncthreads();

    for (int mag_y = thread_y; mag_y < BLOCK_Y + 2; mag_y += BLOCK_Y)
    {
        for (int mag_x = thread_x; mag_x < BLOCK_X + 2; mag_x += BLOCK_X)
        {
            int p00 = shared_src[mag_y][mag_x];
            int p01 = shared_src[mag_y][mag_x + 1];
            int p02 = shared_src[mag_y][mag_x + 2];

            int p10 = shared_src[mag_y + 1][mag_x];
            int p12 = shared_src[mag_y + 1][mag_x + 2];

            int p20 = shared_src[mag_y + 2][mag_x];
            int p21 = shared_src[mag_y + 2][mag_x + 1];
            int p22 = shared_src[mag_y + 2][mag_x + 2];

            float grad_x = float(-p00 + p02 - 2 * p10 + 2 * p12 - p20 + p22);
            float grad_y = float(-p00 - 2 * p01 - p02 + p20 + 2 * p21 + p22);

            float mag_val;
            if (L2gradient)
                mag_val = grad_x * grad_x + grad_y * grad_y;
            else
                mag_val = fabsf(grad_x) + fabsf(grad_y);

            shared_mag[mag_y][mag_x] = mag_val;

            // dir 0: left        →     right
            // dir 1: top-left    →     bottom-right
            // dir 2: top         →     bottom
            // dir 3: bottom-left →     top-right
            unsigned char dir_val = 0; 

            float abs_gx = fabsf(grad_x);
            float abs_gy = fabsf(grad_y);

            if (abs_gy < 0.41421356f * abs_gx)
                dir_val = 0;
            else if (abs_gy > 2.41421356f * abs_gx)
                dir_val = 2;
            else
                dir_val = (grad_x * grad_y < 0.0f) ? 3 : 1;

            shared_dir[mag_y][mag_x] = dir_val;
        }
    }

    __syncthreads();

    if (pixel_x >= width || pixel_y >= height)
        return;

    int cx = thread_x + 1;
    int cy = thread_y + 1;

    float center_mag = shared_mag[cy][cx];
    unsigned char center_dir = shared_dir[cy][cx];

    float nms_val = 0.0f;

    if (center_dir == 0)
    {
        float left  = shared_mag[cy][cx - 1];
        float right = shared_mag[cy][cx + 1];
        if (center_mag > left && center_mag >= right)
            nms_val = center_mag;
    }
    else if (center_dir == 1)
    {
        float tl = shared_mag[cy - 1][cx - 1];
        float br = shared_mag[cy + 1][cx + 1];
        if (center_mag > tl && center_mag > br)
            nms_val = center_mag;
    }
    else if (center_dir == 2)
    {
        float top    = shared_mag[cy - 1][cx];
        float bottom = shared_mag[cy + 1][cx];
        if (center_mag > top && center_mag >= bottom)
            nms_val = center_mag;
    }
    else
    {
        float bl  = shared_mag[cy + 1][cx - 1];
        float tr = shared_mag[cy - 1][cx + 1];
        if (center_mag > bl && center_mag > tr)
            nms_val = center_mag;
    }

    int out_idx = pixel_y * width + pixel_x;
    if (L2gradient)
    {
        if (nms_val > high_thresh * high_thresh)
            out[out_idx] = 2;
        else if (nms_val > low_thresh * low_thresh)
            out[out_idx] = 0;
        else
            out[out_idx] = 1;
    }
    else
    {
        if (nms_val > high_thresh)
            out[out_idx] = 2;
        else if (nms_val > low_thresh)
            out[out_idx] = 0;
        else
            out[out_idx] = 1;
    }
    
}

bool cuda_canny_sobel_mag_dir_nms(const Mat& src, Mat& dst, float low_thresh, float high_thresh, int aperture_size, bool L2gradient)
{
    CV_UNUSED(aperture_size);

    if (src.empty())
        return false;

    if (src.type() != CV_8UC1)
        return false;
    
    Mat src_cont = src.isContinuous() ? src : src.clone();

    int width = src_cont.cols;
    int height = src_cont.rows;

    size_t src_bytes = size_t(width) * size_t(height) * sizeof(unsigned char);
    size_t out_bytes = size_t(width) * size_t(height) * sizeof(unsigned char);

    unsigned char* d_src = nullptr;
    unsigned char* d_out = nullptr;

    if (cudaMalloc(&d_src, src_bytes) != cudaSuccess)
        return false;

    if (cudaMalloc(&d_out, out_bytes) != cudaSuccess)
    {
        cudaFree(d_src);
        return false;
    }

    if (cudaMemcpy(d_src, src_cont.data, src_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    {
        cudaFree(d_src);
        cudaFree(d_out);
        return false;
    }

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    SobelMagDirNMSKernel<<<grid, block>>>(d_src, width, height, low_thresh, high_thresh, d_out, L2gradient);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "GPU kernel time: " << ms << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    if (cudaDeviceSynchronize() != cudaSuccess)
    {
        cudaFree(d_src);
        cudaFree(d_out);
        return false;
    }

    dst.create(src.size(), CV_8U);

    if (cudaMemcpy(dst.data, d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        cudaFree(d_src);
        cudaFree(d_out);
        return false;
    }

    cudaFree(d_src);
    cudaFree(d_out);

    return true;
}

}