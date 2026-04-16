#include <opencv2/opencv.hpp>
#include <iostream>
#include "canny_cuda.hpp"
#include <vector>
#include "hysteris_cpu.hpp"
#include <omp.h>
#include <cstring>
#include <chrono>

using namespace cv;

bool cuda_Canny(InputArray _src, OutputArray _dst, float low_thresh, float high_thresh, int aperture_size, bool L2gradient) {
    if (_src.empty()) return false;
    if (_src.type() != CV_8UC1) return false;
    if (aperture_size != 3) return false;

    Mat src = _src.getMat();
    _dst.create(src.size(), CV_8U);
    Mat dst = _dst.getMat();

    int W = src.cols;
    int H = src.rows;
    int mapW = W + 2;
    int mapH = H + 2;

    int num_threads = omp_get_max_threads();
    // std::cout << "Using threads: " << num_threads << std::endl;

    // auto t0 = std::chrono::high_resolution_clock::now();

    bool result = cuda_canny_sobel_mag_dir_nms(src, dst, low_thresh, high_thresh, aperture_size, L2gradient);

    // auto t1 = std::chrono::high_resolution_clock::now();

    if (!result) return false;

    std::vector<unsigned char> map(mapW * mapH, 1);

    for (int y = 0; y < H; y++)
    {
        memcpy(map.data() + (y+1)*mapW + 1, dst.ptr<uchar>(y), W);
    }

    // auto t2 = std::chrono::high_resolution_clock::now();

    hysteresis_omp_numa(map.data(), mapW, mapH, num_threads);
    // hysteresis_omp_numa(map.data(), mapW, mapH, 16);

    // auto t3 = std::chrono::high_resolution_clock::now();

    // hysteresis_scalar(map.data(), mapW, mapH);
    final_output(map.data(), mapW, dst.data, W, H);

    // auto t4 = std::chrono::high_resolution_clock::now();

    // std::cout << "[Timing] " 
    // << "GPU: " << std::chrono::duration<double, std::milli>(t1 - t0).count()
    // << "ms | memcpy: " << std::chrono::duration<double, std::milli>(t2 - t1).count()
    // << "ms | hyst: " << std::chrono::duration<double, std::milli>(t3 - t2).count()
    // << "ms | final: " << std::chrono::duration<double, std::milli>(t4 - t3).count()
    // << " ms\n";

    return result;
}