#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <cctype>
#include <dirent.h>
#include "canny_cuda.hpp"

// int main()
// {
//     cv::Mat img = cv::imread("test.jpg", 0);
//     cv::Mat edges;

//     if (img.empty())
//     {
//         std::cout << "Image load failed\n";
//         return -1;
//     }

//     auto t0 = std::chrono::high_resolution_clock::now();
//     book ok1 = cuda_Canny(img, edges, 50, 150, 3, false);
//     auto t1 = std::chrono::high_resolution_clock::now();
//     std::cout << "[Custom] GPU - SobelMagDirNMSThresh & CPU - HystersisFinalpass total: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms\n";

//     auto t2 = std::chrono::high_resolution_clock::now();
//     book ok2 = cuda_Canny(img, edges, 50, 150, 3, false);
//     auto t3 = std::chrono::high_resolution_clock::now();
//     std::cout << "[Custom] GPU - SobelMagDirNMSThresh & CPU - HystersisFinalpass total: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms\n";

//     if (!ok1 || !ok2)
//     {
//         std::cout << "cuda_Canny failed\n";
//         return -1;
//     }

//     cv::imwrite("out.png", edges);
//     std::cout << "Saved: out.png\n";

//     return 0;
// }

static bool is_image_file(const std::string& filename)
{
    size_t pos = filename.find_last_of('.');
    if (pos == std::string::npos) return false;

    std::string ext = filename.substr(pos);
    for (char& c : ext) c = static_cast<char>(std::tolower(c));

    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tif" || ext == ".tiff";
}

int main()
{
    const std::string input_dir = "./images_2k";
    const std::string custom_dir = "./out_custom_2k";
    const std::string opencv_dir = "./out_opencv_2k";

    int image_count = 0;

    double custom_total_ms = 0.0;
    double opencv_total_ms = 0.0;

    long long total_pixels = 0;
    long long total_equal_pixels = 0;
    long long total_intersection = 0;
    long long total_union = 0;

    bool warmed_up = false;

    DIR* dir = opendir(input_dir.c_str());
    if (!dir) {
        std::cerr << "Cannot open directory: " << input_dir << std::endl;
        return 1;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != NULL)
    {
        std::string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;

        if (!is_image_file(filename)) continue;

        std::string in_path = input_dir + "/" + filename;
        std::string stem = filename.substr(0, filename.find_last_of('.'));

        cv::Mat img = cv::imread(in_path, cv::IMREAD_GRAYSCALE);
        if (img.empty())
        {
            std::cout << "Skip: " << in_path << "\n";
            continue;
        }

        cv::Mat custom_edges, opencv_edges;

        if (!warmed_up)
        {

            bool ok = cuda_Canny(img, custom_edges, 50, 150, 3, false);
            if (!ok)
            {
                std::cout << "Warm-up custom failed: " << in_path << "\n";
                closedir(dir);
                return -1;
            }

            cv::setNumThreads(1);
            cv::Canny(img, opencv_edges, 50, 150, 3, false);

            warmed_up = true;
        }

        // custom
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = cuda_Canny(img, custom_edges, 50, 150, 3, false);
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!ok)
        {
            std::cout << "Custom failed: " << in_path << "\n";
            continue;
        }

        double custom_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // openCV
        cv::setNumThreads(1);
        auto t2 = std::chrono::high_resolution_clock::now();
        cv::Canny(img, opencv_edges, 50, 150, 3, false);
        auto t3 = std::chrono::high_resolution_clock::now();

        double opencv_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

        custom_total_ms += custom_ms;
        opencv_total_ms += opencv_ms;
        image_count++;

        const std::string custom_out = custom_dir + "/" + stem + "_custom.png";
        const std::string opencv_out = opencv_dir + "/" + stem + "_opencv.png";

        cv::imwrite(custom_out, custom_edges);
        cv::imwrite(opencv_out, opencv_edges);

        cv::Mat eq_mask;
        cv::compare(custom_edges, opencv_edges, eq_mask, cv::CMP_EQ);
        long long equal_pixels = cv::countNonZero(eq_mask);
        long long pixels = static_cast<long long>(img.rows) * img.cols;

        total_equal_pixels += equal_pixels;
        total_pixels += pixels;

        cv::Mat custom_bin = (custom_edges > 0);
        cv::Mat opencv_bin = (opencv_edges > 0);

        cv::Mat inter, uni;
        cv::bitwise_and(custom_bin, opencv_bin, inter);
        cv::bitwise_or(custom_bin, opencv_bin, uni);

        long long intersection = cv::countNonZero(inter);
        long long union_count = cv::countNonZero(uni);

        total_intersection += intersection;
        total_union += union_count;

        std::cout << "[" << image_count << "] " << filename << " | custom: " << custom_ms << " ms" << " | opencv: " << opencv_ms << " ms" << "\n";
    }

    closedir(dir);

    if (image_count == 0)
    {
        std::cout << "No valid images found in " << input_dir << "\n";
        return 0;
    }

    double custom_avg = custom_total_ms / image_count;
    double opencv_avg = opencv_total_ms / image_count;
    double pixel_accuracy = (total_pixels > 0) ? (100.0 * static_cast<double>(total_equal_pixels) / static_cast<double>(total_pixels)) : 0.0;
    double iou = (total_union > 0) ? (100.0 * static_cast<double>(total_intersection) / static_cast<double>(total_union)) : 100.0;

    std::cout << "\n========= Result =========\n";
    std::cout << "Images processed: " << image_count << "\n";
    std::cout << "Custom avg time: " << custom_avg << "\n";
    std::cout << "OpenCV avg time: " << opencv_avg << "\n";
    std::cout << "Pixel accuracy: " << pixel_accuracy << "\n";
    std::cout << "IoU: " << iou << "\n";

    return 0;
}