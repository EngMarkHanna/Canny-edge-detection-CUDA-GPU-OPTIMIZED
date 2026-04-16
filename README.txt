============================================================
  full_testing — Self-contained Canny pipeline (single file)
============================================================

Contents
--------
  canny_all.cu       All pipeline stages + main:
                       1. gaussian_coarse_4x4        (GPU)
                       2. SobelNMSFixedKernel        (GPU)
                       3. hysteresis_omp_numa        (CPU, OpenMP)
                       4. final_output               (CPU)
                       5. remap_and_border           (GPU)
  stb_image.h        Header-only image loader (PNG/JPG/BMP/TIFF)
  stb_image_write.h  Header-only PNG writer
  Makefile           One target: canny_all.x
  test_canny.py      Correctness + timing sweep vs cv2.Canny
  run_canny_sequence.bash
                     Quick-run script: runs test_canny.py then
                     canny_all.x on a single 4K image (low=50,
                     high=150, threads=8).
  README.txt         This file

Generated files (after running):
  canny_all.x / .exe        Compiled binary
  canny_timing.txt          Per-stage timing table
  canny_output.png          Binary edge image (0 or 255)
  canny_test_results.txt    CSV summary from test_canny.py

============================================================
  Build
============================================================

Linux (from this folder):
    make clean
    make

  Produces:  canny_all.x

Windows (MSVC 14.29, from a Developer Command Prompt in this
folder):
    nvcc canny_all.cu -o canny_all.exe ^
        -allow-unsupported-compiler -O2 ^
        -Xcompiler "/openmp /O2" ^
        -I"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/include"

  Produces:  canny_all.exe

============================================================
  Run — single image
============================================================

    ./canny_all.x  <image_path>  <low>  <high>  <threads>

  Arguments:
    image_path  Path to input image (PNG/JPG/BMP/TIFF).
                Loaded as grayscale; W and H are read from the
                file header — you do NOT specify them.
    low         Canny low threshold  (float, e.g.  50)
    high        Canny high threshold (float, e.g. 150)
    threads     OpenMP thread count for CPU hysteresis
                (8 is the measured sweet spot on Xeon Silver 4208)

  Example:
    ./canny_all.x  ../pictures/images_2k/img_001.jpg  50  150  8

  Output:
    canny_timing.txt    Per-stage timing table + total + wall time.
    canny_output.png    Binary edge image (0 or 255, grayscale PNG).
    stdout              "Timing table written to canny_timing.txt"
                        "Edge image written to canny_output.png"

  Timing methodology:
    The pipeline runs 10 times (NREPS=10) on the same image.
    All timings in canny_timing.txt are per-run averages over
    those 10 repetitions, giving stable estimates that smooth
    out run-to-run jitter.

  The timing table contains:
    H2D, Gaussian, Sobel+NMS, D2H, Hysteresis, Final output,
    TOTAL (sum of stages), Wall (start -> end).
    All values are averages over 10 runs.

============================================================
  Correctness + timing sweep (Python driver)
============================================================

  Single image (mirrors canny_all.x — positional args):
      python test_canny.py  <image_path>  <low>  <high>  <threads>
    e.g.
      python test_canny.py  ../pictures/images_2k/img_001.jpg  50  150  8

  Folder sweep (default — both 2K and 4K folders):
      python test_canny.py  --low 50 --high 150 -T 8 -n 50

  For each image in ../pictures/images_2k and images_4k:
    1. Runs cv2.GaussianBlur(5x5, sigma=1.4) + cv2.Canny
       (L2gradient=True) 10 times (NREPS=10) and reports the
       average — matches canny_all.cu's internal loop count.
    2. Invokes ./canny_all.x on the same image (one call;
       the binary loops 10 times internally).
    3. Reads canny_output.png written by the CUDA binary.
    4. Parses canny_timing.txt to extract CUDA total time
       (already a per-run average over 10 reps).
    5. Pixel-by-pixel compares the two edge maps, prints
       match percentage.

  Both OpenCV and CUDA timings are averages over 10 runs per
  image, giving stable estimates that smooth out jitter.

  Table columns:  image, size, match%, OpenCV ms, CUDA ms, speedup.
  Per-folder averages at the bottom. Summary CSV written to
  canny_test_results.txt.

============================================================
  Clean
============================================================

    make clean

  Removes:
    canny_all.x, canny_all.exe
    canny_timing.txt, canny_output.png
    canny_test_results.txt
