# full_testing/Makefile — Self-contained Canny: all kernels + CPU hysteresis.
#
# Usage:
#   make           # build canny_all.x
#   make clean     # remove build artifacts + generated output files
#
# Run:
#   ./canny_all.x <image_path> <low> <high> <threads>
#   python test_canny.py --low 50 --high 150 -T 8 -n 50
#
# Windows (MSVC 14.29, from a Developer Command Prompt in this folder):
#   nvcc canny_all.cu -o canny_all.exe ^
#       -allow-unsupported-compiler -O2 -Xcompiler "/openmp /O2" ^
#       -I"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.29.30133/include"

.PHONY: all clean

# Xeon Silver 4208 host. `-march=native` picks up everything available.
CPUFLAGS = -O3 -fopenmp -march=native

# canny_all.cu pulls in stb_image.h and stb_image_write.h from this folder.
DEPS = canny_all.cu stb_image.h stb_image_write.h

all: canny_all.x

canny_all.x: $(DEPS)
	nvcc canny_all.cu -o canny_all.x -O3 -Xcompiler "$(CPUFLAGS)" --resource-usage

clean:
	rm -f canny_all.x canny_all.exe \
	      canny_timing.txt canny_output.png canny_test_results.txt
