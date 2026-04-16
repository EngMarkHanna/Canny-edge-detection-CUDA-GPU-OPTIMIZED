all: test

# ---
canny_cuda.o: canny_cuda.cu
	nvcc `pkg-config --cflags opencv4` -c canny_cuda.cu -o canny_cuda.o

hysteris_cpu.o: hysteris_cpu.cpp
	g++ -std=c++17 -fopenmp -c hysteris_cpu.cpp `pkg-config --cflags opencv4` -o hysteris_cpu.o

test: test.cpp canny_cuda.cpp canny_cuda.o hysteris_cpu.o
	g++ -std=c++17 -fopenmp test.cpp canny_cuda.cpp canny_cuda.o hysteris_cpu.o \
		`pkg-config --cflags --libs opencv4` \
		-L/usr/local/cuda/lib64 -lcudart \
		-o test
# ---

# ---
# canny_cuda.o: canny_cuda.cu
# 	nvcc `pkg-config --cflags opencv4` -c canny_cuda.cu -o canny_cuda.o

# canny.o: canny.cpp
# 	g++ -c canny.cpp `pkg-config --cflags opencv4`

# # test: test.cpp canny_cuda.cpp canny_cuda.o
# test: test.cpp canny_cuda.cpp canny_cuda.o canny.o
# 	nvcc test.cpp canny_cuda.cpp canny_cuda.o canny.o\
# 		`pkg-config --cflags --libs opencv4` \
# 		-o test
# ---

clean:
	rm -f *.o test