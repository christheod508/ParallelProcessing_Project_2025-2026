.PHONY: all build build_simd build_openmp build_mpi build_cuda gendata clean run run_serial run_simd run_openmp run_mpi run_cuda benchmark benchmark_all benchmark_openmp

# Default target
all: build build_simd build_openmp build_mpi build_cuda

# Build the serial C program
build: scaler_serial

scaler_serial: scaler_serial.c
	gcc -O3 -march=native scaler_serial.c -o scaler_serial -lm

# Build the SIMD C program with AVX2
build_simd: scaler_simd

scaler_simd: scaler_simd.c
	gcc -O3 -march=native -mavx2 -mfma scaler_simd.c -o scaler_simd -lm

# Build the OpenMP C program
build_openmp: scaler_openmp

scaler_openmp: scaler_openmp.c
	gcc -O3 -march=native -fopenmp scaler_openmp.c -o scaler_openmp -lm

# Build the MPI C program
build_mpi: scaler_mpi

scaler_mpi: scaler_mpi.c
	mpicc -O3 -march=native scaler_mpi.c -o scaler_mpi -lm

# Build the CUDA program
build_cuda: scaler_cuda

scaler_cuda: scaler_cuda.cu
	 /usr/local/cuda-12.2/bin/nvcc -O3 -arch=sm_70 scaler_cuda.cu -o scaler_cuda

# Generate test data
gendata:
	python3 generate_data.py --samples 10000000 --features 32 --output data_10M_32.bin --dtype float64

# Run serial version
run_serial_standard: build 
	./scaler_serial data_10M_32.bin out_serial.bin 10000000 32 standard

run_serial_minmax: build 
	./scaler_serial data_10M_32.bin out_serial_minmax.bin 10000000 32 minmax

# Run SIMD version
run_simd_standard: build_simd 
	./scaler_simd data_10M_32.bin out_simd.bin 10000000 32 standard

run_simd_minmax: build_simd
	./scaler_simd data_10M_32.bin out_simd_minmax.bin 10000000 32 minmax

# Run OpenMP version with different thread counts
run_openmp_1: build_openmp 
	./scaler_openmp data_10M_32.bin out_openmp_1.bin 10000000 32 standard 1 
	
run_openmp_2: build_openmp 
	./scaler_openmp data_10M_32.bin out_openmp_2.bin 10000000 32 standard 2

run_openmp_4: build_openmp 
	./scaler_openmp data_10M_32.bin out_openmp_4.bin 10000000 32 standard 4

run_openmp_8: build_openmp 
	./scaler_openmp data_10M_32.bin out_openmp_8.bin 10000000 32 standard 8											

run_openmp_16: build_openmp 
	./scaler_openmp data_10M_32.bin out_openmp_16.bin 10000000 32 standard 16

# Run MPI version with different process counts (adjust binary/data names as needed)
run_mpi_1: build_mpi
	mpirun -np 1 ./scaler_mpi data_10M_32.bin out_mpi_1.bin 10000000 32 standard

run_mpi_2: build_mpi
	mpirun -np 2 ./scaler_mpi data_10M_32.bin out_mpi_2.bin 10000000 32 standard

run_mpi_4: build_mpi
	mpirun -np 4 ./scaler_mpi data_10M_32.bin out_mpi_4.bin 10000000 32 standard

run_mpi_8: build_mpi
	mpirun -np 8 ./scaler_mpi data_10M_32.bin out_mpi_8.bin 10000000 32 standard

run_mpi_16: build_mpi
	mpirun -np 16 ./scaler_mpi data_10M_32.bin out_mpi_16.bin 10000000 32 standard

# Run CUDA version
run_cuda_standard: build_cuda
	./scaler_cuda data_10M_32.bin out_cuda.bin 10000000 32 standard

run_cuda_minmax: build_cuda
	./scaler_cuda data_10M_32.bin out_cuda_minmax.bin 10000000 32 minmax

# Run all and compare
run: run_serial run_simd run_openmp

# Clean build artifacts and generated data
clean:
	rm -f scaler_serial scaler_simd scaler_openmp scaler_mpi scaler_cuda out*.bin 