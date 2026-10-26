 #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define BLOCK_SIZE 256
#define BLOCK_ROWS 100000

static double time_diff_ms(const struct timespec* start,
                           const struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

// ============================================================================
// CUDA Kernels for Statistics Computation
// ============================================================================

/**
 * Kernel: Compute sum and sum of squares for one column
 * Each block processes BLOCK_SIZE rows, then reduces results
 */
__global__ void compute_sum_sumsq_kernel(
    const double* data,           // Input data (N * D)
    long long N,                  // Number of rows
    int D,                        // Number of columns
    int col,                      // Column to process
    double* block_sums,           // Output: sums per block
    double* block_sumsqs)         // Output: sum of squares per block
{
    __shared__ double shared_sum[BLOCK_SIZE];
    __shared__ double shared_sumsq[BLOCK_SIZE];
    
    int tidx = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    
    // Each thread processes multiple rows
    for (long long i = gidx; i < N; i += gridDim.x * blockDim.x) {
        double val = data[i * D + col];
        local_sum += val;
        local_sumsq += val * val;
    }
    
    shared_sum[tidx] = local_sum;
    shared_sumsq[tidx] = local_sumsq;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tidx < stride) {
            shared_sum[tidx] += shared_sum[tidx + stride];
            shared_sumsq[tidx] += shared_sumsq[tidx + stride];
        }
        __syncthreads();
    }
    
    // Write results
    if (tidx == 0) {
        block_sums[blockIdx.x] = shared_sum[0];
        block_sumsqs[blockIdx.x] = shared_sumsq[0];
    }
}

/**
 * Kernel: Find min and max for one column
 */
__global__ void compute_minmax_kernel(
    const double* data,
    long long N,
    int D,
    int col,
    double* block_mins,
    double* block_maxs)
{
    __shared__ double shared_min[BLOCK_SIZE];
    __shared__ double shared_max[BLOCK_SIZE];
    
    int tidx = threadIdx.x;
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    
    double local_min = DBL_MAX;
    double local_max = -DBL_MAX;
    
    // Each thread processes multiple rows
    for (long long i = gidx; i < N; i += gridDim.x * blockDim.x) {
        double val = data[i * D + col];
        if (val < local_min) local_min = val;
        if (val > local_max) local_max = val;
    }
    
    shared_min[tidx] = local_min;
    shared_max[tidx] = local_max;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tidx < stride) {
            if (shared_min[tidx + stride] < shared_min[tidx])
                shared_min[tidx] = shared_min[tidx + stride];
            if (shared_max[tidx + stride] > shared_max[tidx])
                shared_max[tidx] = shared_max[tidx + stride];
        }
        __syncthreads();
    }
    
    // Write results
    if (tidx == 0) {
        block_mins[blockIdx.x] = shared_min[0];
        block_maxs[blockIdx.x] = shared_max[0];
    }
}

/**
 * Kernel: Apply StandardScaler (z-score normalization)
 * scaling = (x - mean) / stddev
 */
__global__ void apply_standard_scaler_kernel(
    double* data,                 // Input/Output data
    long long N,
    int D,
    const double* mean,
    const double* stddev)
{
    long long row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < D) {
        double val = data[row * D + col];
        double std = stddev[col];
        
        if (std > 1e-10) {  // Avoid division by zero
            data[row * D + col] = (val - mean[col]) / std;
        } else {
            data[row * D + col] = 0.0;  // Or keep original value
        }
    }
}

/**
 * Kernel: Apply MinMaxScaler
 * scaling = (x - min) / (max - min)
 */
__global__ void apply_minmax_scaler_kernel(
    double* data,
    long long N,
    int D,
    const double* minv,
    const double* maxv)
{
    long long row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < N && col < D) {
        double val = data[row * D + col];
        double range = maxv[col] - minv[col];
        
        if (range > 1e-10) {
            data[row * D + col] = (val - minv[col]) / range;
        } else {
            data[row * D + col] = 0.0;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

void compute_statistics_gpu(
    const double* d_data,
    long long N,
    int D,
    double* h_mean,
    double* h_variance,
    double* h_minv,
    double* h_maxv)
{
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    double* d_block_sums, * d_block_sumsqs;
    double* d_block_mins, * d_block_maxs;
    
    CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_sumsqs, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_mins, num_blocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_block_maxs, num_blocks * sizeof(double)));
    
    // Compute statistics for each column
    for (int col = 0; col < D; col++) {
        // Sum and SumSq
        compute_sum_sumsq_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_data, N, D, col, d_block_sums, d_block_sumsqs);
        CUDA_CHECK(cudaGetLastError());
        
        // Reduce block results
        double* h_block_sums = (double*)malloc(num_blocks * sizeof(double));
        double* h_block_sumsqs = (double*)malloc(num_blocks * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_sumsqs, d_block_sumsqs, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        
        double total_sum = 0.0, total_sumsq = 0.0;
        for (int i = 0; i < num_blocks; i++) {
            total_sum += h_block_sums[i];
            total_sumsq += h_block_sumsqs[i];
        }
        
        h_mean[col] = total_sum / N;
        h_variance[col] = (total_sumsq / N) - (h_mean[col] * h_mean[col]);
        
        free(h_block_sums);
        free(h_block_sumsqs);
        
        // Min and Max
        compute_minmax_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_data, N, D, col, d_block_mins, d_block_maxs);
        CUDA_CHECK(cudaGetLastError());
        
        double* h_block_mins = (double*)malloc(num_blocks * sizeof(double));
        double* h_block_maxs = (double*)malloc(num_blocks * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_block_mins, d_block_mins, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_block_maxs, d_block_maxs, num_blocks * sizeof(double), cudaMemcpyDeviceToHost));
        
        h_minv[col] = h_block_mins[0];
        h_maxv[col] = h_block_maxs[0];
        for (int i = 1; i < num_blocks; i++) {
            if (h_block_mins[i] < h_minv[col]) h_minv[col] = h_block_mins[i];
            if (h_block_maxs[i] > h_maxv[col]) h_maxv[col] = h_block_maxs[i];
        }
        
        free(h_block_mins);
        free(h_block_maxs);
    }
    
    CUDA_CHECK(cudaFree(d_block_sums));
    CUDA_CHECK(cudaFree(d_block_sumsqs));
    CUDA_CHECK(cudaFree(d_block_mins));
    CUDA_CHECK(cudaFree(d_block_maxs));
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "./scaler_cuda input.bin output.bin N D mode\n");
        fprintf(stderr, "mode = standard | minmax\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    long long N = atoll(argv[3]);
    int D = atoi(argv[4]);
    const char* mode = argv[5];

    struct timespec t0, t1, t2, t3, t4;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    // Allocate host arrays
    double* h_data = (double*)malloc(N * D * sizeof(double));
    if (!h_data) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    // Read input file
    FILE* f = fopen(input_file, "rb");
    if (!f) {
        perror("fopen input");
        return 1;
    }
    size_t read_count = fread(h_data, sizeof(double), N * D, f);
    fclose(f);
    
    if ((long long)read_count != N * D) {
        fprintf(stderr, "Read error: expected %lld elements, got %zu\n", N * D, read_count);
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    // Allocate GPU memory
    double* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, N * D * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * D * sizeof(double), cudaMemcpyHostToDevice));

    clock_gettime(CLOCK_MONOTONIC, &t2);

    // Compute statistics
    double* h_mean = (double*)malloc(D * sizeof(double));
    double* h_variance = (double*)malloc(D * sizeof(double));
    double* h_minv = (double*)malloc(D * sizeof(double));
    double* h_maxv = (double*)malloc(D * sizeof(double));
    double* h_stddev = (double*)malloc(D * sizeof(double));

    compute_statistics_gpu(d_data, N, D, h_mean, h_variance, h_minv, h_maxv);

    for (int i = 0; i < D; i++) {
        h_stddev[i] = sqrt(h_variance[i]);
    }

    // Copy statistics to GPU
    double* d_mean, * d_stddev, * d_minv, * d_maxv;
    CUDA_CHECK(cudaMalloc(&d_mean, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_stddev, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minv, D * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_maxv, D * sizeof(double)));
    
    CUDA_CHECK(cudaMemcpy(d_mean, h_mean, D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stddev, h_stddev, D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_minv, h_minv, D * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_maxv, h_maxv, D * sizeof(double), cudaMemcpyHostToDevice));

    clock_gettime(CLOCK_MONOTONIC, &t3);

    // Apply scaling
    dim3 block_dim(16, 16);  // 256 threads per block
    dim3 grid_dim((N + block_dim.x - 1) / block_dim.x,
                  (D + block_dim.y - 1) / block_dim.y);

    if (strcmp(mode, "standard") == 0) {
        apply_standard_scaler_kernel<<<grid_dim, block_dim>>>(
            d_data, N, D, d_mean, d_stddev);
    } else if (strcmp(mode, "minmax") == 0) {
        apply_minmax_scaler_kernel<<<grid_dim, block_dim>>>(
            d_data, N, D, d_minv, d_maxv);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * D * sizeof(double), cudaMemcpyDeviceToHost));

    clock_gettime(CLOCK_MONOTONIC, &t4);

    // Write output
    f = fopen(output_file, "wb");
    if (!f) {
        perror("fopen output");
        return 1;
    }
    fwrite(h_data, sizeof(double), N * D, f);
    fclose(f);

    // Print timing
    printf("I/O Read Time:      %.3f ms\n", time_diff_ms(&t0, &t1));
    printf("GPU Upload Time:    %.3f ms\n", time_diff_ms(&t1, &t2));
    printf("Statistics Time:    %.3f ms\n", time_diff_ms(&t2, &t3));
    printf("Scaling Time:       %.3f ms\n", time_diff_ms(&t3, &t4));
    printf("Total Time:         %.3f ms\n", time_diff_ms(&t0, &t4));

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_stddev));
    CUDA_CHECK(cudaFree(d_minv));
    CUDA_CHECK(cudaFree(d_maxv));
    
    free(h_data);
    free(h_mean);
    free(h_variance);
    free(h_stddev);
    free(h_minv);
    free(h_maxv);

    return 0;
}
