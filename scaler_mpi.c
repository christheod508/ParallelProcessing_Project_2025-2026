#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

static double time_diff_ms(double start, double end) {
    return (end - start) * 1000.0;
}

static void compute_range(long long N, int size, int rank, long long* start_row, long long* local_N) {
    long long base = N / size;
    int rem = N % size;
    if (rank < rem) {
        *local_N = base + 1;
        *start_row = rank * (*local_N);
    } else {
        *local_N = base;
        *start_row = rem * (base + 1) + (rank - rem) * base;
    }
}

int main(int argc, char* argv[]) {
    // Initialize the MPI environment: this must be called by every process.
    MPI_Init(&argc, &argv);

    // Get the rank (unique process ID) and the total number of processes.
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        if (rank == 0) {
            fprintf(stderr, "Usage:\n");
            fprintf(stderr, "./scaler_mpi input.bin output.bin N D mode\n");
            fprintf(stderr, "mode = standard | minmax\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char* input_file  = argv[1];
    const char* output_file = argv[2];
    long long N = atoll(argv[3]);
    int D = atoi(argv[4]);
    const char* mode = argv[5];

    const long long BLOCK_ROWS = 100000; // per-process block

    long long start_row, local_N;
    compute_range(N, size, rank, &start_row, &local_N);

    // allocate local statistics
    double* local_sum = (double*)calloc(D, sizeof(double));
    double* local_sumsq = (double*)calloc(D, sizeof(double));
    double* local_min = (double*)malloc(D * sizeof(double));
    double* local_max = (double*)malloc(D * sizeof(double));
    if (!local_sum || !local_sumsq || !local_min || !local_max) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int j = 0; j < D; j++) {
        local_min[j] = DBL_MAX;
        local_max[j] = -DBL_MAX;
    }

    double t0 = MPI_Wtime();

    // Open input file using MPI I/O. Each process can read its own region of the file in parallel.
    MPI_File infile;
    if (MPI_File_open(MPI_COMM_WORLD, (char*)input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Cannot open input file %s\n", input_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Buffer for blocks
    size_t max_block_elems = (size_t)BLOCK_ROWS * D;
    double* buffer = (double*)malloc(max_block_elems * sizeof(double));
    if (!buffer) {
        fprintf(stderr, "Rank %d: Buffer allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Phase 1: read local chunk in blocks and compute local statistics
    long long rows_done = 0;
    while (rows_done < local_N) {
        long long rows_to_read = BLOCK_ROWS;
        if (rows_done + rows_to_read > local_N) rows_to_read = local_N - rows_done;
        MPI_Offset offset = ((MPI_Offset)(start_row + rows_done) * D) * sizeof(double);
        size_t elems = (size_t)rows_to_read * D;
        MPI_Status status;

        // Parallel read: each rank reads its own contiguous chunk from the global file
        // using an absolute offset, so no single process must read the entire dataset.
        MPI_File_read_at(infile, offset, buffer, (MPI_Offset)elems, MPI_DOUBLE, &status);
        int count;
        // MPI_Get_count returns the number of MPI_DOUBLE elements actually read.
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        if (count <= 0) break;
        size_t rows_read = (size_t)count / D;
        for (size_t i = 0; i < rows_read; i++) {
            for (int j = 0; j < D; j++) {
                double x = buffer[i * D + j];
                local_sum[j] += x;
                local_sumsq[j] += x * x;
                if (x < local_min[j]) local_min[j] = x;
                if (x > local_max[j]) local_max[j] = x;
            }
        }
        rows_done += rows_read;
    }

    MPI_File_close(&infile);

    // Allocate arrays for global reductions.
    // These arrays will hold the combined statistics from all processes.
    double* global_sum = (double*)calloc(D, sizeof(double));
    double* global_sumsq = (double*)calloc(D, sizeof(double));
    double* global_min = (double*)malloc(D * sizeof(double));
    double* global_max = (double*)malloc(D * sizeof(double));
    if (!global_sum || !global_sumsq || !global_min || !global_max) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Reduce local statistics to the root process (rank 0):
    // - sums and sumsqs are summed across all processes
    // - mins are reduced with MPI_MIN and maxes with MPI_MAX
    MPI_Reduce(local_sum, global_sum, D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_sumsq, global_sumsq, D, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_min, global_min, D, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(local_max, global_max, D, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double* mean = (double*)malloc(D * sizeof(double));
    double* stddev = (double*)malloc(D * sizeof(double));
    if (!mean || !stddev) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    double t1 = MPI_Wtime();

    if (rank == 0) {
        for (int j = 0; j < D; j++) {
            mean[j] = global_sum[j] / (double)N;
            double var = (global_sumsq[j] / (double)N) - (mean[j] * mean[j]);
            if (var < 0.0) var = 0.0;
            stddev[j] = sqrt(var);
            if (fabs(stddev[j]) < 1e-12) stddev[j] = 1.0;
        }
    }

    // Broadcast the final statistics from rank 0 to all processes so each process
    // can apply the same normalization formula locally.
    MPI_Bcast(mean, D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(stddev, D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_min, D, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(global_max, D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    // Phase 2: open files and write scaled output
    MPI_File outfile;
    if (MPI_File_open(MPI_COMM_WORLD, (char*)output_file, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfile) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Cannot open output file %s\n", output_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Re-open input for reading blocks again
    if (MPI_File_open(MPI_COMM_WORLD, (char*)input_file, MPI_MODE_RDONLY, MPI_INFO_NULL, &infile) != MPI_SUCCESS) {
        if (rank == 0) fprintf(stderr, "Cannot reopen input file %s\n", input_file);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    rows_done = 0;
    while (rows_done < local_N) {
        long long rows_to_read = BLOCK_ROWS;
        if (rows_done + rows_to_read > local_N) rows_to_read = local_N - rows_done;
        MPI_Offset offset = ((MPI_Offset)(start_row + rows_done) * D) * sizeof(double);
        size_t elems = (size_t)rows_to_read * D;
        MPI_Status status;

        // Read the same segment again to apply scaling locally.
        MPI_File_read_at(infile, offset, buffer, (MPI_Offset)elems, MPI_DOUBLE, &status);
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        if (count <= 0) break;
        size_t rows_read = (size_t)count / D;
        for (size_t i = 0; i < rows_read; i++) {
            for (int j = 0; j < D; j++) {
                double* x = &buffer[i * D + j];
                if (strcmp(mode, "standard") == 0) {
                    *x = (*x - mean[j]) / stddev[j];
                } else if (strcmp(mode, "minmax") == 0) {
                    double denom = global_max[j] - global_min[j];
                    if (fabs(denom) < 1e-12) denom = 1.0;
                    *x = (*x - global_min[j]) / denom;
                } else {
                    if (rank == 0) fprintf(stderr, "Unknown mode %s\n", mode);
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
        }
        // write scaled block at correct offset
        // Write the scaled block back to the output file at the same global offset.
        MPI_File_write_at(outfile, offset, buffer, (MPI_Offset)(rows_read * D), MPI_DOUBLE, &status);
        rows_done += rows_read;
    }

    MPI_File_close(&infile);
    MPI_File_close(&outfile);

    // Finalize MPI: this cleans up the MPI environment and should be called by every process.
    double t3 = MPI_Wtime();

    // Print sample statistics from rank 0
    if (rank == 0) {
        printf("\nSample statistics:\n");
        int preview_cols = D < 5 ? D : 5;
        for (int j = 0; j < preview_cols; j++) {
            printf("Column %d\n", j);
            printf("  Mean     = %g\n", mean[j]);
            printf("  Stddev   = %g\n", stddev[j]);
            printf("  Min      = %g\n", global_min[j]);
            printf("  Max      = %g\n", global_max[j]);
        }

        printf("Scaling completed successfully.\n");
        printf("\nTiming Information (ms):\n");
        printf("Phase 1 (local stats + reduce): %.2f ms\n", time_diff_ms(t0, t1));
        printf("Phase 1b (compute+bcast):     %.2f ms\n", time_diff_ms(t1, t2));
        printf("Phase 2 (scaling+IO):         %.2f ms\n", time_diff_ms(t2, t3));
        printf("Total:                        %.2f ms\n", time_diff_ms(t0, t3));
    }

    free(buffer);
    free(local_sum);
    free(local_sumsq);
    free(local_min);
    free(local_max);
    free(global_sum);
    free(global_sumsq);
    free(global_min);
    free(global_max);
    free(mean);
    free(stddev);

    MPI_Finalize();
    return 0;
}
