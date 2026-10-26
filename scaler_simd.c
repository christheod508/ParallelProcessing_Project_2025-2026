#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <immintrin.h>
#include <time.h>

// Υπολογισμός διαφοράς χρόνου σε milliseconds
static double time_diff_ms(const struct timespec* start,
                           const struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

int main(int argc, char* argv[]) {
    // Ελέγχουμε τα ορίσματα της γραμμής εντολών.
    if (argc < 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "./scaler_simd input.bin output.bin N D mode\n");
        fprintf(stderr, "mode = standard | minmax\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    long long N = atoll(argv[3]);  // αριθμός γραμμών
    int D = atoi(argv[4]);         // αριθμός χαρακτηριστικών ανά γραμμή
    const char* mode = argv[5];    // standard ή minmax

    const size_t BLOCK_ROWS = 100000; // μέγεθος μπλοκ για streaming
    const int SIMD_WIDTH = 4;        // 4 doubles σε AVX2 256-bit

    // Buffer για ανάγνωση μπλοκ δεδομένων
    double* buffer = (double*)malloc(BLOCK_ROWS * D * sizeof(double));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Πίνακες στατιστικών για κάθε στήλη
    double* sum = (double*)calloc(D, sizeof(double));
    double* sumsq = (double*)calloc(D, sizeof(double));
    double* minv = (double*)malloc(D * sizeof(double));
    double* maxv = (double*)malloc(D * sizeof(double));
    double* mean = (double*)malloc(D * sizeof(double));
    double* variance = (double*)malloc(D * sizeof(double));
    double* stddev = (double*)malloc(D * sizeof(double));

    if (!sum || !sumsq || !minv || !maxv || !mean || !variance || !stddev) {
        fprintf(stderr, "Memory allocation failed\n");
        free(buffer);
        free(sum);
        free(sumsq);
        free(minv);
        free(maxv);
        free(mean);
        free(variance);
        free(stddev);
        return 1;
    }

    // Αρχικοποίηση min/max τιμών
    for (int j = 0; j < D; j++) {
        minv[j] = DBL_MAX;
        maxv[j] = -DBL_MAX;
    }

    // Άνοιγμα αρχείου εισόδου για διάβασμα δυαδικών doubles
    FILE* fin = fopen(input_file, "rb");
    if (!fin) {
        fprintf(stderr, "Cannot open input file\n");
        free(buffer);
        free(sum);
        free(sumsq);
        free(minv);
        free(maxv);
        free(mean);
        free(variance);
        free(stddev);
        return 1;
    }

    long long rows_processed = 0;
    struct timespec phase1_start, phase1_end, phase2_start, phase2_end;
    clock_gettime(CLOCK_MONOTONIC, &phase1_start);

    // Φάση 1: υπολογισμός στατιστικών (mean, variance, min, max)
    while (rows_processed < N) {
        long long rows_to_read = BLOCK_ROWS;
        if (rows_processed + rows_to_read > N) {
            rows_to_read = N - rows_processed;
        }

        size_t elements_to_read = (size_t)rows_to_read * D;
        size_t elements_read = fread(buffer, sizeof(double), elements_to_read, fin);
        if (elements_read == 0) break;

        size_t rows_read = elements_read / D;
        if (rows_read == 0) break;

        if (D >= SIMD_WIDTH) {
            // Χρησιμοποιούμε AVX2 για να επεξεργαστούμε 4 doubles κάθε φορά
            for (size_t i = 0; i < rows_read; i++) {
                int j = 0;
                for (; j <= D - SIMD_WIDTH; j += SIMD_WIDTH) {
                    __m256d x_vec = _mm256_loadu_pd(&buffer[i * D + j]);

                    // sum += x
                    __m256d sum_vec = _mm256_loadu_pd(&sum[j]);
                    sum_vec = _mm256_add_pd(sum_vec, x_vec);
                    _mm256_storeu_pd(&sum[j], sum_vec);

                    // sumsq += x*x
                    __m256d sumsq_vec = _mm256_loadu_pd(&sumsq[j]);
                    sumsq_vec = _mm256_fmadd_pd(x_vec, x_vec, sumsq_vec);
                    _mm256_storeu_pd(&sumsq[j], sumsq_vec);

                    // min/max γίνεται με scalar τιμές για κάθε στοιχείο
                    for (int k = 0; k < SIMD_WIDTH; k++) {
                        double x = buffer[i * D + j + k];
                        int idx = j + k;
                        if (x < minv[idx]) minv[idx] = x;
                        if (x > maxv[idx]) maxv[idx] = x;
                    }
                }

                // Τελικό scalar tail για τα υπόλοιπα στοιχεία
                for (; j < D; j++) {
                    double x = buffer[i * D + j];
                    sum[j] += x;
                    sumsq[j] += x * x;
                    if (x < minv[j]) minv[j] = x;
                    if (x > maxv[j]) maxv[j] = x;
                }
            }
        } else {
            // Αν ο αριθμός χαρακτηριστικών είναι μικρότερος από 4, πάμε scalar
            for (size_t i = 0; i < rows_read; i++) {
                for (int j = 0; j < D; j++) {
                    double x = buffer[i * D + j];
                    sum[j] += x;
                    sumsq[j] += x * x;
                    if (x < minv[j]) minv[j] = x;
                    if (x > maxv[j]) maxv[j] = x;
                }
            }
        }

        rows_processed += rows_read;
    }

    fclose(fin);
    clock_gettime(CLOCK_MONOTONIC, &phase1_end);

    // Υπολογισμός μέσου και τυπικής απόκλισης για κάθε στήλη
    for (int j = 0; j < D; j++) {
        mean[j] = sum[j] / N;
        variance[j] = (sumsq[j] / N) - (mean[j] * mean[j]);
        if (variance[j] < 0.0) {
            variance[j] = 0.0;
        }
        stddev[j] = sqrt(variance[j]);
        if (fabs(stddev[j]) < 1e-12) {
            stddev[j] = 1.0;
        }
    }

    // Εκτύπωση δειγματικών στατιστικών
    printf("\nSample statistics:\n");
    int preview_cols = D < 5 ? D : 5;
    for (int j = 0; j < preview_cols; j++) {
        printf("Column %d\n", j);
        printf("  Mean     = %g\n", mean[j]);
        printf("  Stddev   = %g\n", stddev[j]);
        printf("  Min      = %g\n", minv[j]);
        printf("  Max      = %g\n", maxv[j]);
    }

    // Φάση 2: εφαρμόζουμε το scaling και γράφουμε το αποτέλεσμα
    FILE* fin2 = fopen(input_file, "rb");
    if (!fin2) {
        fprintf(stderr, "Cannot reopen input file\n");
        free(buffer);
        free(sum);
        free(sumsq);
        free(minv);
        free(maxv);
        free(mean);
        free(variance);
        free(stddev);
        return 1;
    }

    FILE* fout = fopen(output_file, "wb");
    if (!fout) {
        fprintf(stderr, "Cannot open output file\n");
        fclose(fin2);
        free(buffer);
        free(sum);
        free(sumsq);
        free(minv);
        free(maxv);
        free(mean);
        free(variance);
        free(stddev);
        return 1;
    }

    rows_processed = 0;
    clock_gettime(CLOCK_MONOTONIC, &phase2_start);

    while (rows_processed < N) {
        long long rows_to_read = BLOCK_ROWS;
        if (rows_processed + rows_to_read > N) {
            rows_to_read = N - rows_processed;
        }

        size_t elements_to_read = (size_t)rows_to_read * D;
        size_t elements_read = fread(buffer, sizeof(double), elements_to_read, fin2);
        if (elements_read == 0) break;

        size_t rows_read = elements_read / D;
        if (rows_read == 0) break;

        if (strcmp(mode, "standard") == 0) {
            // Standard scaling: (x - mean) / stddev
            if (D >= SIMD_WIDTH) {
                for (size_t i = 0; i < rows_read; i++) {
                    int j = 0;
                    for (; j <= D - SIMD_WIDTH; j += SIMD_WIDTH) {
                        __m256d x_vec = _mm256_loadu_pd(&buffer[i * D + j]);
                        __m256d mean_vec = _mm256_loadu_pd(&mean[j]);
                        __m256d stddev_vec = _mm256_loadu_pd(&stddev[j]);

                        __m256d result = _mm256_sub_pd(x_vec, mean_vec);
                        result = _mm256_div_pd(result, stddev_vec);
                        _mm256_storeu_pd(&buffer[i * D + j], result);
                    }
                    for (; j < D; j++) {
                        buffer[i * D + j] = (buffer[i * D + j] - mean[j]) / stddev[j];
                    }
                }
            } else {
                for (size_t i = 0; i < rows_read; i++) {
                    for (int j = 0; j < D; j++) {
                        buffer[i * D + j] = (buffer[i * D + j] - mean[j]) / stddev[j];
                    }
                }
            }
        } else if (strcmp(mode, "minmax") == 0) {
            // Min-max scaling: (x - min) / (max - min)
            if (D >= SIMD_WIDTH) {
                __m256d eps = _mm256_set1_pd(1e-12);
                __m256d one = _mm256_set1_pd(1.0);
                __m256d neg_zero = _mm256_set1_pd(-0.0);
                for (size_t i = 0; i < rows_read; i++) {
                    int j = 0;
                    for (; j <= D - SIMD_WIDTH; j += SIMD_WIDTH) {
                        __m256d x_vec = _mm256_loadu_pd(&buffer[i * D + j]);
                        __m256d minv_vec = _mm256_loadu_pd(&minv[j]);
                        __m256d maxv_vec = _mm256_loadu_pd(&maxv[j]);

                        __m256d denom = _mm256_sub_pd(maxv_vec, minv_vec);
                        __m256d abs_denom = _mm256_andnot_pd(neg_zero, denom);
                        __m256d mask = _mm256_cmp_pd(abs_denom, eps, _CMP_LT_OQ);
                        denom = _mm256_blendv_pd(denom, one, mask);

                        __m256d result = _mm256_sub_pd(x_vec, minv_vec);
                        result = _mm256_div_pd(result, denom);
                        _mm256_storeu_pd(&buffer[i * D + j], result);
                    }
                    for (; j < D; j++) {
                        double denom = maxv[j] - minv[j];
                        if (fabs(denom) < 1e-12) denom = 1.0;
                        buffer[i * D + j] = (buffer[i * D + j] - minv[j]) / denom;
                    }
                }
            } else {
                for (size_t i = 0; i < rows_read; i++) {
                    for (int j = 0; j < D; j++) {
                        double denom = maxv[j] - minv[j];
                        if (fabs(denom) < 1e-12) denom = 1.0;
                        buffer[i * D + j] = (buffer[i * D + j] - minv[j]) / denom;
                    }
                }
            }
        } else {
            fprintf(stderr, "Unknown mode\n");
            fclose(fin2);
            fclose(fout);
            free(buffer);
            free(sum);
            free(sumsq);
            free(minv);
            free(maxv);
            free(mean);
            free(variance);
            free(stddev);
            return 1;
        }

        fwrite(buffer, sizeof(double), rows_read * D, fout);
        rows_processed += rows_read;
    }

    fclose(fin2);
    fclose(fout);
    clock_gettime(CLOCK_MONOTONIC, &phase2_end);

    free(buffer);
    free(sum);
    free(sumsq);
    free(minv);
    free(maxv);
    free(mean);
    free(variance);
    free(stddev);

    printf("Scaling completed successfully.\n");
    printf("\nTiming Information (SIMD version with AVX2):\n");
    printf("Phase 1 (Statistics): %.2f ms\n", time_diff_ms(&phase1_start, &phase1_end));
    printf("Phase 2 (Scaling):    %.2f ms\n", time_diff_ms(&phase2_start, &phase2_end));
    printf("Total:               %.2f ms\n", time_diff_ms(&phase1_start, &phase2_end));

    return 0;
}
