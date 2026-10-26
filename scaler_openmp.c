#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <time.h>

// Υπολογισμός διαφοράς χρόνου σε milliseconds
static double time_diff_ms(const struct timespec* start,
                           const struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

int main(int argc, char* argv[]) {
    // Έλεγχος ορισμάτων γραμμής εντολών
    if (argc < 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "./scaler_openmp input.bin output.bin N D mode [num_threads]\n");
        fprintf(stderr, "mode = standard | minmax\n");
        return 1;
    }

    const char* input_file = argv[1];
    const char* output_file = argv[2];
    long long N = atoll(argv[3]);  // συνολικός αριθμός γραμμών
    int D = atoi(argv[4]);         // αριθμός χαρακτηριστικών ανά γραμμή
    const char* mode = argv[5];    // τύπος scaling
    int num_threads = (argc >= 7) ? atoi(argv[6]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    const size_t BLOCK_ROWS = 100000; // μέγεθος μπλοκ για ανάγνωση και επεξεργασία

    // Buffer για διαδοχική ανάγνωση του αρχείου
    double* buffer = (double*)malloc(BLOCK_ROWS * D * sizeof(double));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Στατιστικά ανά στήλη
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

    // Αρχικοποίηση ελαχίστων/μεγίστων
    for (int j = 0; j < D; j++) {
        minv[j] = DBL_MAX;
        maxv[j] = -DBL_MAX;
    }

    printf("OpenMP Configuration:\n");
    printf("  Max threads available: %d\n", omp_get_max_threads());
    printf("  Threads to use: %d\n", num_threads);
    printf("  Block size: %zu rows\n\n", BLOCK_ROWS);

    // Άνοιγμα αρχείου εισόδου
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

    // Φάση 1: υπολογισμός στατιστικών
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

        // Παράλληλη επεξεργασία των γραμμών με OpenMP
        #pragma omp parallel
        {
            // Κάθε νήμα διατηρεί τοπικά arrays για να μειώνει τις συγχρονίσεις
            double* thread_sum = (double*)calloc(D, sizeof(double));
            double* thread_sumsq = (double*)calloc(D, sizeof(double));
            double* thread_minv = (double*)malloc(D * sizeof(double));
            double* thread_maxv = (double*)malloc(D * sizeof(double));

            if (!thread_sum || !thread_sumsq || !thread_minv || !thread_maxv) {
                fprintf(stderr, "Thread memory allocation failed\n");
                free(thread_sum);
                free(thread_sumsq);
                free(thread_minv);
                free(thread_maxv);
                #pragma omp critical
                {
                    // Αν αποτύχει κάποια νήμα, προχωράμε στον τερματισμό
                }
            }

            for (int j = 0; j < D; j++) {
                thread_minv[j] = DBL_MAX;
                thread_maxv[j] = -DBL_MAX;
            }

            // Κάθε νήμα επεξεργάζεται ένα μέρος των γραμμών
            #pragma omp for schedule(static)
            for (size_t i = 0; i < rows_read; i++) {
                for (int j = 0; j < D; j++) {
                    double x = buffer[i * D + j];
                    thread_sum[j] += x;
                    thread_sumsq[j] += x * x;
                    if (x < thread_minv[j]) thread_minv[j] = x;
                    if (x > thread_maxv[j]) thread_maxv[j] = x;
                }
            }

            // Συγχώνευση των τοπικών αποτελεσμάτων στην παγκόσμια κατάσταση
            #pragma omp critical
            {
                for (int j = 0; j < D; j++) {
                    sum[j] += thread_sum[j];
                    sumsq[j] += thread_sumsq[j];
                    if (thread_minv[j] < minv[j]) minv[j] = thread_minv[j];
                    if (thread_maxv[j] > maxv[j]) maxv[j] = thread_maxv[j];
                }
            }

            free(thread_sum);
            free(thread_sumsq);
            free(thread_minv);
            free(thread_maxv);
        }

        rows_processed += rows_read;
    }

    fclose(fin);
    clock_gettime(CLOCK_MONOTONIC, &phase1_end);

    // Υπολογισμός μέσου και τυπικής απόκλισης από τα συνολικά στατιστικά
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

    printf("Sample statistics:\n");
    int preview_cols = D < 5 ? D : 5;
    for (int j = 0; j < preview_cols; j++) {
        printf("Column %d\n", j);
        printf("  Mean     = %g\n", mean[j]);
        printf("  Stddev   = %g\n", stddev[j]);
        printf("  Min      = %g\n", minv[j]);
        printf("  Max      = %g\n", maxv[j]);
    }

    // Ξανανοίγουμε το αρχείο εισόδου για τη δεύτερη φάση
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
            // Παράλληλη εφαρμογή standard scaling
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows_read; i++) {
                for (int j = 0; j < D; j++) {
                    buffer[i * D + j] = (buffer[i * D + j] - mean[j]) / stddev[j];
                }
            }
        } else if (strcmp(mode, "minmax") == 0) {
            // Παράλληλη εφαρμογή minmax scaling
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows_read; i++) {
                for (int j = 0; j < D; j++) {
                    double denom = maxv[j] - minv[j];
                    if (fabs(denom) < 1e-12) denom = 1.0;
                    buffer[i * D + j] = (buffer[i * D + j] - minv[j]) / denom;
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
    printf("\nTiming Information (OpenMP version):\n");
    printf("Phase 1 (Statistics): %.2f ms\n", time_diff_ms(&phase1_start, &phase1_end));
    printf("Phase 2 (Scaling):    %.2f ms\n", time_diff_ms(&phase2_start, &phase2_end));
    printf("Total:               %.2f ms\n", time_diff_ms(&phase1_start, &phase2_end));

    return 0;
}
