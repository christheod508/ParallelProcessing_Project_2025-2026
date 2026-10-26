#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

static double time_diff_ms(const struct timespec* start,
                           const struct timespec* end) {
    return (end->tv_sec - start->tv_sec) * 1000.0 +
           (end->tv_nsec - start->tv_nsec) / 1e6;
}

int main(int argc, char* argv[]) {
    // Έλεγχος ορίων εισόδου: πρέπει να υπάρχουν 5 ορίσματα.
    if (argc < 6) {
        fprintf(stderr, "Usage:\n");
        fprintf(stderr, "./scaler input.bin output.bin N D mode\n");
        fprintf(stderr, "mode = standard | minmax\n");
        return 1;
    }

    // Διαβάζουμε τα ορίσματα της γραμμής εντολών.
    const char* input_file  = argv[1];
    const char* output_file = argv[2];
    long long N = atoll(argv[3]);      // αριθμός γραμμών
    int D = atoi(argv[4]);             // αριθμός χαρακτηριστικών ανά γραμμή
    const char* mode = argv[5];        // μέθοδος κανονικοποίησης

    const size_t BLOCK_ROWS = 100000;  // επεξεργασία σε μπλοκ για μικρότερη μνήμη

    // Δεσμεύουμε προσωρινό buffer για ένα μπλοκ γραμμών.
    double* buffer = (double*)malloc(BLOCK_ROWS * D * sizeof(double));
    if (!buffer) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Πίνακες για στατιστικά ανά στήλη.
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

    // Αρχικοποιούμε min/max σε αντίστροφα όρια.
    for (int j = 0; j < D; j++) {
        minv[j] = DBL_MAX;
        maxv[j] = -DBL_MAX;
    }

    // Ανοίγουμε το αρχείο εισόδου για ανάγνωση δυαδικών double.
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

    struct timespec phase1_start, phase1_end, phase2_start, phase2_end;
    clock_gettime(CLOCK_MONOTONIC, &phase1_start);

    // Πρώτη φάση: υπολογισμός μέσου, διακύμανσης, min και max για κάθε στήλη.
    long long rows_processed = 0;
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

        for (size_t i = 0; i < rows_read; i++) {
            for (int j = 0; j < D; j++) {
                double x = buffer[i * D + j];
                sum[j] += x;                 // άθροισμα
                sumsq[j] += x * x;           // άθροισμα τετραγώνων
                if (x < minv[j]) minv[j] = x; // ελάχιστη τιμή
                if (x > maxv[j]) maxv[j] = x; // μέγιστη τιμή
            }
        }

        rows_processed += rows_read;
    }
    fclose(fin);
    clock_gettime(CLOCK_MONOTONIC, &phase1_end);

    // Υπολογισμός μέσου και τυπικής απόκλισης ανά στήλη.
    for (int j = 0; j < D; j++) {
        mean[j] = sum[j] / N;
        variance[j] = (sumsq[j] / N) - (mean[j] * mean[j]);
        if (variance[j] < 0.0) {
            variance[j] = 0.0; // αποτροπή αρνητικής στρογγυλοποίησης
        }
        stddev[j] = sqrt(variance[j]);
        if (fabs(stddev[j]) < 1e-12) {
            stddev[j] = 1.0; // αποφυγή διαίρεσης με μηδέν
        }
    }

    // Εκτύπωση δείγματος στατιστικών για τις πρώτες στήλες.
    printf("\nSample statistics:\n");
    int preview_cols = D < 5 ? D : 5;
    for (int j = 0; j < preview_cols; j++) {
        printf("Column %d\n", j);
        printf("  Mean     = %g\n", mean[j]);
        printf("  Stddev   = %g\n", stddev[j]);
        printf("  Min      = %g\n", minv[j]);
        printf("  Max      = %g\n", maxv[j]);
    }

    // Δεύτερη φάση: ανοίγουμε ξανά το αρχείο εισόδου για να γράψουμε τα κανονικοποιημένα δεδομένα.
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

        for (size_t i = 0; i < rows_read; i++) {
            for (int j = 0; j < D; j++) {
                double* x = &buffer[i * D + j];
                if (strcmp(mode, "standard") == 0) {
                    // Standard scaling: (x - mean) / stddev
                    *x = (*x - mean[j]) / stddev[j];
                } else if (strcmp(mode, "minmax") == 0) {
                    // Min-Max scaling: (x - min) / (max - min)
                    double denom = maxv[j] - minv[j];
                    if (fabs(denom) < 1e-12) {
                        denom = 1.0; // αποφυγή διαίρεσης με μηδέν
                    }
                    *x = (*x - minv[j]) / denom;
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
            }
        }

        fwrite(buffer, sizeof(double), rows_read * D, fout);
        rows_processed += rows_read;
    }

    fclose(fin2);
    fclose(fout);
    clock_gettime(CLOCK_MONOTONIC, &phase2_end);

    // Απελευθέρωση μνήμης πριν την έξοδο.
    free(buffer);
    free(sum);
    free(sumsq);
    free(minv);
    free(maxv);
    free(mean);
    free(variance);
    free(stddev);

    printf("Scaling completed successfully.\n");
    printf("\nTiming Information:\n");
    printf("Phase 1 (statistics): %.2f ms\n", time_diff_ms(&phase1_start, &phase1_end));
    printf("Phase 2 (scaling):    %.2f ms\n", time_diff_ms(&phase2_start, &phase2_end));
    printf("Total:               %.2f ms\n", time_diff_ms(&phase1_start, &phase2_end));
    return 0;
}
