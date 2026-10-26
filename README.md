# Parallel Scaler
### Παράλληλη Προεπεξεργασία Μεγάλου Όγκου Δεδομένων


## Περιγραφή

Σύστημα παράλληλης κανονικοποίησης (scaling) μεγάλων αριθμητικών δεδομένων τύπου `double`. Διαβάζει έναν πίνακα **N × D** από δυαδικό αρχείο, υπολογίζει στατιστικά ανά στήλη και εφαρμόζει έναν από τους δύο μετασχηματισμούς:

- **StandardScaler**: `x' = (x - μ) / σ`
- **MinMaxScaler**: `x' = (x - min) / (max - min)`

---

## Υλοποιήσεις

| Αρχείο | Τεχνολογία | Εκτελέσιμο |
|---     |---         |---         |
| `scaler_serial.c` | Σειριακή (αναφοράς) | `scaler_serial` |
| `scaler_simd.c` | SIMD / AVX2 | `scaler_simd` |
| `scaler_openmp.c` | OpenMP | `scaler_openmp` |
| `scaler_mpi.c` | MPI | `scaler_mpi` |
| `scaler_cuda.cu` | CUDA / GPU | `scaler_cuda` |

---

## Δομή Αρχείων

```
parallel_scaler/
├── scaler_serial.c
├── scaler_simd.c
├── scaler_openmp.c
├── scaler_mpi.c
├── scaler_cuda.cu
├── generate_data.py
├── makefile
└── README.md
```

---

## Απαιτήσεις Συστήματος

- **OS**: WSL Ubuntu (ή native Linux)
- **Σειριακή / SIMD / OpenMP**: `gcc` με υποστήριξη `-march=native`
- **SIMD**: Επεξεργαστής με AVX2 (256-bit vector registers)
- **MPI**: `mpicc` (π.χ. OpenMPI ή MPICH)
- **CUDA**: `nvcc` + CUDA Toolkit, GPU με `sm_70` ή νεότερο
- **Python**: 3.x με `numpy` και `scikit-learn`

---

## Εγκατάσταση

### Python εξαρτήσεις

```bash
python3 -m pip install numpy scikit-learn
```

### MPI (αν δεν είναι εγκατεστημένο)

```bash
sudo apt-get install libopenmpi-dev openmpi-bin
```

---

## Κατασκευή (Build)

### Όλες οι εκδόσεις μαζί

```bash
make all
```

### Μεμονωμένα

```bash
make build          # Σειριακή
make build_simd     # SIMD/AVX2
make build_openmp   # OpenMP
make build_mpi      # MPI
make build_cuda     # CUDA
```


---

## Δημιουργία Δεδομένων

```bash
python3 generate_data.py \
  --samples <N> \
  --features <D> \
  --output <αρχείο.bin> \
  --dtype float64 \
  [--seed 42] \
  [--targets-output <y.bin>]
```

## Εκτέλεση

Κοινή μορφή εντολής για όλες τις εκδόσεις:

```
./scaler_* <input.bin> <output.bin> <N> <D> <mode> [num_threads]
```

| Παράμετρος | Περιγραφή |
|---         |---         |
| `input.bin` | Δυαδικό αρχείο εισόδου (raw doubles, row-major) |
| `output.bin` | Δυαδικό αρχείο εξόδου |
| `N` | Αριθμός γραμμών (δείγματα) |
| `D` | Αριθμός στηλών (χαρακτηριστικά) |
| `mode` | `standard` ή `minmax` |
| `num_threads` | Αριθμός threads/processes (μόνο OpenMP & MPI) |

### Σειριακή έκδοση

```bash
./scaler_serial data_10M_128.bin out_serial.bin 10000000 128 standard
./scaler_serial data_10M_128.bin out_serial_minmax.bin 10000000 128 minmax
```

### SIMD έκδοση

```bash
./scaler_simd data_10M_128.bin out_simd.bin 10000000 128 standard
```

### OpenMP έκδοση

```bash
./scaler_openmp data_10M_128.bin out_openmp_4.bin 10000000 128 standard 4
```

### MPI έκδοση

```bash
mpirun -np 4 ./scaler_mpi data_10M_128.bin out_mpi_4.bin 10000000 128 standard
```

### CUDA έκδοση
Για την ενεργοποίηση του CUDA compiler εκτελέστηκε:
```bash
module load nvhpc
```

```bash
./scaler_cuda data_10M_128.bin out_cuda.bin 10000000 128 standard
```

---

## Targets Makefile

### Εκτέλεση σειριακής

```bash
make run_serial_standard
make run_serial_minmax
```

### Εκτέλεση SIMD

```bash
make run_simd_standard
make run_simd_minmax
```

### Εκτέλεση OpenMP (διαφορετικά threads)

```bash
make run_openmp_1
make run_openmp_2
make run_openmp_4
make run_openmp_8
make run_openmp_16
```

### Εκτέλεση MPI (διαφορετικές διεργασίες)

```bash
make run_mpi_1
make run_mpi_2
make run_mpi_4
make run_mpi_8
make run_mpi_16
```

### Εκτέλεση CUDA

```bash
make run_cuda_standard
make run_cuda_minmax
```

---

## Μορφή Δεδομένων

- Τα αρχεία είναι **raw binary** χωρίς επικεφαλίδα ή metadata.
- Τα στοιχεία είναι τύπου `double` (8 bytes) σε **row-major** διάταξη: `x00, x01, ..., x0,D-1, x10, ...`
- Μέγεθος αρχείου: `N × D × 8` bytes.
- Οι διαστάσεις N και D δίνονται **από τη γραμμή εντολών**.

---

## Έλεγχος CUDA (αν υπάρχει GPU)

```bash
nvidia-smi       # Πληροφορίες GPU
nvcc --version   # Έκδοση CUDA compiler
```

 Για την ενεργοποίηση του CUDA compiler και των βιβλιοθηκών εκτελέστηκε:
 module load nvhpc

Το διαθέσιμο module nvhpc/26.1 αποδείχθηκε ασύμβατο με τον driver του Krylov100, καθώς χρησιμοποιεί CUDA 13.1 runtime ενώ ο driver υποστηρίζει έως CUDA 12.2. Η λύση ήταν η χρήση του nvcc που βρίσκεται απευθείας στο /usr/local/cuda-12.2/bin/nvcc, το οποίο παράγει binary συμβατό με τον εγκατεστημένο driver CUDA 12.2:

/usr/local/cuda-12.2/bin/nvcc -O3 -arch=sm_70 scaler_cuda.cu -o scaler_cuda


Εκτέλεση. Η εκτέλεση της CUDA έκδοσης έγινε επίσης στον Krylov100, όπου τα δεδομένα εισόδου μεταφέρθηκαν από το τοπικό μηχάνημα μέσω scp:

scp -r /home/xrist/parallel_scaler hpcgrp02@scgroup3.ceid.upatras.gr:~/

---
