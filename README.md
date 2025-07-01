# Python DGEMM Compute Benchmark

**Note:** This benchmark/repository is closely based on the one used for the [NERSC-10 benchmarks](https://www.nersc.gov/systems/nersc-10/benchmarks/)

This benchmark uses matrix multiplication to measure a processor's sustainable rate of double-precision floating point operations. A secondary goal is to demonstrate good
performance when using Python language; python-dgemm uses the NumPy library as an interface to the underlying `DGEMM` function. Drop-in replacements for NumPy may be used
to run the benchmark on GPUs or other accelerators.

## Status

Stable

## Maintainers

- @aturner-epcc ([https://github.com/aturner-epcc](https://github.com/aturner-epcc))

## Overview

### Software

- [https://github.com/UoB-HPC/BabelStream](https://github.com/UoB-HPC/BabelStream)

### Architectures

- CPU: x86, Arm
- GPU: NVIDIA, AMD, Intel

### Languages and programming models

- Programming languages: Python
- Parallel models: OpenMP
- Accelerator offload models: CUDA, HIP

## Building the benchmark

The benchmark consists of a simple Python script so no explicit build is required. You will need
a working Python 3 and associated NumPy (or equivalent) installation. We provide an example way
to install these requirements below but this should not be taken as the optimal approach for the
system you are running on.

We plan to provide a Spack build package in the future and will add Spack install instructions.

When used in procurement, certain modifications to the script are allowed to optimise performance. 
Allowed modifications are described in more detail below.

### Manual installation example

#### Python & NumPy

Python (>=3.10) and NumPy are easily installed using [`conda`](https://docs.conda.io/en/latest/), as shown below. Alternatively, these packages may be installed using `pip` or from source.

You can start from an existing conda installation or install your own [miniconda](https://docs.conda.io/en/latest/miniconda.html) or [miniforge](https://github.com/conda-forge/miniforge).

```
conda create -n py-dgemm python=3.10 -y
conda activate py-dgemm
conda install "blas=*=*mkl" numpy
```

The preceding example illustrates selection of the MKL as the BLAS library underlying NumPy. Other BLAS libraries may be configured as described in the [NumPy User Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/packages.html#installing-numpy-with-blas-variants)

#### NumPy Replacements (optional)

The use of an hardware-specific NumPy replacement package is optional. Selection and installation of a suitable package is highly architecture specific. This example is based on the use of CuPy.

The  [CuPy](https://cupy.dev/) package is  imported on lines [python-dgemm.py:8-12](python-dgemm.py#L8). These lines should be un-commented and modified to import the appropriate accelerated NumPy package.

The following commands will add CuPy to the conda environment:

```
conda install -c conda-forge cupy cudatoolkit=11.7
```

Note that if you choose to install CuPy, you should ensure to specify a cudatoolkit version that matches the latest version supported by the CUDA drivers on your system (11.7, in our example).

#### Installation Help

Additional guidance for each of the aforementioned packages can be found in their respective installation guides.

- [NumPy installation](https://numpy.org/install/)
- [CuPy installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)

## Running the benchmark

At the moment, only manual run instructions are available. We plan to add ReFrame
run instructions in the future.

### Running manually

The `python-dgemm` benchmark accepts several command line options:

```bash
> ./python-dgemm.py --help
usage: python-dgemm.py [-h] [--niterations NITERATIONS] [--nsize NSIZE] [--accelerator] [--shownumpy]

optional arguments:
  -h, --help            show this help message and exit
  --niterations NITERATIONS
                        number of iterations
  --nsize NSIZE         dimension of square matrix
  --accelerator         option to use accelerator
  --shownumpy           show numpy configuration
```

Note that use of the `--accelerator` option requires modification of the code as described below in
"Allowed modifications" section to support the chosen type of accelerator.

The following example multiplies 4096 x 4096 matrices.

```
# Using NumPy on a 64-core CPU
> export OMP_NUM_THREADS=64
> python python-dgemm.py --nsize 4096

# Offloaded DGEMM to accelerator
> python python-dgemm.py --nsize 4096 --accelerator
```

No external validation tests are required; `python-dgemm.py` includes a validation test at the end of its execution.

We supply example job submission scripts:

- [ARCHER2 CPU (AMD 7742), Slurm](archer2-cpu-amd7742.slurm )

## Reporting results

The primary figure of merit (FOM) is performance (Gflop/s) of the Best iteration. Report all data printed to stdout.

## Example performance data

### ARCHER2

```
# ARCHER2 output
Running on 1 nodes, 1 processes per node, 128 threads per process
srun --hint=nomultithread --distribution=block:block python3 python-dgemm.py --nsize 8192
Requested Arguments:
  niterations : 10
  nsize       : 8192
  accelerator : False
  testseed    : None
Preparing Matrix arrays
Memory required: 1.500 GB
Time for Array Allocation (sec): 0.000029
Time for Array Initialization (sec): 120.867
Running matmul...
Synchronization Overhead (sec): 3.34e-06
First of 10 iterations (sec): 0.493110
Running correctness test...

Correctness test: Passed
FlopCount: 1.099646e+12
Iteration (int)   Time(s) Gflop/s
First (0)         0.49311  2230.0
Last (9)          0.49970  2200.6
Best (6)          0.46417  2369.0
```

### Allowed Modifications

All code modifications to `python-dgemm.py` must be confined to the region above the line `#// CODE BELOW THIS LINE SHOULD NOT BE MODIFIED`

Drop-in replacements for NumPy (e.g. [CuPy](https://cupy.dev)) may used.

Acceptable replacements must

- be loaded using the standard Python import command (`import NumpyReplacement`).
- use the same matrix multiplication interface used by [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).

The underlying dgemm function may be substituted using other libraries, and kernel launch parameters may be modified.

If a NumPy replacement is used, then it should be imported using the `xp` alias: `import NumpyReplacement as xp`. The runtime option `--accelerator` will be used to switch between NumPy and `xp`.

Custom implementations of the matrix initialization function (`initialize_accel_arrays`) are permitted. 

Function decoration, tuned kernel launch parameters, and JIT modules(e.g. [Numba](https://numba.pydata.org/) ) may be used to optimize the matrix initialization.
The formulas used to initialize the matrices (i.e. `A[j, k] = j*math.sin(j) + k*math.cos(k)` ) may not be changed.

Double precision (FP64) must be used throughout the benchmark.

The matrix size to be used for the performance test may be tuned to maximize performance. It can be adjusted without modifying the source code by using the `--nsize` command line argument

## License

This benchmark description and associated files are released under the MIT license.

## References

- NERSC-10 benchmark suite, accessed 3 Jul 2024, [https://gitlab.com/NERSC/N10-benchmarks/py-dgemm](https://gitlab.com/NERSC/N10-benchmarks/py-dgemm)
