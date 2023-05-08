# NERSC 10 Python DGEMM Compute Benchmark

This benchmark uses Python libraries to compute a
matrix multiplication. NumPy's `matmul` function
calls the underlying `gemm` function. 
To force `numpy.matmul` to use `gemm`, it's important
that B is not the transpose of A.

The benchmark has an option to compute the matrix
multiplication on an accelerator rather than a
CPU. The Offeror may elect to use this capability
with a "numpy-like" library. The idea is to demonstrate
good `gemm` performance while using Python libaries.

## Permitted Modifications

Offerors are permitted to modify the benchmark in the following ways:

The Offeror may elect to use the `--accelerator` option
in this benchmark. If they chose to use this option, they
may change the functions that are marked. They may not change
code below the `#// CODE BELOW THIS LINE SHOULD NOT BE MODIFIED`
line.

The Offeror may change the libraries used in this function
to an accelerator-specific library or libraries. Further,
they may change the kernel launch parameters. 

Note that any
option they chose must follow a functional form similar 
[NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
In other words, substituting an
`import numpy as xp` for `import libx as xp` is expected
to work without modification.

## Run Rules

The spirit of this benchmark is to measure `GEMM` performance
via the NumPy library or a so-called "drop-in replacement"
for NumPy on an accelerator. Solutions that circumvent these
requirements will not be accepted.

## How to Set up Environment, Run and Verify

You may install the environment via pip or conda. We have
found better performance with the MKL library, so we will
provide directions to install with MKL. The benchmark also
provides the option to use Numba and CuPy, so we will provide
installation instructions for those libraries, as well.

You can start from an existing conda installation or install
your own [miniconda](https://docs.conda.io/en/latest/miniconda.html)
or [miniforge](https://github.com/conda-forge/miniforge).

```
conda create -n py-dgemm python=3.10 -y
conda activate py-dgemm
conda install blas=*=*mkl numpy
conda install numba
conda install -c conda-forge cupy cudatoolkit=11.7
```

Note that if you choose to install cupy, you should ensure that you specify
a cudatoolkit version that matches the latest version
supported by the CUDA drivers on your system (11.7, in our case).

[NumPy installation](https://numpy.org/install/)
[Numba installation](https://numba.pydata.org/numba-doc/latest/user/installing.html)
[CuPy installation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-conda-forge)


The size of the matrix can be changed on the command line:

```
python python-dgemm.py --nsize 100
```

will execute using 100x100 block matrices. 

For testing, the number of iterations can also be changed:

```
python python-dgemm.py --niterations 3
```

For testing, the numpy configuration can be shown with:

```
python python-degemm.py --shownumpy
```

The Offeror has the option running the benchmark on an 
acclerator rather than a CPU. This can be enabled
by specified the `--accelerator` argument:

```
python python-dgemm.py --accelerator
```

The Offeror must modify the permitted parts of the code
to support their chosen type of acclerator.

## How to report

The primary FOM is "Gflop/s". Report all data printed to stdout.

### Sample output:

```
stephey@nid004235:~/py-dgemm/python-dgemm> OMP_NUM_THREADS=256 python python-dgemm.py --shownumpy
Requested Arguments:
  niterations : 10
  nsize       : 8000
  accelerator : False

blas_armpl_info:
  NOT AVAILABLE
blas_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/include']
blas_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/include']
lapack_armpl_info:
  NOT AVAILABLE
lapack_mkl_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/include']
lapack_opt_info:
    libraries = ['mkl_rt', 'pthread']
    library_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/lib']
    define_macros = [('SCIPY_MKL_H', None), ('HAVE_CBLAS', None)]
    include_dirs = ['/global/common/software/das/stephey/conda/conda_envs/test/include']
Supported SIMD extensions in this NumPy install:
    baseline = SSE,SSE2,SSE3
    found = SSSE3,SSE41,POPCNT,SSE42,AVX,F16C,FMA3,AVX2
    not found = AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
None
Preparing Matrix arrays
Memory required: 1.431 GB
Time for Array Allocation (sec): 0.000020
Time for Array Initialization (sec): 40.780
Running matmul...
Synchronization Overhead (sec): 1.43e-06
First of 10 iterations (sec): 0.675387
Running correctness test...

Correctness test: Passed
FlopCount: 1.024128e+12
Iteration (int)   Time(s) Gflop/s
First (0)         0.67539  1516.4
Last (9)          0.59715  1715.0
Best (1)          0.55232  1854.2
```

```
(cupy)stephey@nid001036:~/py-dgemm/python-dgemm> OMP_NUM_THREADS=256 python python-dgemm.py --accelerator
Requested Arguments:
  niterations : 10
  nsize       : 8000
  accelerator : True

Using accelerated numpy:
  <module 'cupy' from '/global/common/software/das/stephey/conda/conda_envs/cupy/lib/python3.10/site-packages/cupy/__init__.py'>

Preparing Matrix arrays
Memory required: 1.431 GB
Time for Array Allocation (sec): 0.472278
Time for Array Initialization (sec): 1.103
Running matmul...
Synchronization Overhead (sec): 3.10e-05
First of 10 iterations (sec): 1.658300
Running correctness test...

Correctness test: Passed
FlopCount: 1.024128e+12
Iteration (int)   Time(s) Gflop/s
First (0)         1.65830   617.6
Last (9)          0.05418 18900.7
Best (3)          0.05389 19003.1
(cupy)stephey@nid001036:~/py-dgemm/python-dgemm> 
```
