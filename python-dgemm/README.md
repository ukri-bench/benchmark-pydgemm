# NERSC 10 Python DGEMM Compute Benchmark

Benchmark Version: 1.0.0

## Benchmark Description

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
may change the `prepare_accel_config` only.

The Offeror may change the libraries used in this function
to an accelerator-specific library or libraries. Further,
they may change the kernel launch parameters. 

Note that any
option they chose must follow a similar functional form
to [NumPy matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html).
In other words, substituting an
`import numpy as xp` for `import libx as xp` is expected
to work without modification.

## Run Rules

Don't cheat

## How to Set up Environment, Run and Verify

To install required Python packages, you will need a Python3
installation.

```
python3 -m pip install numpy
```

is all that is required for the CPU-only benchmark. Other required
libraries may also be installed via `pip`.

The size of the matrix can be changed on the command line:

```
python python-dgemm.py --nsize 100
```

will execute using 100x100 block matrices. 

For testing, the number of iterations can also be changed:

```
python python-dgemm.py --niterations 3
```

The Offeror has the option running the benchmark on an 
acclerator rather than a CPU. This can be enabled
by specified the `--accelerator` argument:

```
python python-dgemm.py --accelerator
```

This option requires the Offeror to modify the `prepare_accel_config`
function.

## How to report

The primary FOM is "total gflop/s rate". Report all data printed to stdout.

### Sample output:

```
(cupy)stephey@perlmutter:login27:~/py-dgemm/python-dgemm> python python-dgemm.py
type of C: <class 'numpy.ndarray'>
correctness test passed
total time for 8000 x 8000 input array creation: 40.28472971916199 s
total benchmark time: 71.11348176002502 s
first iteration gflop/s: 3414.9207940385572
last iteration gflop/s: 3193.528395079785
best iteration gflop/s: 3635.744709581615
```

```
(cupy)stephey@perlmutter:login27:~/py-dgemm/python-dgemm> python python-dgemm.py --accelerator
using accelerator: True
type of C: <class 'cupy.ndarray'>
correctness test passed
total time for 8000 x 8000 input array creation: 0.15517330169677734 s
total benchmark time: 0.42073655128479004 s
first iteration gflop/s: 38672.182529254635
last iteration gflop/s: 144611696.85735005
best iteration gflop/s: 144611696.85735005
```
