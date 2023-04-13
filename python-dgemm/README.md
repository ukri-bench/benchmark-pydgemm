# Python DGEMM Benchmark

Initial sketch of Python-based DGEMM benchmark for N10 vendors

## Environment set up

You can create a cupy-enabled custom conda environment via

```
salloc -N 1 -t 30 -C gpu -G 1 -A nstaff -q interactive
module load python
conda create -n cupy python=3.10 -y 
conda activate cupy
conda install numba click cupy -c conda-forge
```

## How to run

You can launch the benchmark using the defaults:

```
python python-dgemm.py
```

The problem size, number of iterations, and acclerator option can be changed:

```
python python-dgemm.py --niterations <default 10> --nsize <default 8000> --accelerator <default False>
```

## Some sample runs on Muller

GPU:

```
(/global/common/software/das/stephey/conda/conda_envs/cupy) stephey@nid001072:/mscratch/sd/s/stephey/python-benchmark/python-dgemm> python python-dgemm.py --accelerator True
using accelerator: True
type of C: <class 'cupy.ndarray'>
correctness test passed
total time for 10 x 8000 x 8000 input array creation is 0.16782903671264648
total time for 10 matmul interations is 0.0010180473327636719 s
total benchmark time is 0.16884708404541016 s
```

CPU:

```
(/global/common/software/das/stephey/conda/conda_envs/cupy) stephey@nid001072:/mscratch/sd/s/stephey/python-benchmark/python-dgemm> python python-dgemm.py --accelerator False
using accelerator: False
type of C: <class 'numpy.ndarray'>
correctness test passed
total time for 10 x 8000 x 8000 input array creation is 8.490462064743042
total time for 10 matmul interations is 12.06446361541748 s
total benchmark time is 20.554925680160522 s
```

## Open questions

I intially had the `from numba import cuda` in an `if` block, but I found that
python threw an error for the `cuda.jit` decorator. Ideally we wouldn't need
to import `numba.cuda` unless required, but for now I left it in the top 
import block.

I just picked a `math.sin` and `math.cos` based function to create the matrix
inputs. My main objective was to avoid creating A and B such that B was the
transpose of A, in which case the `numpy.matmul` function can take a shortcut.
We want to force it to use `dgemm`.  If there are more interesting or expensive
functions to try here, we can definitely change this part out. Our main
limitation is the [relatively small subset of
functionality](https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html)
supported in Numba CUDA.

I also randomly chose a kernel size for Numba CUDA of [16, 16, 4]. I think my
initial choice of [32, 32, 32] exceeded the thread limit.  There is probably a
more optimal choice, although I think it will change depending on the
accelerator hardware, and we currently do allow them to edit this part.

I am taking the total time for all iterations over `np.matmul` (aka `dgemm`)-
we may want to split this out per iteration? I'm not sure what you all prefer. 


