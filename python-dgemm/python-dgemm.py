#!/usr/bin/python3

import time
import click
import numpy as np
import math
import numba
from numba import cuda

# initial sketch of python dgemm benchmark for n10 vendors

def create_input_arrays(niterations, nsize, accelerator):

    # preallocate memory for cpu arrays
    A = np.zeros((niterations, nsize, nsize))
    B = np.zeros((niterations, nsize, nsize))
    # call our numba jitted function
    tstart_cpu_create = time.time()
    [A, B] = create_cpu_arrays(niterations, nsize, A, B)
    tend_cpu_create = time.time()
    deltat_cpu_create = tend_cpu_create - tstart_cpu_create
    # our cpu arrays will be our reference arrays- the vendor cannot change the reference
    A_ref = A
    B_ref = B

    if accelerator:
        # vendor is free to adjust creation of gpu input arrays
        # choice of input library or libraries is flexible
        # vendor is also free to adjust kernel launch configuration
        #######################edit#########################
        import cupy as xp
        # TODO: figure out how we can do this import here if possible
        #from numba import cuda

        A = xp.zeros((niterations, nsize, nsize))
        B = xp.zeros((niterations, nsize, nsize))

        #32 by 32 by 32 results in a kernel launch error, maybe due to too many threads
        threadsperblock = (16, 16, 4) #will need to optimize
        blockspergrid_x = math.ceil(niterations / threadsperblock[0])
        blockspergrid_y = math.ceil(nsize / threadsperblock[1])
        blockspergrid_z = math.ceil(nsize / threadsperblock[2])
        blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
        # launch kernel
        tstart_gpu_create = time.time()
        create_gpu_arrays[blockspergrid, threadsperblock](niterations, nsize, A, B)
        cuda.synchronize()
        tend_gpu_create = time.time()
        deltat_gpu_create = tend_gpu_create - tstart_gpu_create
        #####################################################
    else:
        deltat_gpu_create = None

    return deltat_cpu_create, deltat_gpu_create, A, B, A_ref, B_ref


# these are our cpu reference arrays
# to force numpy.matmul to use gemm, it's important that B is not the transpose of A
# feedback welcome on how to make the array creation more interesting/expensive
@numba.jit()
def create_cpu_arrays(niterations, nsize, A, B):
    for i in range(niterations):
        for j in range(nsize):
            for k in range(nsize):
                A[i, j, k] = i*math.sin(j) + j*math.cos(k)
                B[i, j, k] = i*math.cos(k) + j*math.sin(j)

    return A, B

# vendor is free to adjust creation of gpu input arrays
# choice of input library or libraries and method is flexible
#####################edit#############################
@cuda.jit()
def create_gpu_arrays(niterations, nsize, A, B):
    i, j, k = cuda.grid(3)
    n = niterations
    m = nsize

    if (i < n) and (j < m) and (k < m):
        A[i, j, k] = i*math.sin(j) + j*math.cos(k)
        B[i, j, k] = i*math.cos(k) + j*math.sin(j)

######################################################


def matmul_loop(niterations, nsize, A, B, accelerator):

    xp = np
    if accelerator:
       # vendor is free to adjust target accelerator library
       # import jax as xp, import pytorch as xp, etc...
       # other syntax must remain the same
       #####################edit#########################
       import cupy as xp
       ##################################################

    # preallocate memory for output
    C = xp.empty((niterations, nsize, nsize))
    
    for i in range(niterations):
        if i == 0:
            # throw away initial iteration
            C[i,:,:] = xp.matmul(A[i,:,:], B[i,:,:])
        if i == 1:
            # start timer on second iteration
            tstart_matmul = time.time()
            C[i,:,:] = xp.matmul(A[i,:,:], B[i,:,:])
        else:
            C[i,:,:] = xp.matmul(A[i,:,:], B[i,:,:])


    # end timer    
    tend_matmul = time.time()
    deltat_matmul = tend_matmul - tstart_matmul
    
    # sanity check
    print("type of C:", type(C))
    print("size of C:", np.shape(C))

    return deltat_matmul, C
    
    
def check_correctness(A, B, C, A_ref, B_ref, accelerator):
    
    # correctness test using numpy as a reference
    # note we are checking against both the matrix inputs and the matrix outputs
    # performance timing will not be reported if correctness fails
    if accelerator:
        #if using accelerator library, it needs to moved back to the host and converted back into numpy format
        #the conversion implementation may differ for different libraries, so vendor is free to adjust
        #######################edit####################
        A_test = A.get()
        B_test = B.get()
        C_test = C.get()
        ###############################################
    else:
        A_test = A
        B_test = B
        C_test = C
    assert np.allclose(A_ref, A_test)
    assert np.allclose(B_ref, B_test)

    #treat numpy matmul as our source of truth    
    C_ref = np.matmul(A_ref, B_ref)
    assert np.allclose(C_ref, C_test)
    print("correctness test passed")
    

def report_performance(niterations, nsize, deltat_cpu_create, deltat_gpu_create, deltat_matmul, accelerator):

    if accelerator:
        print("total time for {} x {} x {} input array creation is {}".format(niterations, nsize, nsize, deltat_gpu_create))
        print("total time for {} matmul interations is {} s".format(niterations, deltat_matmul))
        total_benchmark_time = deltat_gpu_create + deltat_matmul
        print("total benchmark time is {} s".format(total_benchmark_time))
        
    else:        
        print("total time for {} x {} x {} input array creation is {}".format(niterations, nsize, nsize, deltat_cpu_create))
        print("total time for {} matmul interations is {} s".format(niterations, deltat_matmul))
        total_benchmark_time = deltat_cpu_create + deltat_matmul
        print("total benchmark time is {} s".format(total_benchmark_time))
    

@click.command()
@click.option('--niterations', default=10, type=int)
@click.option('--nsize', default=8000, type=int)
@click.option('--accelerator', default=False, type=bool)


def main(niterations, nsize, accelerator):
   
    print("using accelerator: {}".format(accelerator))
    [deltat_cpu_create, deltat_gpu_create, A, B, A_ref, B_ref] = create_input_arrays(niterations, nsize, accelerator)
    [deltat_matmul, C] = matmul_loop(niterations, nsize, A, B, accelerator)
    check_correctness(A, B, C, A_ref, B_ref, accelerator)
    report_performance(niterations, nsize, deltat_cpu_create, deltat_gpu_create, deltat_matmul, accelerator)
    
if __name__ == '__main__':
    main()

