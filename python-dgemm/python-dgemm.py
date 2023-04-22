#!/usr/bin/python3

import time
import argparse
import numpy as np
import math


#// ------------------------------------------------------- //
#// Function: create_cpu_arrays
#//
#// Vendor may modify this call to use specialized libraries
#// and/or add decorators.
#// Note: this time is reported but is not part of
#// of the official measurement.
#// ------------------------------------------------------- //
def create_cpu_arrays(nsize, A, B):
    for j in range(nsize):
        for k in range(nsize):
            A[j, k] = j*math.sin(j) + j*math.cos(k)
            B[j, k] = j*math.cos(k) + j*math.sin(j)
    return A, B


#// ------------------------------------------------------- //
#// Function: prepare_accel_config. Optional.
#//
#// Vendor may modify this call to use accelerator-specific
#// libraries (shown here: cupy and numba.cuda). Accelerator
#// kernel parameters may also be changed.
#// Note: this time is reported but is not part of
#// of the official measurement.
#// ------------------------------------------------------- //
def prepare_accel_config(nsize, A, B):
    import cupy as cp
    from numba import cuda

    xp = cp

    A = cp.zeros((nsize, nsize))
    B = cp.zeros((nsize, nsize))

    @cuda.jit
    def create_accel_arrays(nsize, A, B):
        j, k = cuda.grid(2)
        m = nsize
        if (j < m) and (k < m):
            A[j, k] = j*math.sin(j) + j*math.cos(k)
            B[j, k] = j*math.cos(k) + j*math.sin(j)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(nsize / threadsperblock[0])
    blockspergrid_y = math.ceil(nsize / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    # launch kernel
    tstart_accel_create = time.time()
    create_accel_arrays[blockspergrid, threadsperblock](nsize, A, B)
    cuda.synchronize()
    tend_accel_create = time.time()
    deltat_accel_create = tend_accel_create - tstart_accel_create
    
    return deltat_accel_create, A, B, xp


#// ------------------------------------------------------- //
#// Function: create_input_arrays
#//
#// Vendor may modify this call to use additional libraries
#// (ex: alternatives to numpy)
#// ------------------------------------------------------- //
def create_input_arrays(nsize, accelerator):
    A = np.zeros((nsize, nsize))
    B = np.zeros((nsize, nsize))
    tstart_cpu_create = time.time()
    [A, B] = create_cpu_arrays(nsize, A, B)
    tend_cpu_create = time.time()
    deltat_cpu_create = tend_cpu_create - tstart_cpu_create
    A_ref = A
    B_ref = B

    #option to use acclerator
    deltat_accel_create = None
    xp = np
    if accelerator:
        [deltat_accel_create, A, B, xp]  = prepare_accel_config(nsize, A, B)

    return deltat_cpu_create, deltat_accel_create, A, B, A_ref, B_ref, xp


#// ------------------------------------------------------- //
#// Function: matmul_loop
#//
#// Vendor may modify this call to use an alternative to
#// numpy, but the general functional form must remain the same
#// ------------------------------------------------------- //
def matmul_loop(niterations, nsize, A, B, xp):

    # preallocate memory for output
    C = xp.empty((nsize, nsize))
    
    deltat_matmul = np.zeros(niterations)
    for i in range(niterations):
        tstart_matmul = time.time()
        C = xp.matmul(A, B)
        tend_matmul = time.time()
        deltat_matmul[i] = tend_matmul - tstart_matmul
    
    # sanity check array type
    print("type of C:", type(C))

    return deltat_matmul, C


#// ------------------------------------------------------- //
#// Function: check_correctness
#//
#// Vendor may not modify this call
#//
#// ------------------------------------------------------- //    
def check_correctness(A, B, C, A_ref, B_ref, accelerator): 
    # performance timing will not be reported if correctness fails
    if accelerator:
        A_test = A.get()
        B_test = B.get()
        C_test = C.get()
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


#// ------------------------------------------------------- //
#// Function: report_performance
#//
#// Vendor may not modify this function
#// 
#// ------------------------------------------------------- //
def report_performance(niterations, nsize, deltat_cpu_create, deltat_accel_create, deltat_matmul, accelerator):
  
    if accelerator:
        print("total time for {} x {} input array creation: {} s".format(nsize, nsize, deltat_accel_create))
        total_benchmark_time = deltat_accel_create + np.sum(deltat_matmul)
    else:        
        print("total time for {} x {} input array creation: {} s".format(nsize, nsize, deltat_cpu_create))
        total_benchmark_time = deltat_cpu_create + np.sum(deltat_matmul)
    print("total benchmark time: {} s".format(total_benchmark_time))

    # prepare to print out first, last, and best 
    first = 0
    last = niterations - 1
    best = np.argmin(deltat_matmul)
    inds = [first, last, best]
    ninds = np.asarray(['first', 'last', 'best'])

    for i, ind in enumerate(inds):
        flops = (nsize**3)*2.0*niterations + nsize*3*niterations
        gflops = (flops/deltat_matmul[ind])/1.0e9
        print("{} iteration gflop/s: {}".format(ninds[i], gflops))
    

#// ------------------------------------------------------- //
#// Function: main
#//
#// Vendor may not modify this function. They can optionally
#// chose to use an acclerator via a command line argument.
#// ------------------------------------------------------- //
def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=8000, help="dimension of square matrix")
    parser.add_argument("--accelerator", required=False, action='store_true', help="option to use accelerator")
    args = parser.parse_args()
    niterations = args.niterations
    nsize = args.nsize
    accelerator = args.accelerator
    
    if accelerator:
        print("using accelerator: {}".format(accelerator))

    # create arrays on cpu and accelerator if that option was requested
    [deltat_cpu_create, deltat_accel_create, A, B, A_ref, B_ref, xp] = create_input_arrays(nsize, accelerator)
    
    # do matmul (dgemm) 
    [deltat_matmul, C] = matmul_loop(niterations, nsize, A, B, xp)

    # check against source of truth (numpy)
    check_correctness(A, B, C, A_ref, B_ref, accelerator)

    # if correctness test has passed, report performance
    report_performance(niterations, nsize, deltat_cpu_create, deltat_accel_create, deltat_matmul, accelerator)
    
if __name__ == '__main__':
    main()

