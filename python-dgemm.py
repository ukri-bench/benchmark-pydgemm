#!/usr/bin/env python3

import time
import argparse
import numpy as np
import math

#accelerated python modules should be loaded here
#substitute your own!
#accelerated-numpy should be renamed "ap"
import cupy 
from numba import cuda
ap = cupy


#// -----
#// Function: numpy_initializer
#// Switch between numpy(np) or accelerated-numpy(ap).
#// If needed, device or library initialization
#// should be added to this function
#// -----
def numpy_initializer(show_numpy):
    if accelerator:
        try: 
            xp = ap
            print("Using accelerated numpy:\n  {}\n".format( ap ) )
        except:
            print("The --accelerator option was used, but no accelerated numpy (e.g. cupy) was found.")
            exit(1)
    else:
        xp = np
        if show_numpy:
            print(np.show_config())
    return xp

#// -----
#// Function: synchronize_host_accel
#// this is a no-op if running on the host
#// May be modified for non-cuda
#// -----
def synchronize_host_accel():
    if accelerator:
        cupy.cuda.runtime.deviceSynchronize()
        

#// -----
#// Function: initialize_accel_arrays
#// Initialize matrices using accelerator memory/processor
#// Here, numba.cuda is used to run this custome kernel on 
#// May be modified for non-cuda or non-numba
#// -----
def initialize_accel_arrays( nsize, A, B ):

    @cuda.jit
    def initialize_accel_arrays_kernel(nsize, A, B):
        j, k = cuda.grid(2)
        m = nsize
        if (j < m) and (k < m):
            A[j, k] = j*math.sin(j) + j*math.cos(k)
            B[j, k] = j*math.cos(k) + j*math.sin(j)

    threadsperblock = (32, 32)
    blockspergrid_x = math.ceil(nsize / threadsperblock[0])
    blockspergrid_y = math.ceil(nsize / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    initialize_accel_arrays_kernel[blockspergrid, threadsperblock](nsize, A, B)
    cuda.synchronize()
    
#// -----
#// CODE BELOW THIS LINE SHOULD NOT BE MODIFIED
#// -----

#// -----
#// Function: initialize_host_arrays
#// Initialize matrices using host memory/processor
#// -----
def initialize_host_arrays( nsize, A, B ):
    for j in range(nsize):
        for k in range(nsize):
            A[j, k] = j*math.sin(j) + j*math.cos(k)
            B[j, k] = j*math.cos(k) + j*math.sin(j)


#// -----
#// Function: create_arrays
#// allocate matrices and call their initialization functions
#// -----
def create_arrays(nsize, xp ):

    def memory_string( memory_bytes ):
        units = ' kMGTPX'
        iunit = int( math.log( memory_bytes, 1024 ) )
        memory_units = memory_bytes / 1024**iunit
        memory_str = "{:.3f} {}B".format( memory_units, units[iunit] )
        return memory_str

    print("Preparing Matrix arrays")
    print("Memory required: {}".format( memory_string( 3 * nsize * nsize * 8 ) ) )

    t_start = time.time()
    A = xp.zeros((nsize,nsize))
    B = xp.zeros((nsize,nsize))
    C = xp.zeros((nsize,nsize))
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Allocation (sec): {:.6f}".format( deltat ) )
    

    t_start = time.time()
    if accelerator:
        initialize_accel_arrays( nsize, A, B )
    else:
        initialize_host_arrays( nsize, A, B )
    t_end = time.time()
    deltat = t_end - t_start
    print("Time for Array Initialization (sec): {:.3f}".format( deltat ) )

    return A, B, C



        
#// ------------------------------------------------------- //
#// Function: matmul_loop
#//
#// Vendor may modify this call to use an alternative to
#// numpy, but the general functional form must remain the same
#// ------------------------------------------------------- //
def matmul_loop(niterations, A, B, C, xp ):


    print("Running matmul...")

    tstart = time.time()
    synchronize_host_accel()
    tend = time.time()
    deltat = tend - tstart
    print("Synchronization Overhead (sec): {:.2e}".format( deltat ) )
       
    deltat = np.zeros( niterations )
    for i in range(niterations):

        synchronize_host_accel()
        tstart = time.time()

        xp.matmul(A, B, out=C )

        synchronize_host_accel()
        tend = time.time()

        deltat[i] = tend - tstart

        if( i==0 ):
            print("First of {:d} iterations (sec): {:.6f}".format( niterations, deltat[0] ) )

    # sanity check array type
    #print("type of C:", type(C))

    return deltat



#// ------------------------------------------------------- //
#// Function: check_correctness
#// Sample a number (ntest) of matrix elements to compare
#// Test against pythonic dot product
#// ------------------------------------------------------- //    
def check_correctness( nsize, A, B, C ):

    print("Running correctness test...")
    
    A_test = A
    B_test = B
    C_test = C
    if accelerator:
        A_test = np.zeros((nsize, nsize))
        B_test = np.zeros((nsize, nsize))
        C_test = np.zeros((nsize, nsize))
        A_test = A.get()
        B_test = B.get()
        C_test = C.get()

    ntest = 1024
    is_correct = True
    rng = np.random.default_rng()
    for itest in range( ntest ):

        i = rng.integers( nsize, size=1 )[0]
        j = rng.integers( nsize, size=1 )[0]
        c_test = C[i,j]
        c_ref  = sum( [Aik * Bkj for Aik, Bkj in zip( A[i,:], B[:,j] ) ] )
        itest_correct = math.isclose( c_ref, c_test )
        
        if not itest_correct:
            msg = "Error Found at row {:d}, col {:d}. Expected: {:e}, Found: {:e}"
            print( msg.format( i, j, c_ref, c_test ) )
            is_correct = False

    print()
    print("Correctness test: {}".format(( "Passed" if is_correct  else "Failed")) )
    return is_correct

    
#// Function: report_performance
def report_performance(niterations, nsize, deltat_matmul ):
  

    flops = 2 * (nsize**3) + 2 * nsize**2    
    gflops = [ flops / t / 1.0e9 for t in deltat_matmul ]

    print_all_iterations = False
    if( print_all_iterations ):
        print("FlopCount: {}".format( flops ) )
        for i in range( niterations ):
            print("iter: {:2d}   time: {:.6f}   gflops: {: 7.2f}".format( i, deltat_matmul[i], gflops[i] ) )
        print("")

        
    ind = { "First":0,
            "Last":niterations-1,
            "Best":np.argmin( deltat_matmul ) }
    
    print("FlopCount: {:e}".format( flops ) )
    print("{:15s}   {:7s} {:7s}".format("Iteration (int)","Time(s)","Gflop/s"))
    for s in ["First", "Last", "Best"]:
        i = ind[s]
        si = "{:s} ({:d})".format( s, i )
        print("{:15s}   {:7.5f} {:7.1f}".format( si, deltat_matmul[i], gflops[i] ) )


#// -----
#// Function: get_args
#// Parse and print the command line arguments
#// -----
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--niterations", type=int, required=False, default=10, help="number of iterations")
    parser.add_argument("--nsize", type=int, required=False, default=8000, help="dimension of square matrix")
    parser.add_argument("--accelerator", required=False, action='store_true', help="option to use accelerator")
    parser.add_argument("--shownumpy", required=False, action='store_true', help="show numpy configuration")
    args = parser.parse_args()

    print("Requested Arguments:")
    print("  {:12s}: {}".format( "niterations", args.niterations ))
    print("  {:12s}: {}".format( "nsize",       args.nsize       ))
    print("  {:12s}: {}".format( "accelerator", args.accelerator ))
    print()
    
    return args

#// ------------------------------------------------------- //
#// Function: main
#// -----
def main():
 
    #retreive command line arguments
    args = get_args()

    #stores accelerator as a global variable
    global accelerator
    accelerator = args.accelerator
    niterations = args.niterations
    nsize       = args.nsize
    show_numpy  = args.shownumpy
    
    #choose the appropriate numpy-like interface:
    #ap for accelerators, or numpy for host
    xp = numpy_initializer( show_numpy )

    #create working arrays on the target processor ( host or accelerator )
    [ A, B, C ] = create_arrays( nsize, xp )
          
    # do matmul (dgemm) 
    deltat_matmul = matmul_loop( niterations, A, B, C, xp )

    # check against source of truth (naive matmul)
    is_correct = check_correctness( nsize, A, B, C )
    assert( is_correct )

    # if correctness test has passed, report performance
    report_performance( niterations, nsize, deltat_matmul )
    
if __name__ == '__main__':
    main()

