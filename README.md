The intitial checkin includes three dgemm benchmarks, all gathered in one place.

- *APEX*: - this is the original version that was meant to target OpenMP systems.
   - The README.APEX file that should be the basis for the N10 readme.
   - The mt-dgemm.APEX.c file has examples of the style and level-of detail needed to describe what source code can be modified. (We don't have do do it this way; we could give the vendors a little more flexibility by describing they kinds of changes that we intend to allow)

- *NERSC_proxies*: derived from the APEX version. Included for completeness. I don't think there's anything that we need to pull from this. However, it does show how a C++ code would would use different dgemm interfaces.

- *python-dgemm*: the current python benchmark


Other notes.

1. We should compare the timings of
 `C = np.matmul( A, B )` vs `np.matmul(A,B)`,
  then decide which one we want to measure.

2. This was in the TRs, but we decided to move it to the README. The Offeror should state the theoretical peak performance for double-precision (64-bit) floating-point operations, describe how it is calculated, and explain any differences between the theoretical peak and the reported py-GEMM result. 
