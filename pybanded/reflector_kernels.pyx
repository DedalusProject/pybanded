"""Cythonized kernels for reflector operations."""

import numpy as np
cimport cython


# Create fused type for double precision real and complex
ctypedef double complex double_complex
ctypedef fused double_rc:
    double
    double_complex


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef banded_ref_matvec_kernel(double_rc[::1, :] Q,
                               double_rc[:] a,
                               double_rc[:] b,
                               int I,
                               int M,
                               int N):
    """Matrix-vector multiplication for banded reflector."""
    # Allocate indeces
    cdef int n, p, di
    # Allocate scratch variables
    cdef double_rc src
    # Copy a to b
    for i in range(I):
        b[i] = a[i]
    # Compute output
    for n in reversed(range(min(N, I))):
        p = min(I-n, M)
        # Compute v.x
        src = 0
        for di in range(p):
            src = src + Q[di, n].conjugate() * b[n+di]
        # Apply reflection to x
        for di in range(p):
            b[n+di] = b[n+di] - 2 * src * Q[di, n]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef banded_ref_rmatvec_kernel(double_rc[::1, :] Q,
                                double_rc[:] a,
                                double_rc[:] b,
                                int I,
                                int M,
                                int N):
    """Hermitian matrix-vector multiplication for banded reflector."""
    # Allocate indeces
    cdef int n, p, di
    # Allocate scratch variables
    cdef double_rc src
    # Copy a to b
    for i in range(I):
        b[i] = a[i]
    # Compute output
    for n in range(min(N, I)):
        p = min(I-n, M)
        # Compute v.x
        src = 0
        for di in range(p):
            src = src + Q[di, n].conjugate() * b[n+di]
        # Apply reflection to x
        for di in range(p):
            b[n+di] = b[n+di] - 2 * src * Q[di, n]

