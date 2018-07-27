"""Cythonized kernels for reflector operations."""
# TODO: optimize by moving copies out to Python?
# TODO: optimize by ensuring contiguous data

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
    cdef int n, m, dm
    # Allocate scratch variables
    cdef double_rc src
    # Copy a to b
    for i in range(I):
        b[i] = a[i]
    # Compute output
    for n in reversed(range(min(N, I))):
        dm = min(M, I-n)
        # Compute v.x
        src = 0
        for m in range(dm):
            src = src + Q[m, n].conjugate() * b[n+m]
        # Apply reflection to x
        for m in range(dm):
            b[n+m] = b[n+m] - 2 * src * Q[m, n]


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
    cdef int n, m, dm
    # Allocate scratch variables
    cdef double_rc src
    # Copy a to b
    for i in range(I):
        b[i] = a[i]
    # Compute output
    for n in range(min(N, I)):
        dm = min(M, I-n)
        # Compute v.x
        src = 0
        for m in range(dm):
            src = src + Q[m, n].conjugate() * b[n+m]
        # Apply reflection to x
        for m in range(dm):
            b[n+m] = b[n+m] - 2 * src * Q[m, n]

