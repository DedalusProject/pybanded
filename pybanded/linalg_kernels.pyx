"""Cythonized kernels for linear algebra."""

cimport cython


# Create fused type for double precision real and complex
ctypedef double complex double_complex
ctypedef fused double_rc:
    double
    double_complex


@cython.cdivision(True)
cdef double_rc csign(double_rc x):
    cdef double_rc out
    if x == 0:
        out = 1
    else:
        out = x / abs(x)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef qr_banded_kernel(double_rc[::1,:] W,
                       double_rc[::1,:] Q,
                       int I,
                       int J,
                       int M,
                       int N,
                       int WU):
    # Allocate indeces
    cdef int n, m, dm, k, xi
    # Allocate scratch variables
    cdef double sr
    cdef double_rc src, x0, u0
    # Loop over columns
    for n in range(N):
        # Determine number of rows including+below diagonal
        dm = min(M, I-n)
        # Compute norm(x)
        x0 = W[WU, n]
        sr = abs(x0)**2
        for m in range(1, dm):
            sr = sr + abs(W[WU+m, n])**2
        src = sr**0.5
        # Compute first index of Householder vector
        u0 = x0 + src * csign(x0)
        # Normalize Householder vector
        sr = sr + abs(u0)**2 - abs(x0)**2
        src = sr**(-0.5)
        Q[0, n] = u0 * src
        for m in range(1, dm):
            Q[m, n] = W[WU+m, n] * src
        # Apply reflection to remainder of W
        for k in range(n, min(n+1+WU, J)):
            xi = WU-(k-n)
            # Compute 2 * v.H @ x
            src = 0
            for m in range(dm):
                src = src + Q[m, n].conjugate() * W[xi+m, k]
            src = 2 * src
            # Apply reflection to x
            for m in range(dm):
                W[xi+m, k] = W[xi+m, k] - src * Q[m, n]

