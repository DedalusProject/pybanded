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
    cdef int j, p, di, k, xi
    # Allocate scratch variables
    cdef double sr
    cdef double_rc src, x0, u0
    # Loop over columns
    for j in range(N):
        # Determine number of rows including+below diagonal
        p = M - max(0, j - (I-M))
        # Compute norm(x)
        x0 = W[WU, j]
        sr = abs(x0)**2
        for di in range(1, p):
            sr = sr + abs(W[WU+di, j])**2
        src = sr**0.5
        # Compute first index of Householder vector
        u0 = x0 + src * csign(x0)
        # Normalize Householder vector
        sr = sr + abs(u0)**2 - abs(x0)**2
        src = sr**(-0.5)
        Q[0, j] = u0 * src
        for di in range(1, p):
            Q[di, j] = W[WU+di, j] * src
        # Apply reflection to remainder of W
        for k in range(j, min(j+1+WU, J)):
            xi = WU-(k-j)
            # Compute v.x
            src = 0
            for di in range(p):
                src = src + Q[di, j].conjugate() * W[xi+di, k]
            # Apply reflection to x
            for di in range(p):
                W[xi+di, k] = W[xi+di, k] - 2 * src * Q[di, j]

