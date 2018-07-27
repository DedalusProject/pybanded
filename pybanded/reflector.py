"""Compound banded reflections in compressed format."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from . import reflector_kernels as kernels


class BandedReflector(spla.LinearOperator):
    """
    Class for compressed storage of compound banded reflections.

    Parameters
    ----------
    I : int
        Matrix size (shape = (I, I))
    dtype : dtype
        Matrix entry datatype
    M : int
        Individual reflector length
    N : int
        Number of reflectors

    Attributes
    ----------
    data : np.ndarray
        Reflectors in compressed format (see notes)

    Notes
    -----
    Each column corresponds to a banded reflector stored in compressed form as
        data[i, j] = v[j+i, j]
    Each individual reflection is given by
        H[i] = I - 2 * outer(v[:,i], conj(v[:,i]))
    The complete compound reflection is then given by
        H = H[0] @ H[1] @ H[2] ...

    """

    def __init__(self, I, dtype, M, N):
        self.I = I
        self.shape = (I, I)
        self.dtype = dtype
        self.M = M
        self.N = N
        # Initialize data with Fortran ordering for faster column operations
        self.data = np.zeros((M, N), dtype=dtype, order='F')

    def copy(self):
        """Create copy."""
        copy = type(self)(self.I, self.dtype, self.M, self.N)
        np.copyto(copy.data, self.data)
        return copy

    def _dense_data(self):
        """Create dense copy of reflectors."""
        I, M, N = self.I, self.M, self.N
        # Create dense reflectors
        dense_data = np.zeros((I, N), dtype=self.dtype, order='F')
        for n in range(min(N, I)):
            p = min(I-n, M)
            dense_data[n:n+p, n] = self.data[0:p, n]
        return dense_data

    def todense(self):
        """Create dense copy of full reflector matrix."""
        I, M, N = self.I, self.M, self.N
        dense_data = self._dense_data()
        # Iteratively apply reflectors
        H = np.identity(I, dtype=self.dtype)
        Id = np.identity(I, dtype=self.dtype)
        for n in range(min(N, I)):
            v = dense_data[:, n]
            H = H @ (Id - 2*np.outer(v, v.conj()))
        return H

    def _matvec(self, v):
        """Matrix-vector multiplication."""
        # Potential replacement: LAPACK
        # Ravel to handle vectors of shape (J,) and (J,1) per scipy spec
        v = v.ravel()
        I, M, N = self.I, self.M, self.N
        # Allocate output
        dtype = np.promote_types(self.dtype, v.dtype)
        out = np.zeros(I, dtype=dtype)
        # Apply kernel
        cast_data = self.data.astype(dtype, copy=False)
        cast_v = v.astype(dtype, copy=False)
        kernels.banded_ref_matvec_kernel(cast_data, cast_v, out, I, M, N)
        return out

    def _rmatvec(self, v):
        """Hermitian matrix-vector multiplication."""
        # Potential replacement: LAPACK
        # Ravel to handle vectors of shape (J,) and (J,1) per scipy spec
        v = v.ravel()
        I, M, N = self.I, self.M, self.N
        # Allocate output
        dtype = np.promote_types(self.dtype, v.dtype)
        out = np.zeros(I, dtype=dtype)
        # Apply kernel
        cast_data = self.data.astype(dtype, copy=False)
        cast_v = v.astype(dtype, copy=False)
        kernels.banded_ref_rmatvec_kernel(cast_data, cast_v, out, I, M, N)
        return out

