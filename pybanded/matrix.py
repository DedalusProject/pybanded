"""Sparse matrices in banded format."""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class BandedMatrix(spla.LinearOperator):
    """
    Class for banded matrices.

    Parameters
    ----------
    shape : tuple
        Matrix shape
    dtype : dtype
        Matrix entry datatype
    L : int
        Number of subdiagonals
    U : int
        Number of superdiagonals

    Attributes
    ----------
    data : np.ndarray
        Data in compressed banded format (see notes)

    Notes
    -----
    Storage scheme follows LAPACK spec for banded matrices:
        http://www.netlib.org/lapack/lug/node124.html
    Specifically, elements are mapped as
        data[i, j] = matrix[j-U+i, j]

    """

    def __init__(self, shape, dtype, L, U):
        self.shape = shape
        self.dtype = dtype
        self.L = L
        self.U = U
        # Initialize data with Fortran ordering for faster column operations
        self.data = np.zeros((U+L+1, shape[1]), dtype=dtype, order='F')

    @classmethod
    def from_sparse(cls, A):
        """Initialize from a scipy sparse matrix."""
        # Convert to sparse DIA format
        A_dia = sp.dia_matrix(A)
        # Build banded matrix
        U = max(0, max(A_dia.offsets))
        L = max(0, max(-A_dia.offsets))
        A_ban = cls(A_dia.shape, A_dia.dtype, L, U)
        # Copy data
        A_ban.data[U-A_dia.offsets] = A_dia.data
        return A_ban

    def todia(self):
        """Convert to scipy sparse dia matrix."""
        offsets = np.arange(-self.L, self.U+1)[::-1]
        A_dia = sp.dia_matrix((self.data, offsets), shape=self.shape)
        return A_dia

    def copy(self):
        """Create copy."""
        copy = type(self)(self.shape, self.dtype, self.L, self.U)
        np.copyto(copy.data, self.data)
        return copy

    def _matvec(self, v):
        """Matrix-vector multiplication."""
        # Potential replacement: LAPACK ?gbmv
        # Ravel to handle vectors of shape (J,) and (J,1) per scipy spec
        v = v.ravel()
        I, J = self.shape
        U, L = self.U, self.L
        # Allocate output
        dtype = np.promote_types(self.dtype, v.dtype)
        out = np.zeros(I, dtype=dtype)
        # Loop over columns
        for j in range(min(I+U, J)):
            # Full indeces
            i0 = min(I, max(0, j-U))
            i1 = min(I, max(0, j+L+1))
            # Compressed indeces
            k0 = i0 - j + U
            k1 = i1 - j + U
            # Add column
            out[i0:i1] += self.data[k0:k1, j] * v[j]
        return out

    def _adjoint(self):
        """Construct adjoint matrix."""
        # Let scipy handle it
        return type(self).from_sparse(self.todia().H)

