"""Linear algebra on banded matrices."""

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from . import linalg_kernels as kernels
from .matrix import BandedMatrix
from .reflector import BandedReflector


def solve_banded(A, b):
    """
    Solve A @ x = b where A is a banded matrix.

    Parameters
    ----------
    A : banded matrix
        LHS matrix
    b : np.ndarray
        RHS vector

    Notes
    -----
    The solve strategy is based on the matrix structure:
        diagonal: simple division
        triangular: scipy.linalg.solve_banded
        other: scipy.linalg.solve_banded
    """
    # Check shapes
    if A.shape[0] != A.shape[1]:
        raise ValueError("Can only solve square matrices.")
    if b.ndim != 1:
        raise ValueError("Can only solve against vectors.")
    if b.shape[0] != A.shape[1]:
        raise ValueError("Shape mismatch.")
    # Dispatch based on structure
    if A.U == A.L == 0:
        # Diagonal
        x = b / A.data[0]
    elif A.U == 0:
        # Lower triangular
        x = sla.solve_banded((A.L, A.U), A.data, b)
    elif A.L == 0:
        # Upper triangular
        x = sla.solve_banded((A.L, A.U), A.data, b)
    else:
        # Use scipy banded solve
        x = sla.solve_banded((A.L, A.U), A.data, b)
    return x


class BandedQR:
    """
    QR decomposition of a banded matrix.

    Parameters
    ----------
    A : banded matrix
        Matrix to factorize

    Attributes
    ----------
    Q : banded reflector
        Orthogonal factor as a series of Householder reflections
    R : banded matrix
        Upper triangular factor

    Methods
    -------
    solve(b)
        Solve matrix against one right-hand side using factorization

    """

    def __init__(self, A):
        self.A = A
        # Compute factorization
        self.Q, self.R = self.factorize(A)
        # Store Q.H
        self.QH = self.Q.H

    @staticmethod
    def factorize(A):
        # Check shape
        m, n = A.shape
        if m < n:
            raise ValueError("Matrix must have at least as many rows as columns.")
        # Copy A to scratch R matrix with expanded storage
        R = BandedMatrix(A.shape, A.dtype, A.L, A.U+A.L)
        R.data[A.L:, :] = A.data
        # Local references
        RU = R.U
        RL = R.L
        M = RL + 1
        N = min(m-1, n)
        # Create matrix for compressed Q
        W = np.zeros((M, N), dtype=R.dtype, order='F')
        # Apply kernel
        kernels.qr_banded_kernel(R.data, W, M, N, m, n, RU)
        # Return R without extra scratch space
        Rn = BandedMatrix(R.shape, R.dtype, L=0, U=RU)
        Rn.data[:] = R.data[:RU+1]
        # Build banded Q
        Q = BandedReflector(m, A.dtype, M, N)
        np.copyto(Q.data, W)
        return Q, Rn

    def solve(self, b):
        """Solve matrix against one right-hand side."""
        # A @ x = b
        # Q @ R @ x = b
        # R @ x = Q.H @ b
        QHb = self.QH @ b
        x = solve_banded(self.R, QHb)
        return x

