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
        triangular: LAPACK *tbtrs
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
        dtype = np.promote_types(A.dtype, b.dtype)
        # Cast A.data in case type promotion is necessary
        cast_Adata = A.data.astype(dtype, order='F', copy=False)
        # Cast b in case type promotion or memory reordering is necessary
        # Force copy since kernel changes b->x inplace
        x = cast_b = b.astype(dtype, order='F', copy=True)
        # Call kernel
        info = kernels.solve_banded_lower_triangular(cast_Adata, cast_b, A.shape[0], A.L)
        # Check for error code
        if info:
            raise RuntimeError("LAPACK returned error code: %i" %info)
    elif A.L == 0:
        # Upper triangular
        dtype = np.promote_types(A.dtype, b.dtype)
        # Cast A.data in case type promotion is necessary
        cast_Adata = A.data.astype(dtype, order='F', copy=False)
        # Cast b in case type promotion or memory reordering is necessary
        # Force copy since kernel changes b->x inplace
        x = cast_b = b.astype(dtype, order='F', copy=True)
        # Call kernel
        info = kernels.solve_banded_upper_triangular(cast_Adata, cast_b, A.shape[0], A.U)
        # Check for error code
        if info:
            raise RuntimeError("LAPACK returned error code: %i" %info)
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
        I, J = A.shape
        if I < J:
            raise ValueError("Matrix must have at least as many rows as columns.")
        # Copy A to work matrix with expanded storage
        WU = A.U + A.L
        WL = A.L
        W = BandedMatrix(A.shape, A.dtype, L=WL, U=WU)
        W.data[A.L:, :] = A.data
        # Create compressed Q
        M = WL + 1
        N = min(I-1, J)
        Q = BandedReflector(I, A.dtype, M, N)
        # Apply kernel
        kernels.banded_qr_kernel(W.data, Q.data, I, J, M, N, WU)
        # Build R from W without extra scratch space
        R = BandedMatrix(W.shape, W.dtype, L=0, U=WU)
        R.data[:] = W.data[:WU+1]
        return Q, R

    def solve(self, b):
        """Solve matrix against one right-hand side."""
        # A @ x = b
        # Q @ R @ x = b
        # R @ x = Q.H @ b
        QHb = self.QH @ b
        x = solve_banded(self.R, QHb)
        return x

