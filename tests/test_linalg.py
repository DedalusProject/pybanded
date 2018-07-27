"""Test banded reflector class."""

import pytest
import numpy as np
import scipy.sparse as sp
import pybanded


def random_banded(I, J, L, U, complex=True):
    """Build banded matrix with random entries."""
    if complex:
        data = np.random.randn(L+U+1, J) + 1j*np.random.randn(L+U+1, J)
    else:
        data = np.random.randn(L+U+1, J)
    offsets = np.arange(-L, U+1)
    A_dia = sp.dia_matrix((data, offsets), shape=(I, J))
    A_ban = pybanded.BandedMatrix.from_sparse(A_dia)
    return A_ban


def random_vector(J, complex=True):
    """Build vector with random entries."""
    if complex:
        return np.random.randn(J) + 1j*np.random.randn(J)
    else:
        return np.random.randn(J)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('L', [0,1,2,3,4])
@pytest.mark.parametrize('U', [0,1,2,3,4])
@pytest.mark.parametrize('A_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_solve_banded(I, L, U, A_complex, x_complex):
    """Test linear solve by reproducing x."""
    J = I
    A = random_banded(I, J, L, U, A_complex)
    A.data += 10  # offset entries to avoid precision issues
    x = random_vector(J, x_complex)
    b = A @ x
    x_solve = pybanded.solve_banded(A, b)
    assert np.allclose(x_solve, x)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('ImJ', [0,1,2,10])
@pytest.mark.parametrize('L', [0,1,2,3,4])
@pytest.mark.parametrize('U', [0,1,2,3,4])
@pytest.mark.parametrize('A_complex', [True,False])
def test_QR_recomposition(I, ImJ, L, U, A_complex):
    """Test recomposition of QR factorization."""
    J = I - ImJ
    A = random_banded(I, J, L, U, A_complex)
    A.data += 10  # offset entries to avoid precision issues
    QR = pybanded.BandedQR(A)
    A_recomp = QR.Q @ QR.R.todense()
    assert np.allclose(A_recomp, A.todense())


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('L', [0,1,2,3,4])
@pytest.mark.parametrize('U', [0,1,2,3,4])
@pytest.mark.parametrize('A_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_QR_solve(I, L, U, A_complex, x_complex):
    """Test linear solve by QR factorization."""
    J = I
    A = random_banded(I, J, L, U, A_complex)
    A.data += 10  # offset entries to avoid precision issues
    QR = pybanded.BandedQR(A)
    x = random_vector(J, x_complex)
    b = A @ x
    x_solve = QR.solve(b)
    assert np.allclose(x_solve, x)

