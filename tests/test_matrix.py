"""Test banded matrix class."""

import pytest
import numpy as np
import scipy.sparse as sp
import pybanded


def random_banded(I, J, L, U, complex=True):
    """Build banded matrix with random entries."""
    if complex:
        data = np.random.randn(L+U+1, J)
    else:
        data = np.random.randn(L+U+1, J) + 1j*np.random.randn(L+U+1, J)
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
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
@pytest.mark.parametrize('A_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_matvec(I, J, U, L, A_complex, x_complex):
    A_ban = random_banded(I, J, U, L, A_complex)
    A_csr = A_ban.todia().tocsr()
    x = random_vector(J, x_complex)
    assert np.allclose(A_ban@x, A_csr@x)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
@pytest.mark.parametrize('A_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_rmatvec(I, J, U, L, A_complex, x_complex):
    A_ban = random_banded(I, J, U, L, A_complex)
    A_csr = A_ban.todia().tocsr()
    x = random_vector(I, x_complex)
    assert np.allclose(A_ban.H@x, A_csr.H@x)

