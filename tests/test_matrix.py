"""Test banded matrix class."""

import pytest
import numpy as np
import scipy.sparse as sp
import pybanded


def random_banded_real(I, J, L, U):
    """Build real banded matrix with random entries."""
    data = np.random.randn(L+U+1, J)
    offsets = np.arange(-L, U+1)
    A_dia = sp.dia_matrix((data, offsets), shape=(I, J))
    A_ban = pybanded.BandedMatrix.from_sparse(A_dia)
    return A_ban


def random_banded_complex(I, J, L, U):
    """Build real banded matrix with random entries."""
    data = np.random.randn(L+U+1, J) + 1j*np.random.randn(L+U+1, J)
    offsets = np.arange(-L, U+1)
    A_dia = sp.dia_matrix((data, offsets), shape=(I, J))
    A_ban = pybanded.BandedMatrix.from_sparse(A_dia)
    return A_ban


def random_vector_real(J):
    """Build vector with random entries."""
    return np.random.randn(J)


def random_vector_complex(J):
    """Build vector with random entries."""
    return np.random.randn(J) + 1j*np.random.randn(J)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
def test_matvec_real(I, J, U, L):
    A_ban = random_banded_real(I, J, U, L)
    A_csr = A_ban.todia().tocsr()
    x = random_vector_real(J)
    assert np.allclose(A_ban@x, A_csr@x)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
def test_matvec_complex(I, J, U, L):
    A_ban = random_banded_complex(I, J, U, L)
    A_csr = A_ban.todia().tocsr()
    x = random_vector_complex(J)
    assert np.allclose(A_ban@x, A_csr@x)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
def test_rmatvec_real(I, J, U, L):
    A_ban = random_banded_real(I, J, U, L)
    A_csr = A_ban.todia().tocsr()
    y = random_vector_real(I)
    assert np.allclose(A_ban.H@y, A_csr.H@y)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('J', [32,33,45])
@pytest.mark.parametrize('U', [0,1,2,3])
@pytest.mark.parametrize('L', [0,1,2,3])
def test_rmatvec_complex(I, J, U, L):
    A_ban = random_banded_complex(I, J, U, L)
    A_csr = A_ban.todia().tocsr()
    y = random_vector_complex(I)
    assert np.allclose(A_ban.H@y, A_csr.H@y)

