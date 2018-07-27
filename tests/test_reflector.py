"""Test banded reflector class."""

import pytest
import numpy as np
import scipy.sparse as sp
import pybanded


def random_ref(I, M, N, complex=True):
    """Build reflector with random entries."""
    if complex:
        data = np.random.randn(M, N) + 1j*np.random.randn(M, N)
        H = pybanded.BandedReflector(I, np.complex128, M, N)
    else:
        data = np.random.randn(M, N)
        H = pybanded.BandedReflector(I, np.float64, M, N)
    np.copyto(H.data, data)
    return H


def random_vector(J, complex=True):
    """Build vector with random entries."""
    if complex:
        return np.random.randn(J) + 1j*np.random.randn(J)
    else:
        return np.random.randn(J)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('M', [1,2,3,4])
@pytest.mark.parametrize('ImN', [0,1,2,3])
@pytest.mark.parametrize('H_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_matvec(I, M, ImN, H_complex, x_complex):
    N = I - ImN
    H = random_ref(I, M, N, H_complex)
    H_dense = H.todense()
    x = random_vector(I, x_complex)
    assert np.allclose(H@x, H_dense@x)


@pytest.mark.parametrize('I', [32,33,45])
@pytest.mark.parametrize('M', [1,2,3,4])
@pytest.mark.parametrize('ImN', [0,1,2,3])
@pytest.mark.parametrize('H_complex', [True,False])
@pytest.mark.parametrize('x_complex', [True,False])
def test_rmatvec(I, M, ImN, H_complex, x_complex):
    N = I - ImN
    H = random_ref(I, M, N, H_complex)
    H_dense = H.todense()
    x = random_vector(I, x_complex)
    assert np.allclose(H.H@x, H_dense.conj().T@x)

