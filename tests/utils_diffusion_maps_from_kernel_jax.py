import pytest
import numpy as np
import jax.numpy as jnp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from pytest import approx

from palantir.utils import diffusion_maps_from_kernel_jax


@pytest.fixture
def mock_kernel():
    size = 50
    A = np.random.rand(size, size)
    return csr_matrix((A + A.T) / 2)  # Ensure symmetric matrix for kernel


def test_diffusion_maps_basic_jax(mock_kernel):
    result = diffusion_maps_from_kernel_jax(mock_kernel)

    assert isinstance(result, dict)
    assert "T" in result and "EigenVectors" in result and "EigenValues" in result

    # Use JAX's todense method to verify the shape
    assert jnp.asarray(result["T"].todense()).shape == (50, 50)
    assert result["EigenVectors"].shape == (50, 10)
    assert result["EigenValues"].shape == (10,)


def test_diffusion_maps_n_components_jax(mock_kernel):
    result = diffusion_maps_from_kernel_jax(mock_kernel, n_components=5)

    assert result["EigenVectors"].shape == (50, 5)
    assert result["EigenValues"].shape == (5,)


def test_diffusion_maps_seed_jax(mock_kernel):
    result1 = diffusion_maps_from_kernel_jax(mock_kernel, seed=0)
    result2 = diffusion_maps_from_kernel_jax(mock_kernel, seed=0)

    # Seed usage should yield the same result
    assert np.allclose(result1["EigenValues"], result2["EigenValues"])


def test_diffusion_maps_eigen_jax(mock_kernel):
    result = diffusion_maps_from_kernel_jax(mock_kernel)
    T_dense = np.asarray(result["T"].todense())
    e_values, e_vectors = eigs(T_dense, 10, tol=1e-4, maxiter=1000)
    # Ensure eigenvalues are sorted for consistent comparison
    expected_eigenvalues = np.sort(np.real(e_values))[::-1]  # Assuming largest first
    computed_eigenvalues = np.sort(result["EigenValues"])[::-1]
    
    # Compare using a relative tolerance
    assert np.allclose(computed_eigenvalues, expected_eigenvalues, rtol=1e-3, atol=1e-4)

