import numpy as np

def genwaves(size, A, phi, k):
    """Generate some plane waves on a 3D grid.

    Each of the N plane waves (of displacement) takes the form:

        A k_hat cos(k_vec x_vec + phi)

    which is a vector field sourced by a density plane wave proportional to

        sin(k_vec x_vec + phi)

    The output is the sum of all N vector waves.

    Parameters
    ----------
    size : int
        1D size of the 3D grid
    A : (N,) array_like
        amplitudes of the plane waves, in unit of length (of the output)
    phi : (N,) array_like
        phases of the plane waves
    k : (N, 3) int array_like
        wavevectors of the plane waves, in unit of the fundamental wavenumber

    Returns
    -------
    Psi : (size, size, size, 3) ndarray
        displacement field

    Notes
    -----
    Plane waves are the eigenmodes of a homogeneous field, therefore are the
    ideal test cases to compare ML with perturbation theory.

    """

    A = np.asarray(A)
    phi = np.asarray(phi)
    k = np.asarray(k)
    if k.ndim == 1:
        k.shape = 1, 3

    assert np.issubdtype(k.dtype, np.integer), "breaking periodicity"
    assert np.all(abs(k) <= size//2), "no higher than Nyquist"
    k2 = (k**2).sum(axis=1, keepdims=True)
    assert np.all(k2 > 0), "no DC mode for simplicity"
    k_hat = k / np.sqrt(k2)

    x = np.stack(np.mgrid[:size, :size, :size], axis=-1)
    k_fun = 2 * np.pi / size # fundamental wavenumber in grid unit
    Psi = A * np.cos(x @ (k.T * k_fun) + phi) # scalar wave amplitudes
    Psi.shape = size, size, size, 1, -1
    Psi = np.sum(Psi * k_hat.T, axis=-1) # sum over vector waves

    return Psi
