""" Contains various functions for mathematical/geological transforms. """
from math import isnan, ceil
from warnings import warn

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
from numba import njit, prange

from .utility_classes import Accumulator



# Device management
def to_device(array, device='cpu'):
    """ Transfer array to chosen GPU, if possible.
    If `cupy` is not installed, does nothing.

    Parameters
    ----------
    device : str or int
        Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
    """
    if isinstance(device, str) and ':' in device:
        device = int(device.split(':')[1])
    if device in ['cuda', 'gpu']:
        device = 0

    if isinstance(device, int):
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device):
                array = cp.asarray(array)
        else:
            warn('Performance Warning: computing metrics on CPU as `cupy` is not available', RuntimeWarning)
    return array

def from_device(array):
    """ Move the data from GPU, if needed.
    If `cupy` is not installed or supplied array already resides on CPU, does nothing.
    """
    if CUPY_AVAILABLE and hasattr(array, 'device'):
        array = cp.asnumpy(array)
    return array



# Functions to compute various distances between two atleast 2d arrays
def correlation(array1, array2, std1, std2, **kwargs):
    """ Compute correlation. """
    _ = kwargs
    window = array1.shape[-1]
    covariation = (array1 * array2).sum(axis=-1) / window
    return covariation / (std1 * std2)


def crosscorrelation(array1, array2, std1, std2, **kwargs):
    """ Compute crosscorrelation. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    window = array1.shape[-1]
    pad_width = [(0, 0)] * (array2.ndim - 1) + [(window//2, window - window//2)]
    padded = xp.pad(array2, pad_width=tuple(pad_width))

    accumulator = Accumulator('argmax')
    for i in range(window):
        corrs = (array1 * padded[..., i:i+window]).sum(axis=-1)
        accumulator.update(corrs)
    return accumulator.get(final=True).astype(float) - window//2


def btch(array1, array2, std1, std2, **kwargs):
    """ Compute Bhattacharyya distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return xp.sqrt(array1 * array2).sum(axis=-1)


def kl(array1, array2, std1, std2, **kwargs):
    """ Compute Kullback-Leibler divergence. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return 1 - (array2 * xp.log2(array2 / array1)).sum(axis=-1)


def js(array1, array2, std1, std2, **kwargs):
    """ Compute Janson-Shannon divergence. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np

    average = (array1 + array2) / 2
    log_average = xp.log2(average)
    div1 = (array1 * (xp.log2(array1) - log_average)).sum(axis=-1)
    div2 = (array2 * (xp.log2(array2) - log_average)).sum(axis=-1)
    return 1 - (div1 + div2) / 2


def hellinger(array1, array2, std1, std2, **kwargs):
    """ Compute Hellinger distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np

    div = xp.sqrt(((xp.sqrt(array1) - xp.sqrt(array2)) ** 2).sum(axis=-1)) / xp.sqrt(2)
    return 1 - div


def tv(array1, array2, std1, std2, **kwargs):
    """ Compute total variation distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return 1 - xp.abs(array2 - array1).sum(axis=-1) / 2


def hilbert(array, axis=-1):
    """ Compute the analytic signal, using the Hilbert transform. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    N = array.shape[axis]
    fft = xp.fft.fft(array, n=N, axis=axis)

    h = xp.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if array.ndim > 1:
        ind = [xp.newaxis] * array.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    result = xp.fft.ifft(fft * h, axis=axis)
    return result



def make_gaussian_kernel(kernel_size=3, sigma=1.):
    """ Create Gaussian kernel with given parameters: kernel size and std. """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    x_points, y_points = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(x_points) + np.square(y_points)) / np.square(sigma))
    gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))
    return gaussian_kernel


def smooth_out(matrix, kernel=None, kernel_size=3, sigma=2.0, iters=1,
               fill_value=None, preserve=True, margin=np.inf, **kwargs):
    """ Convolve the matrix with a given kernel (or Gaussian with desired parameters).
    A special treatment is given to the missing points (marked with either `fill_value` or `np.nan`),
    and to areas with high variance.

    Parameters
    ----------
    matrix : ndarray
        Array to smooth values in.
    kernel : ndarray or None
        Kernel to convolve with.
    kernel_size : int
        If the kernel is not provided, shape of the square Gaussian kernel.
    sigma : number
        If the kernel is not provided, std of the Gaussian kernel.
    iters : int
        Number of smoothening iterations to perform.
    fill_value : number
        Value to ignore in convolutions.
    preserve : bool
        If False, then all the missing values remain missing in the resulting array.
        If True, then missing values are filled with weighted average of nearby points.
    margin : number
        If the distance between anchor point and the point inside filter is bigger than the margin,
        then the point is ignored in convolutions.
        Can be used for separate smoothening on sides of discontinuity.
    kwargs : other params
        Not used.
    """
    _ = kwargs

    # Convert all the fill values to nans
    matrix = matrix.astype(np.float32).copy()
    if fill_value is not None:
        matrix[matrix == fill_value] = np.nan

    # Pad and make kernel, if needed
    smoothed = np.pad(matrix, kernel_size, constant_values=np.nan)
    kernel = kernel if kernel is not None else make_gaussian_kernel(kernel_size, sigma)

    # Apply smoothing multiple times. Note that there is no dtype conversion in between
    for _ in range(iters):
        smoothed = _smooth_out(smoothed, kernel, preserve=preserve, margin=margin)
    smoothed = smoothed[kernel_size:-kernel_size, kernel_size:-kernel_size]

    # Remove all the unwanted values
    if preserve:
        smoothed[np.isnan(matrix)] = np.nan

    # Convert nans back to fill value
    if fill_value is not None:
        smoothed[np.isnan(smoothed)] = fill_value
    return smoothed

@njit(parallel=True)
def _smooth_out(src, kernel, preserve, margin):
    """ Jit-accelerated function to apply smoothing. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = int(np.floor(kernel.shape[0] / 2))
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(k, i_range - k):
        for xline in range(k, x_range - k):
            central = src[iline, xline]
            if (preserve is True) and isnan(central):
                continue

            element = src[iline-k:iline+k+1, xline-k:xline+k+1]

            s, sum_weights = 0.0, 0.0
            for item, weight in zip(element.ravel(), raveled_kernel):
                if not isnan(item):
                    if abs(item - central) <= margin or isnan(central):
                        s += item * weight
                        sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst


def digitize(matrix, quantiles):
    """ Convert continious metric into binarized version with thresholds defined by `quantiles`. """
    bins = np.nanquantile(matrix, np.sort(quantiles)[::-1])

    if len(bins) > 1:
        digitized = np.digitize(matrix, [*bins, np.nan]).astype(float)
        digitized[digitized > 0] -= 1
    else:
        digitized = np.zeros_like(matrix, dtype=np.float64)
        digitized[matrix <= bins[0]] = 1.0

    digitized[np.isnan(matrix)] = np.nan
    return digitized


def gridify(matrix, frequencies, iline=True, xline=True, full_lines=True):
    """ Convert digitized map into grid with various frequencies corresponding to different bins. """
    values = np.unique(matrix[~np.isnan(matrix)])

    if len(values) != len(frequencies):
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        multiplier = np.power(max_freq/min_freq, 1/(len(values) - 1))
        frequencies = [np.rint(max_freq / (multiplier ** i))
                       for i, _ in enumerate(values)]
    else:
        frequencies = np.sort(frequencies)[::-1]

    grid = np.zeros_like(matrix)
    for value, freq in zip(values, frequencies):
        idx_1, idx_2 = np.nonzero(matrix == value)

        if iline:
            mask = (idx_1 % freq == 0)
            if full_lines:
                grid[idx_1[mask], :] = 1
            else:
                grid[idx_1[mask], idx_2[mask]] = 1

        if xline:
            mask = (idx_2 % freq == 0)
            if full_lines:
                grid[:, idx_2[mask]] = 1
            else:
                grid[idx_1[mask], idx_2[mask]] = 1

    grid[np.isnan(matrix)] = np.nan
    return grid


@njit(parallel=True)
def perturb(data, perturbations, window):
    """ Take a subset of size `window` from each trace, with the center being shifted by `perturbations`. """
    #pylint: disable=not-an-iterable
    i_range, x_range = data.shape[:2]
    output = np.full((i_range, x_range, window), 0.0)
    central = ceil(data.shape[-1] / 2)
    low = window // 2
    high = max(window - low, 0)

    for il in prange(i_range):
        for xl in range(x_range):
            start = central + perturbations[il, xl]
            output[il, xl, :] = data[il, xl, start-low:start+high]
    return output


@njit
def histo_reduce(data, bins):
    """ Convert each entry in data to histograms according to `bins`. """
    #pylint: disable=not-an-iterable
    i_range, x_range = data.shape[:2]

    hist_matrix = np.full((i_range, x_range, len(bins) - 1), np.nan)
    for il in prange(i_range):
        for xl in prange(x_range):
            hist_matrix[il, xl] = np.histogram(data[il, xl], bins=bins)[0]
    return hist_matrix
