""" Utility functions. """
from math import isnan
from collections import OrderedDict
from threading import RLock
from functools import wraps
from hashlib import blake2b

from tqdm import tqdm
import numpy as np
import pandas as pd
import segyio

from numba import njit, prange
from ..batchflow import Sampler



def file_print(msg, path):
    """ Print to file. """
    with open(path, 'w') as file:
        print(msg, file=file)



class SafeIO:
    """ Opens the file handler with desired `open` function, closes it at destruction.
    Can log open and close actions to the `log_file`.
    getattr, getitem and `in` operator are directed to the `handler`.
    """
    def __init__(self, path, opener=open, log_file=None, **kwargs):
        self.path = path
        self.log_file = log_file
        self.handler = opener(path, **kwargs)

        if self.log_file:
            self._info(self.log_file, f'Opened {self.path}')

    def _info(self, log_file, msg):
        with open(log_file, 'a') as f:
            f.write('\n' + msg)

    def __getattr__(self, key):
        return getattr(self.handler, key)

    def __getitem__(self, key):
        return self.handler[key]

    def __contains__(self, key):
        return key in self.handler

    def __del__(self):
        self.handler.close()

        if self.log_file:
            self._info(self.log_file, f'Closed {self.path}')


class IndexedDict(OrderedDict):
    """ Allows to use both indices and keys to subscript. """
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)



def stable_hash(key):
    """ Hash that stays the same between different runs of Python interpreter. """
    if not isinstance(key, (str, bytes)):
        key = ''.join(sorted(str(key)))
    if not isinstance(key, bytes):
        key = key.encode('ascii')
    return str(blake2b(key).hexdigest())

class Singleton:
    """ There must be only one!"""
    instance = None
    def __init__(self):
        if not Singleton.instance:
            Singleton.instance = self

class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to class methods.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    storage : None, OrderedDict or PickleDict
        Storage to use.
        If None, then no caching is applied.
    classwide : bool
        If True, then first argument of a method (self) is changed to class name for the purposes on hashing.
    anchor : bool
        If True, then code of the whole directory this file is located is used to create a persistent hash
        for the purposes of storing.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    pickle_module: str
        Module to use to save/load files on disk. Used only if `storage` is :class:`.PickleDict`.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Notes
    -----
    All arguments to the decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=None, classwide=False, attributes=None):
        self.maxsize = maxsize
        self.classwide = classwide

        # Make `attributes` always a list
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton()
        self.lock = RLock()
        self.reset()


    def reset(self):
        """ Clear cache and stats. """
        self.cache = OrderedDict()
        self.is_full = False
        self.stats = {'hit': 0, 'miss': 0}

    def make_key(self, args, kwargs):
        """ Create a key from a combination of instance reference or class reference,
        method args, and instance attributes.
        """
        key = list(args)
        # key[0] is `instance` if applied to a method
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(key[0], attr))
                key.append(attr_hash)

        if self.classwide:
            key[0] = key[0].__class__
        return tuple(key)


    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = self.make_key(args, kwargs)

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache.get(key, self.default)
                if result is not self.default:
                    del self.cache[key]
                    self.cache[key] = result
                    self.stats['hit'] += 1
                    return result

            # The result was not found in cache: evaluate function
            result = func(*args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats['miss'] += 1
                if key in self.cache:
                    pass
                elif self.is_full:
                    self.cache.popitem(last=False)
                    self.cache[key] = result
                else:
                    self.cache[key] = result
                    self.is_full = (len(self.cache) >= self.maxsize)
            return result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        return wrapper



#TODO: rethink
def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

    Parameters
    ----------
    path : str
        Location of original .sgy cube.
    geometry : SeismicGeometry
        Infered information about original cube.
    path_save : str
        Place to save subcube.
    i_range : array-like
        Ilines to include in subcube.
    x_range : array-like
        Xlines to include in subcube.

    Notes
    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.
    """
    i_low, i_high = i_range[0], i_range[-1]
    x_low, x_high = x_range[0], x_range[-1]

    with segyio.open(path, 'r', strict=False) as src:
        src.mmap()
        spec = segyio.spec()
        spec.sorting = int(src.sorting)
        spec.format = int(src.format)
        spec.samples = range(geometry.depth)
        spec.ilines = geometry.ilines[i_low:i_high]
        spec.xlines = geometry.xlines[x_low:x_high]

        with segyio.create(path_save, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]

            c = 0
            for il_ in tqdm(spec.ilines):
                for xl_ in spec.xlines:
                    tr_ = geometry.il_xl_trace[(il_, xl_)]
                    dst.header[c] = src.header[tr_]
                    dst.header[c][segyio.TraceField.FieldRecord] = il_
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_FILE] = il_

                    dst.header[c][segyio.TraceField.TraceNumber] = xl_ - geometry.xlines_offset
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_LINE] = xl_ - geometry.xlines_offset
                    dst.trace[c] = src.trace[tr_]
                    c += 1
            dst.bin = src.bin
            dst.bin = {segyio.BinField.Traces: c}

    # Check that repaired cube can be opened in 'strict' mode
    with segyio.open(path_save, 'r', strict=True) as _:
        pass

#TODO: rename, add some defaults
def convert_point_cloud(path, path_save, names=None, order=None, transform=None):
    """ Change set of columns in file with point cloud labels.
    Usually is used to remove redundant columns.

    Parameters
    ----------
    path : str
        Path to file to convert.
    path_save : str
        Path for the new file to be saved to.
    names : str or sequence of str
        Names of columns in the original file. Default is Petrel's export format, which is
        ('_', '_', 'iline', '_', '_', 'xline', 'cdp_x', 'cdp_y', 'height'), where `_` symbol stands for
        redundant keywords like `INLINE`.
    order : str or sequence of str
        Names and order of columns to keep. Default is ('iline', 'xline', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    data = df.loc[:, order]
    if transform:
        data = data.apply(transform)
    data.to_csv(path_save, sep=' ', index=False, header=False)


def gen_crop_coordinates(point, horizon_matrix, zero_traces,
                         stride, shape, fill_value, zeros_threshold=0,
                         empty_threshold=5, safe_stripe=0, num_points=2):
    """ Generate crop coordinates next to the point with maximum horizon covered area.

    Parameters
    ----------
    point : array-like
        Coordinates of the point.
    horizon_matrix : ndarray
        `Full_matrix` attribute of the horizon.
    zero_traces : ndarray
        A boolean ndarray indicating zero traces in the cube.
    stride : int
        Distance between the point and a corner of a crop.
    shape : array-like
        The desired shape of the crops.
        Note that final shapes are made in both xline and iline directions. So if
        crop_shape is (1, 64, 64), crops of both (1, 64, 64) and (64, 1, 64) shape
        will be defined.    fill_value : int
    zeros_threshold : int
        A maximum number of bad traces in a crop.
    empty_threshold : int
        A minimum number of points with unknown horizon per crop.
    safe_stripe : int
        Distance between a crop and the ends of the cube.
    num_points : int
        Returned number of crops. The maximum is four.
    """
    candidates, shapes = [], []
    orders, intersections = [], []
    hor_height = horizon_matrix[point[0], point[1]]
    ilines_len, xlines_len = horizon_matrix.shape

    tested_iline_positions = [max(0, point[0] - stride),
                              min(point[0] - shape[1] + stride, ilines_len - shape[1])]

    for il in tested_iline_positions:
        if (il > safe_stripe) and (il + shape[1] < ilines_len - safe_stripe):
            num_missing_traces = np.sum(zero_traces[il: il + shape[1],
                                                    point[1]: point[1] + shape[0]])
            if num_missing_traces <= zeros_threshold:
                horizon_patch = horizon_matrix[il: il + shape[1],
                                               point[1]:point[1] + shape[0]]
                num_empty = np.sum(horizon_patch == fill_value)
                if num_empty > empty_threshold:
                    candidates.append([il, point[1],
                                       hor_height - shape[2] // 2])
                    shapes.append([shape[1], shape[0], shape[2]])
                    orders.append([0, 2, 1])
                    intersections.append(shape[1] - num_empty)

    tested_xline_positions = [max(0, point[1] - stride),
                              min(point[1] - shape[1] + stride, xlines_len - shape[1])]

    for xl in tested_xline_positions:
        if (xl > safe_stripe) and (xl + shape[1] < xlines_len - safe_stripe):
            num_missing_traces = np.sum(zero_traces[point[0]: point[0] + shape[0],
                                                    xl: xl + shape[1]])
            if num_missing_traces <= zeros_threshold:
                horizon_patch = horizon_matrix[point[0]:point[0] + shape[0],
                                               xl: xl + shape[1]]
                num_empty = np.sum(horizon_patch == fill_value)
                if num_empty > empty_threshold:
                    candidates.append([point[0], xl,
                                       hor_height - shape[2] // 2])
                    shapes.append(shape)
                    orders.append([2, 0, 1])
                    intersections.append(shape[1] - num_empty)

    if len(candidates) == 0:
        return None

    candidates_array = np.array(candidates)
    shapes_array = np.array(shapes)
    orders_array = np.array(orders)

    top = np.argsort(np.array(intersections))[:num_points]
    return (candidates_array[top], \
                shapes_array[top], \
                orders_array[top])


@njit
def groupby_mean(array):
    """ Faster version of mean-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1], 1

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1]
            c += 1
        else:
            output[position, :2] = prev
            output[position, -1] = s / c
            position += 1

            prev = curr
            s, c = array[i, -1], 1

    output[position, :2] = prev
    output[position, -1] = s / c
    position += 1
    return output[:position]


@njit
def groupby_min(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = min(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]


@njit
def groupby_max(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = max(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]




@njit
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.
    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        if p <= ticks[0]:
            values[i] = ticks[0]
        elif p >= ticks[-1]:
            values[i] = ticks[-1]
        else:
            ix = np.searchsorted(ticks, p)

            if abs(ticks[ix] - p) <= abs(ticks[ix-1] - p):
                values[i] = ticks[ix]
            else:
                values[i] = ticks[ix-1]
    return values


@njit
def find_min_max(array):
    """ Get both min and max values in just one pass through array."""
    n = array.size
    max_val = min_val = array[0]
    for i in range(1, n):
        min_val = min(array[i], min_val)
        max_val = max(array[i], max_val)
    return min_val, max_val



def compute_running_mean(x, kernel_size):
    """ Fast analogue of scipy.signal.convolve2d with gaussian filter. """
    k = kernel_size // 2
    padded_x = np.pad(x, (k, k), mode='symmetric')
    cumsum = np.cumsum(padded_x, axis=1)
    cumsum = np.cumsum(cumsum, axis=0)
    return _compute_running_mean_jit(x, kernel_size, cumsum)

@njit
def _compute_running_mean_jit(x, kernel_size, cumsum):
    """ Jit accelerated running mean. """
    #pylint: disable=invalid-name
    k = kernel_size // 2
    result = np.zeros_like(x).astype(np.float32)

    canvas = np.zeros((cumsum.shape[0] + 2, cumsum.shape[1] + 2))
    canvas[1:-1, 1:-1] = cumsum
    cumsum = canvas

    for i in range(k, x.shape[0] + k):
        for j in range(k, x.shape[1] + k):
            d = cumsum[i + k + 1, j + k + 1]
            a = cumsum[i - k, j  - k]
            b = cumsum[i - k, j + 1 + k]
            c = cumsum[i + 1 + k, j - k]
            result[i - k, j - k] = float(d - b - c + a) /  float(kernel_size ** 2)
    return result


def mode(array):
    """ Compute mode of the array along the last axis. """
    nan_mask = np.max(array, axis=-1)
    return nb_mode(array, nan_mask)

@njit
def nb_mode(array, mask):
    """ Compute mode of the array along the last axis. """
    #pylint: disable=not-an-iterable
    i_range, x_range = array.shape[:2]
    temp = np.full((i_range, x_range), np.nan)

    for il in prange(i_range):
        for xl in prange(x_range):
            if not isnan(mask[il, xl]):

                current = array[il, xl, :]
                counter = {}
                frequency = 0
                for i in current:
                    if i in counter:
                        counter[i] += 1
                    else:
                        counter[i] = 0

                    if counter[i] > frequency:
                        element = i
                        frequency = counter[i]

                temp[il, xl] = element
    return temp


class HorizonSampler(Sampler):
    """ Compact version of histogram-based sampler for 3D points structure. """
    def __init__(self, histogram, seed=None, **kwargs):
        super().__init__(histogram, seed, **kwargs)
        # Bins and their probabilities: keep only non-zero ones
        bins = histogram[0]
        probs = (bins / np.sum(bins)).reshape(-1)
        self.nonzero_probs_idx = np.asarray(probs != 0.0).nonzero()[0]
        self.nonzero_probs = probs[self.nonzero_probs_idx]

        # Edges of bins: keep lengths and diffs between successive edges
        self.edges = histogram[1]
        self.lens_edges = np.array([len(edge) - 1 for edge in self.edges])
        self.divisors = [np.array(self.lens_edges[i+1:]).astype(np.int64)
                         for i, _ in enumerate(self.edges)]
        self.shifts_edges = [np.diff(edge)[0] for edge in self.edges]

        # Uniform sampler
        self.state = np.random.RandomState(seed=seed)
        self.state_sampler = self.state.uniform

    def sample(self, size):
        """ Generate random sample from histogram distribution. """
        # Choose bin indices
        indices = np.random.choice(self.nonzero_probs_idx, p=self.nonzero_probs, size=size)

        # Convert bin indices to its starting coordinates
        low = generate_points(self.edges, divisors=self.divisors, lengths=self.lens_edges, indices=indices)
        high = low + self.shifts_edges
        return self.state_sampler(low=low, high=high)

@njit
def generate_points(edges, divisors, lengths, indices):
    """ Accelerate sampling method of `HorizonSampler`. """
    low = np.zeros((len(indices), len(lengths)))

    for i, idx in enumerate(indices):
        for j, (edge, divisors_, length) in enumerate(zip(edges, divisors, lengths)):
            idx_copy = idx
            for divisor in divisors_:
                idx_copy //= divisor
            low[i, j] = edge[idx_copy % length]
    return low


def make_distribution_function(array, num_bins=30):
    """ Make vectorized cumulative distribution function for an array. Preserves the input's shape.
    Used for increasing the contrast ration of an image.
    """
    # make 1d-histogram
    densities, coords = np.histogram(array, bins=num_bins, density=True)

    # compute value-series of cumulative distribution function
    dist_func_series = np.cumsum((coords[1:] - coords[:-1]) * densities)
    dist_func_series = np.array([0] + dist_func_series.tolist() + [1])

    # extend coordinates-vec
    coords = np.array(coords.tolist() + [np.inf]).reshape(1, -1)

    # form vectorized cumulative distribution function
    def dist_func(x):
        initial_shape = x.shape
        x = x.reshape(-1, 1)
        bin_nums = np.argmax(x < coords, axis=-1)
        return dist_func_series[bin_nums].reshape(initial_shape)
    
    return dist_func