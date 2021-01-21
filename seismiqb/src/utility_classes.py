""" Helper classes. """
from time import perf_counter
from collections import OrderedDict, defaultdict
from threading import RLock
from functools import wraps
from hashlib import blake2b

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
from numba import njit

from ..batchflow import Sampler



class Accumulator:
    """ Class to accumulate statistics over streamed matrices.
    An example of usage:
        one can either store matrices and take a mean along desired axis at the end of their generation,
        or sequentially update the `mean` matrix with the new data by using this class.
    Note the latter approach is inherintly slower, but requires O(N) times less memory,
    where N is the number of accumulated matrices.

    This class is intended to be used in the following manner:
        - initialize the instance with desired aggregation
        - iteratively call `update` method with new matrices
        - to get the aggregated result, use `get` method

    NaNs are ignored in all computations.
    This class works with both CPU (`numpy`) and GPU (`cupy`) arrays and automatically detects current device.

    Parameters
    ----------
    agg : str
        Which type of aggregation to use. Currently, following modes are implemented:
            - 'mean' works by storing matrix of sums and non-nan counts.
            To get the mean result, the sum is divided by the counts
            - 'std' works by keeping track of sum of the matrices, sum of squared matrices,
            and non-nan counts. To get the result, we subtract squared mean from mean of squared values
            - 'min', 'max' works by iteratively updating the matrix of minima/maxima values
            - 'argmin', 'argmax' iteratively updates index of the minima/maxima values in the passed matrices
            - 'stack' just stores the matrices and concatenates them along (new) last axis
            - 'mode' stores supplied matrices and computes mode along the last axis during the `get` call
    amortize : bool
        If True, then supplied matrices are stacked into ndarray, and then aggregation is applied.
        If False, then accumulation logic is applied.
        Allows for trade-off between memory usage and speed: `amortize=False` is faster,
        but takes more memory resources.
    total : int or None
        If integer, then total number of matrices to be aggregated.
        Used to reduce the memory footprint if `amortize` is set to False.
    axis : int
        Axis to stack matrices on and to apply aggregation funcitons.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, agg='mean', amortize=False, total=None, axis=0):
        self.agg = agg
        self.amortize = amortize
        self.total = total
        self.axis = axis

        self.initialized = False


    def init(self, matrix):
        """ Initialize all the containers on first `update`. """
        # No amortization: collect all the matrices and apply reduce afterwards
        self.module = cp.get_array_module(matrix) if CUPY_AVAILABLE else np
        self.n = 1

        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                self.values = self.module.empty((self.total, *matrix.shape))
                self.values[0, ...] = matrix
            else:
                self.values = [matrix]

            self.initialized = True
            return

        # Amortization: init all the containers
        if self.agg in ['mean', 'nanmean']:
            # Sum of values and counts of non-nan
            self.value = matrix
            self.counts = (~self.module.isnan(matrix)).astype(self.module.int32)

        elif self.agg in ['min', 'nanmin', 'max', 'nanmax']:
            self.value = matrix

        elif self.agg in ['std', 'nanstd']:
            # Same as means, but need to keep track of mean of squares and squared mean
            self.means = matrix
            self.squared_means = matrix ** 2
            self.counts = (~self.module.isnan(matrix)).astype(self.module.int32)

        elif self.agg in ['argmin', 'argmax', 'nanargmin', 'nanargmax']:
            # Keep the current maximum/minimum and update indices matrix, if needed
            self.value = matrix
            self.indices = self.module.zeros_like(matrix)

        self.initialized = True
        return


    def update(self, matrix):
        """ Update containers with new matrix. """
        if not self.initialized:
            self.init(matrix.copy())
            return

        # No amortization: just store everything
        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                self.values[self.n, ...] = matrix
            else:
                self.values.append(matrix)

            self.n += 1
            return

        # Amortization: update underlying containers
        slc = ~self.module.isnan(matrix)

        if self.agg in ['min', 'nanmin']:
            self.value[slc] = self.module.fmin(self.value[slc], matrix[slc])

        elif self.agg in ['max', 'nanmax']:
            self.value[slc] = self.module.fmax(self.value[slc], matrix[slc])

        elif self.agg in ['mean', 'nanmean']:
            mask = np.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = 0.0
            self.value[slc] += matrix[slc]
            self.counts[slc] += 1

        elif self.agg in ['std', 'nanstd']:
            mask = np.logical_and(slc, self.module.isnan(self.means))
            self.means[mask] = 0.0
            self.squared_means[mask] = 0.0
            self.means[slc] += matrix[slc]
            self.squared_means[slc] += matrix[slc] ** 2
            self.counts[slc] += 1

        elif self.agg in ['argmin', 'nanargmin']:
            mask = self.module.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = matrix[mask]
            self.indices[mask] = self.n

            slc_ = matrix < self.value
            self.value[slc_] = matrix[slc_]
            self.indices[slc_] = self.n

        elif self.agg in ['argmax', 'nanargmax']:
            mask = self.module.logical_and(slc, self.module.isnan(self.value))
            self.value[mask] = matrix[mask]
            self.indices[mask] = self.n

            slc_ = matrix > self.value
            self.value[slc_] = matrix[slc_]
            self.indices[slc_] = self.n

        self.n += 1
        return

    def get(self, final=False):
        """ Use stored matrices to get the aggregated result. """
        # No amortization: apply function along the axis to the stacked array
        if self.amortize is False or self.agg in ['stack', 'mode']:
            if self.total:
                stacked = self.values
            else:
                stacked = self.module.stack(self.values, axis=self.axis)

            if final:
                self.values = None

            if self.agg in ['stack']:
                value = stacked

            elif self.agg in ['mode']:
                uniques = self.module.unique(stacked)

                accumulator = Accumulator('argmax')
                for item in uniques[~self.module.isnan(uniques)]:
                    counts = (stacked == item).sum(axis=self.axis)
                    accumulator.update(counts)
                indices = accumulator.get(final=True)
                value = uniques[indices]
                value[self.module.isnan(self.module.max(stacked, axis=self.axis))] = self.module.nan

            else:
                value = getattr(self.module, self.agg)(stacked, axis=self.axis)

            return value

        # Amortization: compute desired aggregation
        if self.agg in ['min', 'nanmin', 'max', 'nanmax']:
            value = self.value

        elif self.agg in ['mean', 'nanmean']:
            slc = self.counts > 0
            value = self.value if final else self.value.copy()
            value[slc] /= self.counts[slc]

        elif self.agg in ['std', 'nanstd']:
            slc = self.counts > 0
            means = self.means if final else self.means.copy()
            means[slc] /= self.counts[slc]

            squared_means = self.squared_means if final else self.squared_means.copy()
            squared_means[slc] /= self.counts[slc]
            value = self.module.sqrt(squared_means - means ** 2)

        elif self.agg in ['argmin', 'argmax', 'nanargmin', 'nanargmax']:
            value = self.indices

        return value



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



class IndexedDict(OrderedDict):
    """ Allows to use both indices and keys to subscript. """
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)



class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to class methods.
    Adds the `use_cache` argument to the decorated method to control whether the caching logic is applied.
    Stored values are individual for each instance of the class.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    apply_by_default : bool
        Whether the cache logic is on by default.

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
    def __init__(self, maxsize=None, attributes=None, apply_by_default=True):
        self.maxsize = maxsize
        self.apply_by_default = apply_by_default

        # Make `attributes` always a list
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton
        self.lock = RLock()
        self.reset()

    def reset(self, instance=None):
        """ Clear cache and stats. """
        if instance is None:
            self.cache = defaultdict(OrderedDict)
            self.is_full = defaultdict(lambda: False)
            self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0})
        else:
            self.cache[instance] = OrderedDict()
            self.is_full[instance] = False
            self.stats[instance] = {'hit': 0, 'miss': 0}

    def make_key(self, instance, args, kwargs):
        """ Create a key from a combination of instance reference, method args, and instance attributes. """
        key = [instance] + list(args)
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(instance, attr))
                key.append(attr_hash)
        return flatten_nested(key)


    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            # Parse the `use_cache`
            if 'use_cache' in kwargs:
                use_cache = kwargs.pop('use_cache')
            else:
                use_cache = self.apply_by_default

            # Skip the caching logic and evaluate function directly
            if not use_cache:
                result = func(instance, *args, **kwargs)
                return result

            key = self.make_key(instance, args, kwargs)

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache[instance].get(key, self.default)
                if result is not self.default:
                    del self.cache[instance][key]
                    self.cache[instance][key] = result
                    self.stats[instance]['hit'] += 1
                    return result

            # The result was not found in cache: evaluate function
            result = func(instance, *args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance]['miss'] += 1
                if key in self.cache[instance]:
                    pass
                elif self.is_full[instance]:
                    self.cache[instance].popitem(last=False)
                    self.cache[instance][key] = result
                else:
                    self.cache[instance][key] = result
                    self.is_full[instance] = (len(self.cache[instance]) >= self.maxsize)
            return result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)
        return wrapper


class SingletonClass:
    """ There must be only one! """
Singleton = SingletonClass()

def stable_hash(key):
    """ Hash that stays the same between different runs of Python interpreter. """
    if not isinstance(key, (str, bytes)):
        key = ''.join(sorted(str(key)))
    if not isinstance(key, bytes):
        key = key.encode('ascii')
    return str(blake2b(key).hexdigest())

def flatten_nested(iterable):
    """ Recursively flatten nested structure of tuples, list and dicts. """
    result = []
    if isinstance(iterable, (tuple, list)):
        for item in iterable:
            result.extend(flatten_nested(item))
    elif isinstance(iterable, dict):
        for key, value in sorted(iterable.items()):
            result.extend((*flatten_nested(key), *flatten_nested(value)))
    else:
        return (iterable,)
    return tuple(result)



class timer:
    """ Context manager for timing the code. """
    def __init__(self, string=''):
        self.string = string
        self.start_time = None

    def __enter__(self):
        self.start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(f'{self.string} evaluated in {(perf_counter() - self.start_time):4.4f} seconds')



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
