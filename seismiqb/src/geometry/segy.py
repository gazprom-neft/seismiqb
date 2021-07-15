""" SEG-Y geometry. """
import os

from itertools import product
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import h5pickle as h5py
import segyio
import cv2

from ..utils import find_min_max
from ..utility_classes import lru_cache, SafeIO

from .base import SeismicGeometry



class SeismicGeometrySEGY(SeismicGeometry):
    """ Class to infer information about SEG-Y cubes and provide convenient methods of working with them.
    A wrapper around `segyio` to provide higher-level API.

    In order to initialize instance, one must supply `path`, `headers` and `index`:
        - `path` is a location of SEG-Y file
        - `headers` is a sequence of trace headers to infer from the file
        - `index_headers` is a subset of `headers` that is used as trace (unique) identifier:
          for example, `INLINE_3D` and `CROSSLINE_3D` has a one-to-one correspondance with trace numbers.
          Another example is `FieldRecord` and `TraceNumber`.
    Default values of `headers` and `index_headers` are ones for post-stack seismic
    (with correctly filled `INLINE_3D` and `CROSSLINE_3D` headers),
    so that post-stack cube can be loaded by providing path only.

    Each instance is basically built around `dataframe` attribute, which describes mapping from
    indexing headers to trace numbers. It is used to, for example, get all trace indices from a desired `FieldRecord`.
    `set_index` method can be called to change indexing headers of the dataframe.

    One can add stats to the instance by calling `collect_stats` method, that makes a full pass through
    the cube in order to analyze distribution of amplitudes. It also collects a number of trace examples
    into `trace_container` attribute, that can be used for later evaluation of various statistics.
    """
    #pylint: disable=attribute-defined-outside-init, too-many-instance-attributes, redefined-builtin
    def __init__(self, path, headers=None, index_headers=None, **kwargs):
        self.structured = False
        self.quantized = False
        self.dataframe = None
        self.segyfile = None

        self.headers = headers or self.HEADERS_POST
        self.index_headers = index_headers or self.INDEX_POST

        super().__init__(path, **kwargs)

    def set_index(self, index_headers, sortby=None):
        """ Change current index to a subset of loaded headers. """
        self.dataframe.reset_index(inplace=True)
        if sortby:
            self.dataframe.sort_values(index_headers, inplace=True, kind='mergesort')# the only stable sorting algorithm
        self.dataframe.set_index(index_headers, inplace=True)
        self.index_headers = index_headers
        self.add_attributes()


    # Methods of inferring dataframe and amplitude stats
    def process(self, collect_stats=False, recollect=False, **kwargs):
        """ Create dataframe based on `segy` file headers. """
        # Note that all the `segyio` structure inference is disabled
        self.segyfile = SafeIO(self.path, opener=segyio.open, mode='r', strict=False, ignore_geometry=True)
        self.segyfile.mmap()

        self.depth = len(self.segyfile.trace[0])
        self.delay = self.segyfile.header[0].get(segyio.TraceField.DelayRecordingTime)
        self.sample_rate = segyio.dt(self.segyfile) / 1000

        # Load all the headers
        dataframe = {}
        for column in self.headers:
            dataframe[column] = self.segyfile.attributes(getattr(segyio.TraceField, column))[slice(None)]

        dataframe = pd.DataFrame(dataframe)
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'index': 'trace_index'}, inplace=True)
        self.dataframe = dataframe.set_index(self.index_headers)

        self.add_attributes()

        # Collect stats, if needed and not collected previously
        if os.path.exists(self.path_meta) and not recollect:
            self.load_meta()
            self.has_stats = True
        elif collect_stats:
            self.collect_stats(**kwargs)

        # Create a matrix with ones at fully-zero traces
        if self.index_headers == self.INDEX_POST and not hasattr(self, 'zero_traces'):
            try:
                size = self.depth // 10
                slc = np.stack([self[:, :, i * size] for i in range(1, 10)], axis=0)
                self.zero_traces = np.zeros(self.lens, dtype=np.int32)
                self.zero_traces[np.std(slc, axis=0) == 0] = 1
            except ValueError: # can't reshape
                pass

        # Store additional segy info
        self.segy_path = self.path
        self.segy_text = [self.segyfile.text[i] for i in range(1 + self.segyfile.ext_headers)]

        # Computed from CDP_X/CDP_Y information
        self.rotation_matrix = self.compute_rotation_matrix()
        self.area = self.compute_area()

    def add_attributes(self):
        """ Infer info about curent index from `dataframe` attribute. """
        self.index_len = len(self.index_headers)
        self._zero_trace = np.zeros(self.depth)

        # Unique values in each of the indexing column
        self.unsorted_uniques = [np.unique(self.dataframe.index.get_level_values(i).values)
                                 for i in range(self.index_len)]
        self.uniques = [np.sort(item) for item in self.unsorted_uniques]
        self.uniques_inversed = [{v: j for j, v in enumerate(self.uniques[i])}
                                 for i in range(self.index_len)]

        self.byte_no = [getattr(segyio.TraceField, h) for h in self.index_headers]
        self.offsets = [np.min(item) for item in self.uniques]
        self.lens = [len(item) for item in self.uniques]
        self.ranges = [(np.min(item), np.max(item)) for item in self.uniques]

        self.cube_shape = np.asarray([*self.lens, self.depth])

    def _get_store_key(self, traceseqno):
        """ get trace lateral coordinates from header """
        header = self.segyfile.header[traceseqno]
        # i -> id in a dataframe
        keys = [header.get(field) for field in self.byte_no]
        store_key = tuple(self.uniques_inversed[j][item] for j, item in enumerate(keys))
        return store_key

    def collect_stats(self, spatial=True, bins=25, num_keep=10000, pbar=True, **kwargs):
        """ Pass through file data to collect stats:
            - min/max values.
            - a number of quantiles of values in the cube.
            - certain amount of traces are stored in a `trace_container` attribute.

        If `spatial` is True, makes an additional pass through the cube to obtain following:
            - min/max/mean/std for every trace - `min_matrix`, `max_matrix` and so on.
            - histogram of values for each trace: - `hist_matrix`.
            - bins for histogram creation: - `bins`.

        Parameters
        ----------
        spatial : bool
            Whether to collect additional stats.
        bins : int or str
            Number of bins or name of automatic algorithm of defining number of bins.
        num_keep : int
            Number of traces to store.
        """
        #pylint: disable=not-an-iterable
        _ = kwargs

        num_traces = len(self.segyfile.header)
        frequency = num_traces // num_keep

        # Get min/max values, store some of the traces
        trace_container = []
        value_min, value_max = np.inf, -np.inf
        min_matrix, max_matrix = np.full(self.lens, np.nan), np.full(self.lens, np.nan)

        for i in tqdm(range(num_traces), desc='Finding min/max', ncols=800, disable=(not pbar)):
            trace = self.segyfile.trace[i]
            store_key = self._get_store_key(i)

            trace_min, trace_max = find_min_max(trace)
            min_matrix[store_key] = trace_min
            max_matrix[store_key] = trace_max

            if i % frequency == 0 and trace_min != trace_max:
                trace_container.extend(trace.tolist())
                #TODO: add dtype for storing

        # Store everything into instance
        self.min_matrix, self.max_matrix = min_matrix, max_matrix
        self.zero_traces = (min_matrix == max_matrix).astype(np.int)
        self.zero_traces[np.isnan(min_matrix)] = 1

        value_min = np.nanmin(min_matrix)
        value_max = np.nanmax(max_matrix)

        # Collect more spatial stats: min, max, mean, std, histograms matrices
        if spatial:
            # Make bins
            bins = np.histogram_bin_edges(None, bins, range=(value_min, value_max)).astype(np.float)
            self.bins = bins

            # Create containers
            hist_matrix = np.full((*self.lens, len(bins)-1), np.nan)

            # Iterate over traces
            description = f'Collecting stats for {self.displayed_name}'
            for i in tqdm(range(num_traces), desc=description, ncols=800, disable=(not pbar)):
                trace = self.segyfile.trace[i]
                store_key = self._get_store_key(i)

                # For each trace, we store an entire histogram of amplitudes
                val_min, val_max = find_min_max(trace)
                if val_min != val_max:
                    histogram = np.histogram(trace, bins=bins)[0]
                    hist_matrix[store_key] = histogram

            # Restore stats from histogram
            midpoints = (bins[1:] + bins[:-1]) / 2
            probs = hist_matrix / np.sum(hist_matrix, axis=-1, keepdims=True)

            mean_matrix = np.sum(probs * midpoints, axis=-1)
            std_matrix = np.sqrt(np.sum((np.broadcast_to(midpoints, (*mean_matrix.shape, len(midpoints))) - \
                                         mean_matrix.reshape(*mean_matrix.shape, 1))**2 * probs,
                                        axis=-1))

            # Store everything into instance
            self.mean_matrix, self.std_matrix = mean_matrix, std_matrix
            self.hist_matrix = hist_matrix

        self.trace_container = np.array(trace_container)
        self.v_uniques = len(np.unique(trace_container))
        self.v_min, self.v_max = value_min, value_max
        self.v_mean, self.v_std = np.mean(trace_container), np.std(trace_container)
        self.v_q001, self.v_q01, self.v_q05 = np.quantile(trace_container, [0.001, 0.01, 0.05])
        self.v_q999, self.v_q99, self.v_q95 = np.quantile(trace_container, [0.999, 0.99, 0.95])
        self.has_stats = True
        self.store_meta()

    # Compute stats from CDP/LINES correspondence
    def compute_rotation_matrix(self):
        """ Compute transform from INLINE/CROSSLINE coordinates to CDP system. """
        ix_points = []
        cdp_points = []

        for _ in range(3):
            idx = np.random.randint(len(self.dataframe))
            trace = self.segyfile.header[idx]

            # INLINE_3D -> CDP_X, CROSSLINE_3D -> CDP_Y
            ix = (trace[segyio.TraceField.INLINE_3D], trace[segyio.TraceField.CROSSLINE_3D])
            cdp = (trace[segyio.TraceField.CDP_X], trace[segyio.TraceField.CDP_Y])

            ix_points.append(ix)
            cdp_points.append(cdp)
        rotation_matrix = cv2.getAffineTransform(np.float32(ix_points), np.float32(cdp_points))
        return rotation_matrix

    def compute_area(self, correct=True, shift=50):
        """ Compute approximate area of the cube in square kilometres.

        Parameters
        ----------
        correct : bool
            Whether to correct computed area for zero traces.
        """
        i = self.ilines[self.ilines_len // 2]
        x = self.xlines[self.xlines_len // 2]

        # Central trace coordinates
        idx = self.dataframe['trace_index'][(i, x)]
        trace = self.segyfile.header[idx]
        cdp_x, cdp_y = (trace[segyio.TraceField.CDP_X], trace[segyio.TraceField.CDP_Y])

        # Two shifted traces
        idx_dx = self.dataframe['trace_index'][(i, x + shift)]
        trace_dx = self.segyfile.header[idx_dx]
        cdp_x_delta = abs(trace_dx[segyio.TraceField.CDP_X] - cdp_x)

        idx_dy = self.dataframe['trace_index'][(i + shift, x)]
        trace_dy = self.segyfile.header[idx_dy]
        cdp_y_delta = abs(trace_dy[segyio.TraceField.CDP_Y] - cdp_y)

        # Traces if CDP_X/CDP_Y coordinate system is rotated on 90 degrees with respect to ILINES/CROSSLINES
        if cdp_x_delta == 0 and cdp_y_delta == 0:
            idx_dx = self.dataframe['trace_index'][(i + shift, x)]
            trace_dx = self.segyfile.header[idx_dx]
            cdp_x_delta = abs(trace_dx[segyio.TraceField.CDP_X] - cdp_x)

            idx_dy = self.dataframe['trace_index'][(i, x + shift)]
            trace_dy = self.segyfile.header[idx_dy]
            cdp_y_delta = abs(trace_dy[segyio.TraceField.CDP_Y] - cdp_y)

        cdp_x_delta /= shift
        cdp_y_delta /= shift

        ilines_km = cdp_y_delta * self.ilines_len / 1000
        xlines_km = cdp_x_delta * self.xlines_len / 1000
        area = ilines_km * xlines_km

        if correct and hasattr(self, 'zero_traces'):
            area -= (cdp_x_delta / 1000) * (cdp_y_delta / 1000) * np.sum(self.zero_traces)
        return round(area, 2)


    # Methods to load actual data from SEG-Y
    # 1D
    def load_trace(self, index):
        """ Load individual trace from segyfile.
        If passed `np.nan`, returns trace of zeros.
        """
        # TODO: can be improved by creating buffer and writing directly to it
        if not np.isnan(index):
            return self.segyfile.trace.raw[int(index)]
        return self._zero_trace

    def load_traces(self, trace_indices):
        """ Stack multiple traces together. """
        # TODO: can be improved by preallocating memory and passing it as a buffer to `load_trace`
        return np.stack([self.load_trace(idx) for idx in trace_indices])

    # 2D
    @lru_cache(128, attributes='index_headers')
    def load_slide(self, loc=None, axis=0, start=None, end=None, step=1, stable=True):
        """ Create indices and load actual traces for one slide.

        If the current index is 1D, then slide is defined by `start`, `end`, `step`.
        If the current index is 2D, then slide is defined by `loc` and `axis`.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        start, end, step : ints
            Parameters of slice loading for 1D index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        if axis in [0, 1]:
            indices = self.make_slide_indices(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)
            slide = self.load_traces(indices)
        elif axis == 2:
            slide = self.segyfile.depth_slice[loc]

            if slide.shape[0] == np.prod(self.lens):
                slide = slide.reshape(self.lens)
            else:
                buffer = np.zeros_like(self.zero_traces, dtype=np.float32)
                buffer[self.zero_traces == 0] = slide
                slide = buffer
        return slide

    def make_slide_indices(self, loc=None, axis=0, start=None, end=None, step=1, stable=True, return_iterator=False):
        """ Choose appropriate version of index creation, depending on length of the current index.

        Parameters
        ----------
        start, end, step : ints
            Parameters of slice loading for 1d index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        return_iterator : bool
            Whether to also return the same iterator that is used to index current `dataframe`.
            Can be useful for subsequent loads from the same place in various instances.
        """
        if self.index_len == 1:
            _ = loc, axis
            result = self.make_slide_indices_1d(start=start, end=end, step=step, stable=stable,
                                                return_iterator=return_iterator)
        elif self.index_len == 2:
            _ = start, end, step
            result = self.make_slide_indices_2d(loc=loc, axis=axis, stable=stable,
                                                return_iterator=return_iterator)
        elif self.index_len == 3:
            raise NotImplementedError('Yet to be done!')
        else:
            raise ValueError('Index lenght must be less than 4. ')
        return result

    def make_slide_indices_1d(self, start=None, end=None, step=1, stable=True, return_iterator=False):
        """ 1D version of index creation. """
        start = start or self.offsets[0]
        end = end or self.uniques[0][-1]

        if stable:
            iterator = self.dataframe.index[(self.dataframe.index >= start) & (self.dataframe.index <= end)]
            iterator = iterator.values[::step]
        else:
            iterator = np.arange(start, end+1, step)

        indices = self.dataframe['trace_index'].reindex(iterator, fill_value=np.nan).values

        if return_iterator:
            return indices, iterator
        return indices

    def make_slide_indices_2d(self, loc, axis=0, stable=True, return_iterator=False):
        """ 2D version of index creation. """
        other_axis = 1 - axis
        location = self.uniques[axis][loc]

        if stable:
            others = self.dataframe[self.dataframe.index.get_level_values(axis) == location]
            others = others.index.get_level_values(other_axis).values
        else:
            others = self.uniques[other_axis]

        iterator = list(zip([location] * len(others), others) if axis == 0 else zip(others, [location] * len(others)))
        indices = self.dataframe['trace_index'].reindex(iterator, fill_value=np.nan).values

        #TODO: keep only uniques, when needed, with `nan` filtering
        if stable:
            indices = np.unique(indices)

        if return_iterator:
            return indices, iterator
        return indices

    # 3D
    def _load_crop(self, locations):
        """ Load 3D crop from the cube.

        Parameters
        ----------
        locations : sequence of slices
            List of desired slices to load: along the first index, the second, and depth.

        Example
        -------
        If the current index is `INLINE_3D` and `CROSSLINE_3D`, then to load
        5:110 ilines, 100:1105 crosslines, 0:700 depths, locations must be::
            [slice(5, 110), slice(100, 1105), slice(0, 700)]
        """
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.cube_shape)])
        indices = self.make_crop_indices(locations)
        crop = self.load_traces(indices)[..., locations[-1]].reshape(shape)
        return crop

    def make_crop_indices(self, locations):
        """ Create indices for 3D crop loading. """
        iterator = list(product(*[[self.uniques[idx][i] for i in range(locations[idx].start, locations[idx].stop)]
                                  for idx in range(2)]))
        indices = self.dataframe['trace_index'].reindex(iterator, fill_value=np.nan).values
        _, unique_ind = np.unique(indices, return_index=True)
        return indices[np.sort(unique_ind, kind='stable')]

    def load_crop(self, locations, threshold=15, mode='adaptive', **kwargs):
        """ Smart choice between using :meth:`._load_crop` and stacking multiple slides created by :meth:`.load_slide`.

        Parameters
        ----------
        mode : str
            If `adaptive`, then function to load is chosen automatically.
            If `slide` or `crop`, then uses that function to load data.
        threshold : int
            Upper bound for amount of slides to load. Used only in `adaptive` mode.
        """
        _ = kwargs
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.cube_shape)])
        axis = np.argmin(shape)
        if mode == 'adaptive':
            if axis in [0, 1]:
                mode = 'slide' if min(shape) < threshold else 'crop'
            else:
                flag = np.prod(shape[:2]) / np.prod(self.cube_shape[:2])
                mode = 'slide' if flag > 0.1 else 'crop'

        if mode == 'slide':
            slc = locations[axis]
            if axis == 0:
                return np.stack([self.load_slide(loc, axis=axis)[locations[1], locations[2]]
                                 for loc in range(slc.start, slc.stop)], axis=axis)
            if axis == 1:
                return np.stack([self.load_slide(loc, axis=axis)[locations[0], locations[2]]
                                 for loc in range(slc.start, slc.stop)], axis=axis)
            if axis == 2:
                return np.stack([self.load_slide(loc, axis=axis)[locations[0], locations[1]]
                                 for loc in range(slc.start, slc.stop)], axis=axis)
        return self._load_crop(locations)


    # Quantization
    def compute_quantization_parameters(self, ranges='q99', clip=True, center=False):
        """ Make bins for int8 quantization and convert value-stats.

        Parameters
        ----------
        ranges : str or sequence of two numbers
            Ranges to quantize data to. Available options are:
                - `q95`, `q99`, `q999` to clip data to respective quantiles.
                - `same` keep the same range of data.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        """
        ranges_dict = {
            'q95': min(abs(self.v_q05), abs(self.v_q95)),
            'q99': min(abs(self.v_q01), abs(self.v_q99)),
            'q999': min(abs(self.v_q001), abs(self.v_q999)),
            'same': max(abs(self.v_min), abs(self.v_max)),
        }
        if ranges in ranges_dict:
            ranges = ranges_dict[ranges]
            ranges = (-ranges, +ranges)

        if center:
            ranges = tuple(item - self.v_mean for item in ranges)

        self.qnt_ranges = ranges
        self.qnt_bins = np.histogram_bin_edges(None, bins=254, range=ranges).astype(np.float)
        self.qnt_clip = clip
        self.qnt_center = center

        # Compute quantized statistics
        quantized_tc = self.quantize(self.trace_container)
        self.qnt_min, self.qnt_max = self.quantize(self.v_min), self.quantize(self.v_max)
        self.qnt_mean, self.qnt_std = np.mean(quantized_tc), np.std(quantized_tc)
        self.qnt_q001, self.qnt_q01, self.qnt_q05 = np.quantile(quantized_tc, [0.001, 0.01, 0.05])
        self.qnt_q999, self.qnt_q99, self.qnt_q95 = np.quantile(quantized_tc, [0.999, 0.99, 0.95])

        # Estimate difference after quantization
        quantized_tc += 127
        restored_tc = self.qnt_bins[quantized_tc]
        self.qnt_error = np.mean(np.abs(restored_tc - self.trace_container)) / self.v_std

    def quantize(self, array):
        """ Convert array of floats to int8 values. """
        if self.qnt_center:
            array -= self.v_mean
        if self.qnt_clip:
            array = np.clip(array, *self.qnt_ranges)
        array = np.digitize(array, self.qnt_bins) - 128
        return array.astype(np.int8)


    # Convert SEG-Y
    def convert(self, format='blosc', path=None, postfix='', projections='ixh',
                quantize=True, ranges='q99', clip=True, center=False, store_meta=True, pbar=True, **kwargs):
        """ Convert SEG-Y file to a more effective storage.

        Parameters
        ----------
        format : {'hdf5', 'qhdf5', 'blosc', 'qblosc}
            Format of storage to convert to: `blosc` takes less space, but a touch slower, than `hdf5`.
            Prefix `q` sets the `quantize` parameter to True.
        path : str
            If provided, then path to save file to.
            Otherwise, file is saved under the same name with different extension.
        postfix : str
            Optional string to add before extension. Used only if the `path` is not provided.
        projections : str
            Which projections of data to store: `i` for iline one, `x` for the crossline, `h` for depth.
        quantize : bool
            Whether to binarize data to `int8` dtype. `ranges`, `clip` and `center` define parameters of quantization.
            Binarization is done uniformly over selected `ranges` of values.
            If True, then `q` is appended to extension.
        ranges : str
            Ranges to quantize data to. Available options are:
                - `q95`, `q99`, `q999` to clip data to respective quantiles.
                - `same` keep the same range of data.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        store_meta : bool
            Whether to store meta near the save file.
        pbar : bool
            Whether to show progress bar during conversion.
        kwargs : dict
            Other parameters, passed directly to the file constructor of chosen format.
            If format is `blosc`:
                - `cname` for algorithm of compression. Default is `lz4hc`.
                - `clevel` for level of compression. Default is 6.
                - `shuffle` for bitshuffle during compression. Default is False.
        """
        #pylint: disable=import-outside-toplevel
        # Select format
        if format.startswith('q'):
            quantize = True
            format = format[1:]

        from .converted import SeismicGeometryConverted
        if format == 'blosc':
            from .blosc import BloscFile
            constructor, mode = BloscFile, 'w'
        elif format == 'hdf5':
            constructor, mode = h5py.File, 'w-'

        # Quantization
        if quantize:
            self.compute_quantization_parameters(ranges=ranges, clip=clip, center=center)
            dtype, transform = np.int8, self.quantize
        else:
            dtype, transform = np.float32, lambda array: array

        if path is None:
            fmt_prefix = 'q' if quantize else ''

            if postfix == '' and len(projections) < 3:
                postfix = '_' + projections

            path = os.path.join(os.path.dirname(self.path), f'{self.short_name}{postfix}.{fmt_prefix}{format}')

        # Remove file, if exists
        if os.path.exists(path):
            os.remove(path)

        # Create file and datasets inside
        with constructor(path, mode=mode, **kwargs) as file:
            total = (('i' in projections) * self.cube_shape[0] +
                     ('x' in projections) * self.cube_shape[1] +
                     ('h' in projections) * self.cube_shape[2])
            progress_bar = tqdm(total=total, ncols=800, disable=(not pbar))
            name = os.path.basename(path)

            for p in projections:
                axis = self.parse_axis(p)
                cube_name = SeismicGeometryConverted.AXIS_TO_NAME[axis]
                order = SeismicGeometryConverted.AXIS_TO_ORDER[axis]
                cube = file.create_dataset(cube_name, shape=self.cube_shape[order], dtype=dtype)

                progress_bar.set_description(f'Creating {name}; {p}-projection')
                for idx in range(self.cube_shape[axis]):
                    slide = self.load_slide(idx, axis=axis, stable=False)
                    slide = slide.T if axis == 1 else slide
                    slide = transform(slide)
                    cube[idx, :, :] = slide
                    progress_bar.update()
            progress_bar.close()

        if store_meta:
            if not self.has_stats:
                self.collect_stats(pbar=pbar)

            path_meta = os.path.splitext(path)[0] + '.meta'
            self.store_meta(path_meta)

        return SeismicGeometry(path)

    def convert_to_hdf5(self, path=None, postfix='', projections='ixh',
                        quantize=True, ranges='q99', clip=True, center=False, store_meta=True, pbar=True, **kwargs):
        """ Convenient alias for HDF5 conversion. """
        kwargs_ = locals()
        kwargs_.pop('self')
        kwargs_.pop('kwargs')
        return self.convert(format='hdf5', **kwargs_, **kwargs)

    def convert_to_blosc(self, path=None, postfix='', projections='ixh',
                         quantize=True, ranges='q99', clip=True, center=False, store_meta=True, pbar=True, **kwargs):
        """ Convenient alias for BLOSC conversion. """
        kwargs_ = locals()
        kwargs_.pop('self')
        kwargs_.pop('kwargs')
        return self.convert(format='blosc', **kwargs_, **kwargs)
