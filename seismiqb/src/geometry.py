""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import sys
import shutil
import itertools

from textwrap import dedent
from random import random
from itertools import product
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import h5py
import segyio
import cv2

from .hdf5_storage import StorageHDF5
from .utils import lru_cache, find_min_max, file_print, parse_axis,\
                   SafeIO, compute_attribute, make_axis_grid, fill_defaults
from .plotters import plot_image



class SpatialDescriptor:
    """ Allows to set names for parts of information about index.
    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    allows to get instance.lens[idx], where `idx` is position of `INLINE_3D` inside instance.index.

    Roughly equivalent to::
    @property
    def ilines_len(self):
        idx = self.index_headers.index('INLINE_3D')
        return self.lens[idx]
    """
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, header=None, attribute=None, name=None):
        self.header = header
        self.attribute = attribute

        if name is not None:
            self.name = name

    def __get__(self, obj, obj_class=None):
        # If attribute is already stored in object, just return it
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]

        # Find index of header, use it to access attr
        try:
            idx = obj.index_headers.index(self.header)
            return getattr(obj, self.attribute)[idx]
        except ValueError as exc:
            raise ValueError(f'Current index does not contain {self.header}.') from exc


def add_descriptors(cls):
    """ Add multiple descriptors to the decorated class.
    Name of each descriptor is `alias + postfix`.

    Roughly equivalent to::
    ilines = SpatialDescriptor('INLINE_3D', 'uniques', 'ilines')
    xlines = SpatialDescriptor('CROSSLINE_3D', 'uniques', 'xlines')

    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    xlines_len = SpatialDescriptor('CROSSLINE_3D', 'lens', 'xlines_len')
    etc
    """
    attrs = ['uniques', 'offsets', 'lens']  # which attrs hold information
    postfixes = ['', '_offset', '_len']     # postfix of current attr

    headers = ['INLINE_3D', 'CROSSLINE_3D'] # headers to use
    aliases = ['ilines', 'xlines']          # alias for header

    for attr, postfix in zip(attrs, postfixes):
        for alias, header in zip(aliases, headers):
            name = alias + postfix
            descriptor = SpatialDescriptor(header=header, attribute=attr, name=name)
            setattr(cls, name, descriptor)
    return cls



@add_descriptors
class SeismicGeometry:
    """ This class selects which type of geometry to initialize: the SEG-Y or the HDF5 one,
    depending on the passed path.

    Independent of exact format, `SeismicGeometry` provides following:
        - Attributes to describe shape and structure of the cube like `cube_shape` and `lens`,
        as well as exact values of file-wide headers, for example, `time_delay` and `sample_rate`.

        - Ability to infer information about the cube amplitudes:
          `trace_container` attribute contains examples of amplitudes inside the cube and allows to compute statistics.

        - If needed, spatial stats can also be inferred: attributes `min_matrix`, `mean_matrix`, etc
          allow to create a complete spatial map (that is view from above) of the desired statistic for the whole cube.
          `hist_matrix` contains a histogram of values for each trace in the cube, and can be used as
          a proxy for amplitudes in each trace for evaluating aggregated statistics.

        - `load_slide` (2D entity) or `load_crop` (3D entity) methods to load data from the cube.
          Load slides takes a number of slide and axis to cut along; makes use of `lru_cache` to work
          faster for subsequent loads. Cache is bound for each instance.
          Load crops works off of complete location specification (3D slice).

        - `quality_map` attribute is a spatial matrix that estimates cube hardness;
          `quality_grid` attribute contains a grid of locations to train model on, based on `quality_map`.

        - `show_slide` method allows to do exactly what the name says, and has the same API as `load_slide`.
          `repr` allows to get a quick summary of the cube statistics.

    Refer to the documentation of respective classes to learn more about their structure, attributes and methods.
    """
    #TODO: add separate class for cube-like labels
    SEGY_ALIASES = ['sgy', 'segy', 'seg']
    HDF5_ALIASES = ['hdf5', 'h5py']
    NPZ_ALIASES = ['npz']

    # Attributes to store during SEG-Y -> HDF5 conversion
    PRESERVED = [
        'depth', 'delay', 'sample_rate', 'cube_shape',
        'segy_path', 'segy_text', 'rotation_matrix',
        'byte_no', 'offsets', 'ranges', 'lens', # `uniques` can't be saved due to different lenghts of arrays
        'ilines', 'xlines', 'ilines_offset', 'xlines_offset', 'ilines_len', 'xlines_len',
        'value_min', 'value_max', 'q01', 'q99', 'q001', 'q999', 'bins', 'zero_traces', '_quality_map',
    ]

    PRESERVED_LAZY = [
        'trace_container', 'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix', 'hist_matrix',
    ]

    # Headers to load from SEG-Y cube
    HEADERS_PRE_FULL = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE', 'CDP', 'CDP_TRACE', 'offset', ]
    HEADERS_POST_FULL = ['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']
    HEADERS_POST = ['INLINE_3D', 'CROSSLINE_3D']

    # Headers to use as id of a trace
    INDEX_PRE = ['FieldRecord', 'TraceNumber']
    INDEX_POST = ['INLINE_3D', 'CROSSLINE_3D']
    INDEX_CDP = ['CDP_Y', 'CDP_X']

    def __new__(cls, path, *args, **kwargs):
        """ Select the type of geometry based on file extension. """
        _ = args, kwargs
        fmt = os.path.splitext(path)[1][1:]

        if fmt in cls.SEGY_ALIASES:
            new_cls = SeismicGeometrySEGY
        elif fmt in cls.HDF5_ALIASES:
            new_cls = SeismicGeometryHDF5
        elif fmt in cls.NPZ_ALIASES:
            new_cls = SeismicGeometryNPZ
        else:
            raise TypeError('Unknown format of the cube.')

        instance = super().__new__(new_cls)
        return instance

    def __init__(self, path, *args, process=True, **kwargs):
        _ = args
        self.path = path

        # Names of different lengths and format: helpful for outside usage
        self.name = os.path.basename(self.path)
        self.short_name = self.name.split('.')[0]
        self.long_name = ':'.join(self.path.split('/')[-2:])
        self.format = os.path.splitext(self.path)[1][1:]

        self._quality_map = None
        self._quality_grid = None

        self.path_meta = None
        self.loaded = []
        self.has_stats = False
        if process:
            self.process(**kwargs)

    def __len__(self):
        """ Number of meaningful traces. """
        if hasattr(self, 'zero_traces'):
            return np.prod(self.zero_traces.shape) - self.zero_traces.sum()
        return len(self.dataframe)


    def store_meta(self, path=None):
        """ Store collected stats on disk. """
        path_meta = path or os.path.splitext(self.path)[0] + '.meta'

        # Remove file, if exists: h5py can't do that
        if os.path.exists(path_meta):
            os.remove(path_meta)

        # Create file and datasets inside
        with h5py.File(path_meta, "a") as file_meta:
            # Save all the necessary attributes to the `info` group
            for attr in self.PRESERVED + self.PRESERVED_LAZY:
                try:
                    if hasattr(self, attr) and getattr(self, attr) is not None:
                        file_meta['/info/' + attr] = getattr(self, attr)
                except ValueError:
                    # Raised when you try to store post-stack descriptors for pre-stack cube
                    pass

    def load_meta(self):
        """ Retrieve stored stats from disk. """
        path_meta = os.path.splitext(self.path)[0] + '.meta'

        # Backward compatibility
        if not os.path.exists(path_meta):
            path_meta = os.path.splitext(self.path)[0] + '.hdf5'
        self.path_meta = path_meta

        for item in self.PRESERVED:
            value = self.load_meta_item(item)
            if value is not None:
                setattr(self, item, value)

    def load_meta_item(self, item):
        """ Load individual item. """
        with h5py.File(self.path_meta, "r") as file_meta:
            try:
                value = file_meta['/info/' + item][()]
                self.loaded.append(item)
                return value
            except KeyError:
                return None

    def __getattr__(self, key):
        """ Load item from stored meta, if needed. """
        if key in self.PRESERVED_LAZY and self.path_meta is not None and key not in self.__dict__:
            return self.load_meta_item(key)
        return object.__getattribute__(self, key)


    def scaler(self, array, mode='minmax'):
        """ Normalize array of amplitudes cut from the cube.

        Parameters
        ----------
        array : ndarray
            Crop of amplitudes.
        mode : str
            If `minmax`, then data is scaled to [0, 1] via minmax scaling.
            If `q` or `normalize`, then data is divided by the maximum of absolute values of the
            0.01 and 0.99 quantiles.
            If `q_clip`, then data is clipped to 0.01 and 0.99 quantiles and then divided by the
            maximum of absolute values of the two.
        """
        if mode in ['q', 'normalize']:
            return array / max(abs(self.q01), abs(self.q99))
        if mode in ['q_clip']:
            return np.clip(array, self.q01, self.q99) / max(abs(self.q01), abs(self.q99))
        if mode == 'minmax':
            scale = (self.value_max - self.value_min)
            return (array - self.value_min) / scale
        raise ValueError('Wrong mode', mode)

    def make_slide_locations(self, loc, axis=0):
        """ Create locations (sequence of locations for each axis) for desired slide along desired axis. """
        axis = parse_axis(axis, self.index_headers)

        locations = [slice(0, item) for item in self.cube_shape]
        locations[axis] = slice(loc, loc + 1)
        return locations


    # Spatial matrices
    @lru_cache(100)
    def get_quantile_matrix(self, q):
        """ Restore the quantile matrix for desired `q` from `hist_matrix`.

        Parameters
        ----------
        q : number
            Quantile to compute. Must be in (0, 1) range.
        """
        #pylint: disable=line-too-long
        threshold = self.depth * q
        cumsums = np.cumsum(self.hist_matrix, axis=-1)

        positions = np.argmax(cumsums >= threshold, axis=-1)
        idx_1, idx_2 = np.nonzero(positions)
        indices = positions[idx_1, idx_2]

        broadcasted_bins = np.broadcast_to(self.bins, (*positions.shape, len(self.bins)))

        q_matrix = np.zeros_like(positions, dtype=np.float)
        q_matrix[idx_1, idx_2] += broadcasted_bins[idx_1, idx_2, indices]
        q_matrix[idx_1, idx_2] += (broadcasted_bins[idx_1, idx_2, indices+1] - broadcasted_bins[idx_1, idx_2, indices]) * \
                                   (threshold - cumsums[idx_1, idx_2, indices-1]) / self.hist_matrix[idx_1, idx_2, indices]
        q_matrix[q_matrix == 0.0] = np.nan
        return q_matrix

    @property
    def quality_map(self):
        """ Spatial matrix to show harder places in the cube. """
        if self._quality_map is None:
            self.make_quality_map([0.1], ['support_js', 'support_hellinger'])
        return self._quality_map

    def make_quality_map(self, quantiles, metric_names, **kwargs):
        """ Create `quality_map` matrix that shows harder places in the cube.

        Parameters
        ----------
        quantiles : sequence of floats
            Quantiles for computing hardness thresholds. Must be in (0, 1) ranges.
        metric_names : sequence or str
            Metrics to compute to assess hardness of cube.
        kwargs : dict
            Other parameters of metric(s) evaluation.
        """
        from .metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_map = GeometryMetrics(self).evaluate('quality_map', quantiles=quantiles, agg=None,
                                                     metric_names=metric_names, **kwargs)
        self._quality_map = quality_map
        return quality_map

    @property
    def quality_grid(self):
        """ Spatial grid based on `quality_map`. """
        if self._quality_grid is None:
            self.make_quality_grid((20, 150))
        return self._quality_grid

    def make_quality_grid(self, frequencies, iline=True, xline=True, margin=0, **kwargs):
        """ Create `quality_grid` based on `quality_map`.

        Parameters
        ----------
        frequencies : sequence of numbers
            Grid frequencies for individual levels of hardness in `quality_map`.
        iline, xline : bool
            Whether to make lines in grid to account for `ilines`/`xlines`.
        margin : int
            Margin of boundaries to not include in the grid.
        kwargs : dict
            Other parameters of grid making.
        """
        from .metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_grid = GeometryMetrics(self).make_grid(self.quality_map, frequencies,
                                                       iline=iline, xline=xline, margin=margin, **kwargs)
        self._quality_grid = quality_grid
        return quality_grid


    # Instance introspection and visualization methods
    def reset_cache(self):
        """ Clear cached slides. """
        if self.structured is False:
            self.load_slide.reset(instance=self)
        else:
            self.file_hdf5.reset()

    @property
    def cache_length(self):
        """ Total amount of cached slides. """
        if self.structured is False:
            length = len(self.load_slide.cache()[self])
        else:
            length = self.file_hdf5.cache_length
        return length

    @property
    def cache_size(self):
        """ Total size of cached slides. """
        if self.structured is False:
            items = self.load_slide.cache()[self].values()
        else:
            items = self.file_hdf5.cache_items

        return sum(item.nbytes / (1024 ** 3) for item in items)

    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        attrs = [
            'dataframe', 'trace_container', 'zero_traces',
            *[attr for attr in self.__dict__
              if 'matrix' in attr or '_quality' in attr],
        ]
        return sum(sys.getsizeof(getattr(self, attr)) for attr in attrs if hasattr(self, attr)) + self.cache_size

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)

    def __repr__(self):
        return 'Inferred geometry for {}: ({}x{}x{})'.format(os.path.basename(self.path), *self.cube_shape)

    def __str__(self):
        msg = f"""
        Geometry for cube              {self.path}
        Current index:                 {self.index_headers}
        Shape:                         {self.cube_shape}
        Time delay and sample rate:    {self.delay}, {self.sample_rate}

        Cube size:                     {os.path.getsize(self.path) / (1024**3):4.3} GB
        Size of the instance:          {self.ngbytes:4.3} GB

        Number of traces:              {np.prod(self.cube_shape[:-1])}
        """
        if hasattr(self, 'zero_traces'):
            msg += f"""Number of non-zero traces:     {np.prod(self.cube_shape[:-1]) - np.sum(self.zero_traces)}
            """

        if self.has_stats:
            msg += f"""
        Num of unique amplitudes:      {len(np.unique(self.trace_container))}
        Mean/std of amplitudes:        {np.mean(self.trace_container):6.6}/{np.std(self.trace_container):6.6}
        Min/max amplitudes:            {self.value_min:6.6}/{self.value_max:6.6}
        q01/q99 amplitudes:            {self.q01:6.6}/{self.q99:6.6}
            """
        return dedent(msg)

    @property
    def axis_names(self):
        """ Names of the axis: multiple headers and `DEPTH` as the last one. """
        return self.index_headers + ['DEPTH']

    def log(self, printer=None):
        """ Log info about cube into desired stream. By default, creates a file next to the cube. """
        if not callable(printer):
            path_log = '/'.join(self.path.split('/')[:-1]) + '/CUBE_INFO.log'
            printer = lambda msg: file_print(msg, path_log)
        printer(str(self))


    def show_snr(self, **kwargs):
        """ Show signal-to-noise map. """
        kwargs = {
            'cmap': 'viridis_r',
            'title': f'Signal-to-noise map of `{self.name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            **kwargs
            }
        matrix = np.log(self.mean_matrix**2 / self.std_matrix**2)
        plot_image(matrix, mode='single', **kwargs)

    def show_slide(self, loc=None, start=None, end=None, step=1, axis=0, zoom_slice=None,
                   n_ticks=5, delta_ticks=100, stable=True, **kwargs):
        """ Show seismic slide in desired place. Works with both SEG-Y and HDF5 files.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        start, end, step : int
            Parameters of slice loading for 1D index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        axis = parse_axis(axis, self.index_headers)
        slide = self.load_slide(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)
        xticks = list(range(slide.shape[0]))
        yticks = list(range(slide.shape[1]))

        if zoom_slice:
            slide = slide[zoom_slice]
            xticks = xticks[zoom_slice[0]]
            yticks = yticks[zoom_slice[1]]

        # Plot params
        if len(self.index_headers) > 1:
            title = f'{self.axis_names[axis]} {loc} out of {self.cube_shape[axis]}'

            if axis in [0, 1]:
                xlabel = self.index_headers[1 - axis]
                ylabel = 'DEPTH'
            else:
                xlabel = self.index_headers[0]
                ylabel = self.index_headers[1]
        else:
            title = '2D seismic slide'
            xlabel = self.index_headers[0]
            ylabel = 'DEPTH'

        xticks = xticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [xticks[-1]]
        xticks = sorted(list(set(xticks)))
        yticks = yticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [yticks[-1]]
        yticks = sorted(list(set(yticks)), reverse=True)

        if len(xticks) > 2 and (xticks[-1] - xticks[-2]) < delta_ticks:
            xticks.pop(-2)
        if len(yticks) > 2 and (yticks[0] - yticks[1]) < delta_ticks:
            yticks.pop(1)

        kwargs = {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'gray',
            'xticks': xticks,
            'yticks': yticks,
            'labeltop': False,
            'labelright': False,
            **kwargs
        }
        plot_image(slide, **kwargs)

    def show_amplitude_hist(self, scaler=None, bins=50, **kwargs):
        """ Show distribution of amplitudes in `trace_container`. Optionally applies chosen `scaler`. """
        data = np.copy(self.trace_container)
        if scaler:
            data = self.scaler(data, mode=scaler)

        kwargs = {
            'title': (f'Amplitude distribution for {self.short_name}' +
                      f'\n Mean/std: {np.mean(data):3.3}/{np.std(data):3.3}'),
            'label': 'Amplitudes histogram',
            'xlabel': 'amplitude',
            'ylabel': 'density',
            **kwargs
        }
        plot_image(data, backend='matplotlib', bins=bins, mode='histogram', **kwargs)


    # Convert HDF5 to SEG-Y
    def make_sgy(self, path_hdf5=None, path_spec=None, postfix='',
                 remove_hdf5=False, zip_result=True, path_segy=None, pbar=False):
        """ Convert POST-STACK HDF5 cube to SEG-Y format with current geometry spec.

        Parameters
        ----------
        path_hdf5 : str
            Path to load hdf5 file from.
        path_spec : str
            Path to load segy file from with geometry spec.
        path_segy : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        """
        path_segy = path_segy or (os.path.splitext(path_hdf5)[0] + postfix + '.sgy')
        if not path_spec:
            if hasattr(self, 'segy_path'):
                path_spec = self.segy_path
            else:
                path_spec = os.path.splitext(self.path) + '.sgy'

        # By default, if path_hdf5 is not provided, `temp.hdf5` next to self.path will be used
        if path_hdf5 is None:
            path_hdf5 = os.path.join(os.path.dirname(self.path), 'temp.hdf5')

        file_hdf5 = StorageHDF5(path_hdf5, mode='r')
        geometry = SeismicGeometry(path_spec)

        segy = geometry.segyfile
        spec = segyio.spec()
        spec.sorting = segyio.TraceSortingFormat.INLINE_SORTING
        spec.format = int(segy.format)
        spec.samples = range(self.depth)

        idx = np.stack(geometry.dataframe.index)
        ilines, xlines = self.load_meta_item('ilines'), self.load_meta_item('xlines')

        i_enc = {num: k for k, num in enumerate(ilines)}
        x_enc = {num: k for k, num in enumerate(xlines)}

        spec.ilines = ilines
        spec.xlines = xlines

        with segyio.create(path_segy, spec) as dst_file:
            # Copy all textual headers, including possible extended
            for i in range(1 + segy.ext_headers):
                dst_file.text[i] = segy.text[i]

            for c, (i, x) in enumerate(tqdm(idx, disable=(not pbar))):
                locs = [i_enc[i], x_enc[x], slice(None)]
                dst_file.header[c] = segy.header[c]
                dst_file.trace[c] = file_hdf5[locs]

            dst_file.bin = segy.bin
            dst_file.bin[segyio.BinField.Traces] = len(idx)

        if remove_hdf5:
            os.remove(path_hdf5)

        if zip_result:
            dir_name = os.path.dirname(os.path.abspath(path_segy))
            file_name = os.path.basename(path_segy)
            shutil.make_archive(os.path.splitext(path_segy)[0], 'zip', dir_name, file_name)

    def cdp_to_lines(self, points):
        """ Convert CDP to lines. """
        inverse_matrix = np.linalg.inv(self.rotation_matrix[:, :2])
        lines = (inverse_matrix @ points.T - inverse_matrix @ self.rotation_matrix[:, 2].reshape(2, -1)).T
        return np.rint(lines)

    def compute_attribute(self, locations=None, window=10, attribute='semblance', device='cpu'):
        """ Compute attribute on cube.

        Parameters
        ----------
        locations : tuple of slices
            slices for each axis of cube to compute attribute. If locations is None,
            attribute will be computed for the whole cube.
        points : np.ndarray
            points where compute the attribute. In other points attribute will be equal to numpy.nan.
        window : int or tuple of ints
            window for the filter.
        stride : int or tuple of ints
            stride to compute attribute
        attribute : str
            name of the attribute

        Returns
        -------
        np.ndarray
            array of the shape corresponding to locations
        """
        if locations is None:
            locations = [slice(0, self.cube_shape[i]) for i in range(3)]
        data = self.file_hdf5[locations]

        return compute_attribute(data, window, device, attribute)

    def create_attribute_hdf5(self, attr, dst, chunk_shape=None, chunk_stride=None, window=10,
                              agg=None, projections='ixh', pbar=False, device='cpu'):
        """ Create hdf5 file from np.ndarray or with geological attribute.

        Parameters
        ----------
        path_hdf5 : str

        src : np.ndarray, iterable or str
            If `str`, must be a name of the attribute to compute.
            If 'iterable, items must be tuples (coord of chunk, chunk).
        chunk_shape : int, tuple or None
            Shape of chunks.
        chunk_stride : int
            Stride for chunks.
        pbar : bool
            Progress bar.
        """
        shape = self.cube_shape

        chunk_shape = fill_defaults(chunk_shape, shape)
        chunk_stride = fill_defaults(chunk_stride, chunk_shape)

        grid = [make_axis_grid((0, shape[i]), chunk_stride[i], shape[i], chunk_shape[i] ) for i in range(3)]

        def _iterator():
            for coord in itertools.product(*grid):
                locations = [slice(coord[i], coord[i] + chunk_shape[i]) for i in range(3)]
                yield coord, self.compute_attribute(locations, window, attribute=attr, device=device)
        chunks = _iterator()
        total = np.prod([len(item) for item in grid])
        chunks = tqdm(chunks, total=total) if pbar else chunks
        return StorageHDF5.create_file_from_iterable(chunks, self.cube_shape, chunk_shape,
                                                     chunk_stride, dst=dst, agg=agg, projection='ixh')

        # self.store_meta(path_meta)

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
    #pylint: disable=attribute-defined-outside-init, too-many-instance-attributes
    def __init__(self, path, headers=None, index_headers=None, **kwargs):
        self.structured = False
        self.dataframe = None
        self.segyfile = None

        self.headers = headers or self.HEADERS_POST
        self.index_headers = index_headers or self.INDEX_POST

        super().__init__(path, **kwargs)


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

        # Create a matrix with ones at fully-zeroes traces
        if self.index_headers == self.INDEX_POST:
            try:
                size = self.depth // 10
                slc = np.stack([self[:, :, i * size] for i in range(1, 10)], axis=-1)
                self.zero_traces = np.zeros(self.lens, dtype=np.int)
                self.zero_traces[np.std(slc, axis=-1) == 0] = 1
            except ValueError: # can't reshape
                pass

        path_meta = os.path.splitext(self.path)[0] + '.meta'
        if os.path.exists(path_meta) and not recollect:
            self.load_meta()
        elif collect_stats:
            self.collect_stats(**kwargs)

        # Store additional segy info, that is preserved in HDF5
        self.segy_path = self.path
        self.segy_text = [self.segyfile.text[i] for i in range(1 + self.segyfile.ext_headers)]
        self.add_rotation_matrix()

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
        self.ranges = [(np.max(item) - np.min(item) + 1) for item in self.uniques]

        self.cube_shape = np.asarray([*self.lens, self.depth])

    def collect_stats(self, spatial=True, bins=25, num_keep=5000, pbar=True, **kwargs):
        """ Pass through file data to collect stats:
            - min/max values.
            - q01/q99 quantiles of amplitudes in the cube.
            - certain amount of traces are stored to `trace_container` attribute.

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

        # Get min/max values, store some of the traces
        trace_container = []
        value_min, value_max = np.inf, -np.inf

        for i in tqdm(range(num_traces), desc='Finding min/max', ncols=1000, disable=(not pbar)):
            trace = self.segyfile.trace[i]

            trace_min, trace_max = find_min_max(trace)
            if trace_min < value_min:
                value_min = trace_min
            if trace_max > value_max:
                value_max = trace_max

            if random() < (num_keep / num_traces) and trace_min != trace_max:
                trace_container.extend(trace.tolist())
                #TODO: add dtype for storing

        # Collect more spatial stats: min, max, mean, std, histograms matrices
        if spatial:
            # Make bins
            bins = np.histogram_bin_edges(None, bins, range=(value_min, value_max)).astype(np.float)
            self.bins = bins

            # Create containers
            min_matrix, max_matrix = np.full(self.lens, np.nan), np.full(self.lens, np.nan)
            hist_matrix = np.full((*self.lens, len(bins)-1), np.nan)

            # Iterate over traces
            description = f'Collecting stats for {self.name}'
            for i in tqdm(range(num_traces), desc=description, ncols=1000, disable=(not pbar)):
                trace = self.segyfile.trace[i]
                header = self.segyfile.header[i]

                # i -> id in a dataframe
                keys = [header.get(field) for field in self.byte_no]
                store_key = [self.uniques_inversed[j][item] for j, item in enumerate(keys)]
                store_key = tuple(store_key)

                # For each trace, we store an entire histogram of amplitudes
                val_min, val_max = find_min_max(trace)
                min_matrix[store_key] = val_min
                max_matrix[store_key] = val_max

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
            self.min_matrix, self.max_matrix = min_matrix, max_matrix
            self.mean_matrix, self.std_matrix = mean_matrix, std_matrix
            self.hist_matrix = hist_matrix
            self.zero_traces = (min_matrix == max_matrix).astype(np.int)
            self.zero_traces[np.isnan(min_matrix)] = 1

        self.value_min, self.value_max = value_min, value_max
        self.trace_container = np.array(trace_container)
        self.q001, self.q01, self.q99, self.q999 = np.quantile(trace_container, [0.001, 0.01, 0.99, 0.999])
        self.has_stats = True
        self.store_meta()

    def add_rotation_matrix(self):
        """ Add transform from INLINE/CROSSLINE corrdinates to CDP system. """
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

        self.rotation_matrix = cv2.getAffineTransform(np.float32(ix_points), np.float32(cdp_points))

    def lines_to_cdp(self, points):
        """ Convert lines to CDP. """
        return (self.rotation_matrix[:, :2] @ points.T + self.rotation_matrix[:, 2].reshape(2, -1)).T

    def compute_area(self, correct=True, shift=50):
        """ Compute approximate area of the cube in square kilometres.

        Parameters
        ----------
        correct : bool
            Whether to correct computed area for zero traces.
        """
        if self.headers != self.HEADERS_POST_FULL:
            raise TypeError('Geometry index must be `POST_FULL`')

        i = self.ilines[self.ilines_len // 2]
        x = self.xlines[self.xlines_len // 2]

        cdp_x, cdp_y = self.dataframe[['CDP_X', 'CDP_Y']].ix[(i, x)]
        cdp_x_delta = abs(self.dataframe[['CDP_X']].ix[(i, x + shift)][0] - cdp_x)
        cdp_y_delta = abs(self.dataframe[['CDP_Y']].ix[(i + shift, x)][0] - cdp_y)

        if cdp_x_delta == 0 and cdp_y_delta == 0:
            cdp_x_delta = abs(self.dataframe[['CDP_X']].ix[(i + shift, x)][0] - cdp_x)
            cdp_y_delta = abs(self.dataframe[['CDP_Y']].ix[(i, x + shift)][0] - cdp_y)

        cdp_x_delta /= shift
        cdp_y_delta /= shift

        ilines_km = cdp_y_delta * self.ilines_len / 1000
        xlines_km = cdp_x_delta * self.xlines_len / 1000
        area = ilines_km * xlines_km

        if correct and hasattr(self, 'zero_traces'):
            area -= (cdp_x_delta / 1000) * (cdp_y_delta / 1000) * np.sum(self.zero_traces)
        return area


    def set_index(self, index_headers, sortby=None):
        """ Change current index to a subset of loaded headers. """
        self.dataframe.reset_index(inplace=True)
        if sortby:
            self.dataframe.sort_values(index_headers, inplace=True, kind='mergesort')# the only stable sorting algorithm
        self.dataframe.set_index(index_headers, inplace=True)
        self.index_headers = index_headers
        self.add_attributes()

    # Methods to load actual data from SEG-Y
    def load_trace(self, index):
        """ Load individual trace from segyfile.
        If passed `np.nan`, returns trace of zeros.
        """
        if not np.isnan(index):
            return self.segyfile.trace.raw[int(index)]
        return self._zero_trace

    def load_traces(self, trace_indices):
        """ Stack multiple traces together. """
        return np.stack([self.load_trace(idx) for idx in trace_indices])


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
            slide = self.segyfile.depth_slice[loc].reshape(self.lens)
        return slide

    def make_slide_indices(self, loc=None, axis=0, start=None, end=None, step=1, stable=True, return_iterator=False):
        """ Choose appropriate version of index creation for various lengths of current index.

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


    def __getitem__(self, key):
        """ Retrieve amplitudes from cube. Uses the usual `Numpy` semantics for indexing 3D array. """
        key_ = list(key)
        if len(key_) != len(self.cube_shape):
            key_ += [slice(None)] * (len(self.cube_shape) - len(key_))

        key, squeeze = [], []
        for i, item in enumerate(key_):
            max_size = self.cube_shape[i]

            if isinstance(item, slice):
                slc = slice(item.start or 0, item.stop or max_size)
            elif isinstance(item, int):
                item = item if item >= 0 else max_size - item
                slc = slice(item, item + 1)
                squeeze.append(i)
            key.append(slc)

        crop = self.load_crop(key)
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    # Convert SEG-Y to HDF5
    def make_hdf5(self, path_hdf5=None, postfix='', unsafe=True, store_meta=True, projections='ixh', pbar=True):
        """ Converts `.segy` cube to `.hdf5` format.

        Parameters
        ----------
        path_hdf5 : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        """

        cube_keys = {'i': 'cube_i', 'x': 'cube_x', 'h': 'cube_h'}
        axes = {'i': [0, 1, 2], 'x': [1, 2, 0], 'h': [2, 0, 1]}

        if self.index_headers != self.INDEX_POST and not unsafe:
            # Currently supports only INLINE/CROSSLINE cubes
            raise TypeError(f'Either set `unsafe=True` or set index to {self.INDEX_POST}')

        path_hdf5 = path_hdf5 or (os.path.splitext(self.path)[0] + postfix + '.hdf5')

        # Remove file, if exists: h5py can't do that
        if os.path.exists(path_hdf5):
            os.remove(path_hdf5)

        # Create file and datasets inside
        with h5py.File(path_hdf5, "w-") as file_hdf5:
            cube_hdf5 = {
                cube_keys[p]: file_hdf5.create_dataset(cube_keys[p], self.cube_shape[axes[p]]) for p in projections
            }

            # Default projection (ilines, xlines, depth) and depth-projection (depth, ilines, xlines)
            total = 0
            if 'i' in projections:
                total += self.cube_shape[0]
            if 'x' in projections:
                total += self.cube_shape[1]
            progress_bar = tqdm(total=total, ncols=1000, disable=(not pbar))

            progress_bar.set_description(f'Converting {self.long_name}; ilines projection')
            for i in range(self.cube_shape[0]):
                slide = self.load_slide(i, stable=False)
                if 'i' in projections:
                    cube_hdf5['cube_i'][i, :, :] = slide.reshape(1, self.cube_shape[1], self.cube_shape[2])
                if 'h' in projections:
                    cube_hdf5['cube_h'][:, i, :] = slide.T
                progress_bar.update()

            # xline-oriented projection: (xlines, depth, ilines)
            if 'x' in projections:
                progress_bar.set_description(f'Converting {self.long_name} to hdf5; xlines projection')
                for x in range(self.cube_shape[1]):
                    slide = self.load_slide(x, axis=1, stable=False).T
                    cube_hdf5['cube_x'][x, :, :,] = slide
                    progress_bar.update()
            progress_bar.close()

        if store_meta:
            if not self.has_stats:
                self.collect_stats(pbar=pbar)

            path_meta = os.path.splitext(path_hdf5)[0] + '.meta'
            self.store_meta(path_meta)



    # Convenient alias
    convert_to_hdf5 = make_hdf5

class SeismicGeometryHDF5(SeismicGeometry):
    """ Class to infer information about HDF5 cubes and provide convenient methods of working with them.

    In order to initialize instance, one must supply `path` to the HDF5 cube.

    All the attributes are loaded directly from HDF5 file itself, so most of the attributes from SEG-Y file
    are preserved, with the exception of `dataframe` and `uniques`.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, path, **kwargs):
        self.structured = True
        self.file_hdf5 = None

        super().__init__(path, **kwargs)

    def process(self, **kwargs):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        _ = kwargs
        self.file_hdf5 = StorageHDF5(self.path, mode='r') # h5py.File(self.path, mode='r')
        self.add_attributes()

    def add_attributes(self):
        """ Store values from `hdf5` file to attributes. """
        self.index_headers = self.INDEX_POST
        self.load_meta()
        if hasattr(self, 'lens'):
            self.cube_shape = np.asarray([self.ilines_len, self.xlines_len, self.depth]) # BC
        else:
            self.cube_shape = self.file_hdf5.shape
            self.lens = self.cube_shape
        self.has_stats = True

    # Methods to load actual data from HDF5
    def load_crop(self, locations, axis=None, **kwargs):
        """ Load 3D crop from the cube.
        Automatically chooses the fastest axis to use: as `hdf5` files store multiple copies of data with
        various orientations, some axis are faster than others depending on exact crop location and size.

        Parameters
        locations : sequence of slices
            Location to load: slices along the first index, the second, and depth.
        axis : str or int
            Identificator of the axis to use to load data.
            Can be `iline`, `xline`, `height`, `depth`, `i`, `x`, `h`, 0, 1, 2.
        """
        return self.file_hdf5.load_crop(locations, axis, **kwargs)

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis. """
        return self.file_hdf5.load_slide(loc, axis, **kwargs)

    def __getitem__(self, key):
        """ Retrieve amplitudes from cube. Uses the usual `Numpy` semantics for indexing 3D array. """
        return self.file_hdf5[key]



class SeismicGeometryNPZ(SeismicGeometry):
    """ Create a Geometry instance from a `numpy`-saved file. Stores everything in memory.
    Can simultaneously work with multiple type of cube attributes, e.g. amplitudes, GLCM, RMS, etc.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, path, **kwargs):
        self.structured = True
        self.file_npz = None
        self.names = None
        self.data = {}

        super().__init__(path, **kwargs)

    def process(self, order=(0, 1, 2), **kwargs):
        """ Create all the missing attributes. """
        self.index_headers = SeismicGeometry.INDEX_POST
        self.file_npz = np.load(self.path, allow_pickle=True, mmap_mode='r')

        self.names = list(self.file_npz.keys())
        self.data = {key : np.transpose(self.file_npz[key], order) for key in self.names}

        data = self.data[self.names[0]]
        self.cube_shape = np.array(data.shape)
        self.lens = self.cube_shape[:2]
        self.zero_traces = np.zeros(self.lens)

        # Attributes
        self.depth = self.cube_shape[2]
        self.delay, self.sample_rate = 0, 0
        self.value_min = np.min(data)
        self.value_max = np.max(data)
        self.q001, self.q01, self.q99, self.q999 = np.quantile(data, [0.001, 0.01, 0.99, 0.999])


    # Methods to load actual data from NPZ
    def load_crop(self, locations, names=None, **kwargs):
        """ Load 3D crop from the cube.

        Parameters
        locations : sequence of slices
            Location to load: slices along the first index, the second, and depth.
        names : sequence
            Names of data attributes to load.
        """
        _ = kwargs
        names = names or self.names[:1]
        shape = np.array([(slc.stop - slc.start) for slc in locations])
        axis = np.argmin(shape)

        crops = [self.data[key][locations[0], locations[1], locations[2]] for key in names]
        crop = np.concatenate(crops, axis=axis)
        return crop

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis. """
        _ = kwargs
        locations = self.make_slide_locations(loc, axis)
        crop = self.load_crop(locations, names=['data'])
        return crop.squeeze()

    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        return sum(sys.getsizeof(self.data[key]) for key in self.names)

    def __getattr__(self, key):
        """ Use default `object` getattr, without `.meta` magic. """
        return object.__getattribute__(self, key)

    def __getitem__(self, key):
        """ Get data from the first named array. """
        return self.data[self.names[0]][key]
