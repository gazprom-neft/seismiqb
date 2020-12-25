""" Horizon class and metrics. """
#pylint: disable=too-many-lines, import-error
import os
from copy import copy
from textwrap import dedent
from itertools import product

import numpy as np
import pandas as pd
import h5py
from numba import njit, prange

import cv2
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from scipy.ndimage import find_objects
from scipy.spatial import Delaunay
from scipy.signal import hilbert
from skimage.measure import label

from .utils import round_to_array, groupby_mean, groupby_min, groupby_max, HorizonSampler, filter_simplices, lru_cache
from .utils import make_gaussian_kernel, retrieve_function_arguments
from .plotters import plot_image, show_3d


class UnstructuredHorizon:
    """  Contains unstructured horizon.

    Initialized from `storage` and `geometry`, where `storage` is a csv-like file.

    The main inner storage is `dataframe`, that is a mapping from indexing headers to horizon depth.
    Since the index of that dataframe is the same, as the index of dataframe of a geometry, these can be combined.

    UnstructuredHorizon provides following features:
        - Method `add_to_mask` puts 1's on the location of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - Few of visualization methods: view from above, slices along iline/xline axis, etc.

    There is a lazy method creation in place: the first person who needs them would need to code them.
    """
    CHARISMA_SPEC = ['INLINE', '_', 'INLINE_3D', 'XLINE', '__', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y', 'height']
    REDUCED_CHARISMA_SPEC = ['INLINE_3D', 'CROSSLINE_3D', 'height']

    FBP_SPEC = ['FieldRecord', 'TraceNumber', 'file_id', 'FIRST_BREAK_TIME']

    def __init__(self, storage, geometry, name=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.format = None

        # Storage
        self.dataframe = None
        self.attached = False

        # Heights information
        self.h_min, self.h_max = None, None
        self.h_mean, self.h_std = None, None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, pd.DataFrame):
            # points-like dataframe
            self.format = 'dataframe'

        elif isinstance(storage, np.ndarray) and storage.ndim == 2 and storage.shape[1] == 3:
            # array with row in (iline, xline, height) format
            self.format = 'points'

        getattr(self, 'from_{}'.format(self.format))(storage, **kwargs)


    def from_points(self, points, **kwargs):
        """ Not needed. """


    def from_dataframe(self, dataframe, attach=True, height_prefix='height', transform=False):
        """ Load horizon data from dataframe.

        Parameters
        ----------
        attack : bool
            Whether to store horizon data in common dataframe inside `geometry` attributes.
        transform : bool
            Whether to transform line coordinates to the cubic ones.
        height_prefix : str
            Column name with height.
        """
        if transform:
            dataframe[height_prefix] = (dataframe[height_prefix] - self.geometry.delay) / self.geometry.sample_rate
        dataframe.rename(columns={height_prefix: self.name}, inplace=True)
        dataframe.set_index(self.geometry.index_headers, inplace=True)
        self.dataframe = dataframe

        self.h_min, self.h_max = self.dataframe.min().values[0], self.dataframe.max().values[0]
        self.h_mean, self.h_std = self.dataframe.mean().values[0], self.dataframe.std().values[0]

        if attach:
            self.attach()


    def from_file(self, path, names=None, columns=None, height_prefix='height', reader_params=None, **kwargs):
        """ Init from path to csv-like file.

        Parameters
        ----------
        names : sequence of str
            Names of columns in file.
        columns : sequence of str
            Names of columns to actually load.
        height_prefix : str
            Column name with height.
        reader_params : None or dict
            Additional parameters for file reader.
        """
        #pylint: disable=anomalous-backslash-in-string
        _ = kwargs
        if names is None:
            with open(path) as file:
                line_len = len(file.readline().split(' '))
            if line_len == 3:
                names = UnstructuredHorizon.REDUCED_CHARISMA_SPEC
            elif line_len == 9:
                names = UnstructuredHorizon.CHARISMA_SPEC
        columns = columns or self.geometry.index_headers + [height_prefix]

        self.path = path
        self.name = os.path.basename(path) if self.name is None else self.name

        defaults = {'sep': '\s+'}
        reader_params = reader_params or {}
        reader_params = {**defaults, **reader_params}
        df = pd.read_csv(path, names=names, usecols=columns, **reader_params)

        # Convert coordinates of horizons to the one that present in cube geometry
        # df[columns] = np.rint(df[columns]).astype(np.int64)
        df[columns] = df[columns].astype(np.int64)
        for i, idx in enumerate(self.geometry.index_headers):
            df[idx] = round_to_array(df[idx].values, self.geometry.uniques[i])

        self.from_dataframe(df, transform=True, height_prefix=columns[-1])

    def attach(self):
        """ Store horizon data in common dataframe inside `geometry` attributes. """
        if not hasattr(self.geometry, 'horizons'):
            self.geometry.horizons = pd.DataFrame(index=self.geometry.dataframe.index)

        self.geometry.horizons = pd.merge(self.geometry.horizons, self.dataframe,
                                          left_index=True, right_index=True,
                                          how='left')
        self.attached = True


    def add_to_mask(self, mask, locations=None, width=3, alpha=1, iterator=None, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : sequence of arrays
            List of desired locations to load: along the first index, the second, and depth.
        width : int
            Width of an added horizon.
        iterator : None or sequence
            If provided, indices to use to load height from dataframe.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        shift_1, shift_2, h_min = [slc.start for slc in locations]
        h_max = locations[-1].stop

        if iterator is None:
            # Usual case
            iterator = list(product(*[[self.geometry.uniques[idx][i]
                                       for i in range(locations[idx].start, locations[idx].stop)]
                                      for idx in range(2)]))
            idx_iterator = np.array(list(product(*[list(range(slc.start, slc.stop)) for slc in locations[:2]])))
            idx_1 = idx_iterator[:, 0] - shift_1
            idx_2 = idx_iterator[:, 1] - shift_2

        else:
            #TODO: remove this and make separate method inside `SeismicGeometry` for loading data with same iterator
            #TODO: think about moving horizons to `geometry` attributes altogether..
            # Currently, used in `show_slide` only:
            axis = np.argmin(np.array([len(np.unique(np.array(iterator)[:, idx])) for idx in range(2)]))
            loc = iterator[axis][0]
            other_axis = 1 - axis

            others = self.geometry.dataframe[self.geometry.dataframe.index.get_level_values(axis) == loc]
            others = others.index.get_level_values(other_axis).values
            others_iterator = np.array([np.where(others == item[other_axis])[0][0] for item in iterator])

            idx_1 = np.zeros_like(others_iterator) if axis == 0 else others_iterator
            idx_2 = np.zeros_like(others_iterator) if axis == 1 else others_iterator


        heights = self.dataframe[self.name].reindex(iterator, fill_value=np.nan).values.astype(np.int32)

        # Filter labels based on height
        heights_mask = np.asarray((np.isnan(heights) == False) & # pylint: disable=singleton-comparison
                                  (heights >= h_min + low) &
                                  (heights <= h_max - high)).nonzero()[0]

        idx_1 = idx_1[heights_mask]
        idx_2 = idx_2[heights_mask]
        heights = heights[heights_mask]
        heights -= (h_min + low)

        # Place values on current heights and shift them one unit below.
        for _ in range(width):
            mask[idx_1, idx_2, heights] = alpha
            heights += 1
        return mask

    # Methods to implement in the future
    def filter_points(self, **kwargs):
        """ Remove points that correspond to bad traces, e.g. zero traces. """

    def merge(self, **kwargs):
        """ Merge two instances into one. """

    def get_cube_values(self, **kwargs):
        """ Get cube values along the horizon.
        Can be easily done via subsequent `segyfile.depth_slice` and `reshape`.
        """

    def dump(self, **kwargs):
        """ Save the horizon to the disk. """

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.cube_name}
        Depths range:           {self.h_min} to {self.h_max}
        Depths mean:            {self.h_mean:.6}
        Depths std:             {self.h_std:.6}
        Length:                 {len(self.dataframe)}
        """
        return dedent(msg)


    # Visualization
    def show_slide(self, loc, width=3, axis=0, stable=True, order_axes=None, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])

        # Create the same indices, as for seismic slide loading
        #TODO: make slide indices shareable
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis, stable=stable)
        _, iterator = self.geometry.make_slide_indices(loc=loc, axis=axis, stable=stable, return_iterator=True)
        shape[1 - axis] = -1

        # Create mask with horizon
        mask = np.zeros_like(seismic_slide.reshape(shape))
        mask = self.add_to_mask(mask, locations, width=width, iterator=iterator if stable else None)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)

        # set defaults if needed and plot the slide
        kwargs = {
            'mode': 'overlap',
            'title': (f'U-horizon `{self.name}` on `{self.geometry.name}`' + '\n ' +
                      f'{self.geometry.index_headers[axis]} {loc} out of {self.geometry.lens[axis]}'),
            'xlabel': self.geometry.index_headers[1 - axis],
            'ylabel': 'Depth', 'y': 1.015,
            **kwargs
        }
        plot_image([seismic_slide, mask], order_axes=order_axes, **kwargs)



class Horizon:
    """ Contains spatially-structured horizon: each point describes a height on a particular (iline, xline).

    Initialized from `storage` and `geometry`.

    Storage can be one of:
        - csv-like file in CHARISMA or REDUCED_CHARISMA format.
        - ndarray of (N, 3) shape.
        - ndarray of (ilines_len, xlines_len) shape.
        - dictionary: a mapping from (iline, xline) -> height.
        - mask: ndarray of (ilines_len, xlines_len, depth) with 1's at places of horizon location.

    Main storages are `matrix` and `points` attributes: they are loaded lazily at the time of first access.
        - `matrix` is a depth map, ndarray of (ilines_len, xlines_len) shape with each point
          corresponding to horizon height at this point. Note that shape of the matrix is generally smaller
          than cube spatial range: that allows to save space.
          Attributes `i_min` and `x_min` describe position of the matrix in relation to the cube spatial range.
          Each point with absent horizon is filled with `FILL_VALUE`.
          Note that since the dtype of `matrix` is `np.int32`, we can't use `np.nan` as the fill value.
          In order to initialize from this storage, one must supply `matrix`, `i_min`, `x_min`.

        - `points` is a (N, 3) ndarray with every row being (iline, xline, height). Note that (iline, xline) are
          stored in cube coordinates that range from 0 to `ilines_len` and 0 to `xlines_len` respectively.
          Stored height is corrected on `time_delay` and `sample_rate` of the cube.
          In order to initialize from this storage, one must supply (N, 3) ndarray.

    Independently of type of initial storage, Horizon provides following:
        - Attributes `i_min`, `x_min`, `i_max`, `x_max`, `h_min`, `h_max`, `h_mean`, `h_std`, `bbox`,
          to completely describe location of the horizon in the 3D volume of the seismic cube.

        - Convenient methods of changing the horizon: `apply_to_matrix` and `apply_to_points`:
          these methods must be used instead of manually permuting `matrix` and `points` attributes.
          For example, filtration or smoothing of a horizon can be done with their help.

        - `sampler` instance to generate points that are close to the horizon.
          Note that these points are scaled into [0, 1] range along each of the coordinate.

        - Method `add_to_mask` puts 1's on the location of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - `evaluate` allows to quickly assess the quality of a seismic reflection;
         for more metrics, check :class:`~.HorizonMetrics`.

        - A number of properties that describe geometrical, geological and mathematical characteristics of a horizon.
          For example, `borders_matrix` and `boundaries_matrix`: the latter containes outer and inner borders;
          `coverage` is the ratio between labeled traces and non-zero traces in the seismic cube;
          `solidity` is the ratio between labeled traces and traces inside the hull of the horizon;
          `perimeter` and `number_of_holes` speak for themselves.

        - Multiple instances of Horizon can be compared against one another and, if needed,
          merged into one (either in-place or not) via `check_proximity`, `overlap_merge`, `adjacent_merge` methods.

        - A wealth of visualization methods: view from above, slices along iline/xline axis, etc.
    """
    #pylint: disable=too-many-public-methods, import-outside-toplevel

    # CHARISMA: default seismic format of storing surfaces inside the 3D volume
    CHARISMA_SPEC = ['INLINE', '_', 'iline', 'XLINE', '__', 'xline', 'cdp_x', 'cdp_y', 'height']

    # REDUCED_CHARISMA: CHARISMA without redundant columns
    REDUCED_CHARISMA_SPEC = ['iline', 'xline', 'height']

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999

    # Correspondence between attribute alias and the class function that calculates it
    METHOD_TO_ATTRIBUTE = {
        'get_cube_values': ['cube_values', 'amplitudes'],
        'get_full_matrix': ['full_matrix', 'heights'],
        'evaluate_metric': ['metrics'],
        'get_instantaneous_phases': ['instant_phases'],
        'get_instantaneous_amplitudes': ['instant_amplitudes'],
        'get_full_binary_matrix': ['full_binary_matrix', 'masks']
    }
    ATTRIBUTE_TO_METHOD = {attr: func for func, attrs in METHOD_TO_ATTRIBUTE.items() for attr in attrs}


    def __init__(self, storage, geometry, name=None, dtype=np.int32, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.dtype = dtype
        self.format = None

        # Location of the horizon inside cube spatial range
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.bbox = None
        self._len = None

        # Underlying data storages
        self._matrix = None
        self._points = None
        self._depths = None

        # Heights information
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None
        self._horizon_metrics = None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name
        self.cube_shape = geometry.cube_shape

        self.sampler = None

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, dict):
            # mapping from (iline, xline) to (height)
            self.format = 'dict'

        elif isinstance(storage, np.ndarray):
            if storage.ndim == 2 and storage.shape[1] == 3:
                # array with row in (iline, xline, height) format
                self.format = 'points'

            elif storage.ndim == 2 and (storage.shape == self.cube_shape[:-1]).all():
                # matrix of (iline, xline) shape with every value being height
                self.format = 'full_matrix'

            elif storage.ndim == 2:
                # matrix of (iline, xline) shape with every value being height
                self.format = 'matrix'

        getattr(self, 'from_{}'.format(self.format))(storage, **kwargs)


    @property
    def points(self):
        """ Storage of horizon data as (N, 3) array of (iline, xline, height) in cubic coordinates.
        If the horizon is created not from (N, 3) array, evaluated at the time of the first access.
        """
        if self._points is None and self.matrix is not None:
            points = self.matrix_to_points(self.matrix)
            points += np.array([self.i_min, self.x_min, 0])
            self._points = points
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.nonzero(matrix != Horizon.FILL_VALUE)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points


    @property
    def matrix(self):
        """ Storage of horizon data as depth map: matrix of (ilines_length, xlines_length) with each point
        corresponding to height. Matrix is shifted to a (i_min, x_min) point so it takes less space.
        If the horizon is created not from matrix, evaluated at the time of the first access.
        """
        if self._matrix is None and self.points is not None:
            self._matrix = self.points_to_matrix(self.points, self.i_min, self.x_min,
                                                 self.i_length, self.x_length, self.dtype)
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @staticmethod
    def points_to_matrix(points, i_min, x_min, i_length, x_length, dtype=np.int32):
        """ Convert array of (N, 3) shape to a depth map (matrix). """
        matrix = np.full((i_length, x_length), Horizon.FILL_VALUE, dtype)
        matrix[points[:, 0].astype(np.int32) - i_min,
               points[:, 1].astype(np.int32) - x_min] = points[:, 2]
        return matrix

    @property
    def depths(self):
        """ Array of depth only. Useful for faster stats computation when initialized from a matrix. """
        if self._depths is None:
            if self._points is not None:
                self._depths = self.points[:, -1]
            else:
                self._depths = self.matrix[self.matrix != self.FILL_VALUE]
        return self._depths


    @property
    def h_min(self):
        """ Minimum depth value. """
        if self._h_min is None:
            self._h_min = np.min(self.depths)
        return self._h_min

    @property
    def h_max(self):
        """ Maximum depth value. """
        if self._h_max is None:
            self._h_max = np.max(self.depths)
        return self._h_max

    @property
    def h_mean(self):
        """ Average depth value. """
        if self._h_mean is None:
            self._h_mean = np.mean(self.depths)
        return self._h_mean

    @property
    def h_std(self):
        """ Std of depths. """
        if self._h_std is None:
            self._h_std = np.std(self.depths)
        return self._h_std

    def __len__(self):
        """ Number of labeled traces. """
        if self._len is None:
            if self._points is not None:
                self._len = len(self.points)
            else:
                self._len = len(self.depths)
        return self._len

    def reset_storage(self, storage=None):
        """ Reset storage along with depth-wise stats. """
        self._depths = None
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None
        self._len = None

        if storage == 'matrix':
            self._matrix = None
        elif storage == 'points':
            self._points = None

    # Coordinate transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] -= self.geometry.ilines_offset
        array[:, 1] -= self.geometry.xlines_offset
        array[:, 2] -= self.geometry.delay
        array[:, 2] /= self.geometry.sample_rate
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array = array.astype(float)
        array[:, 0] += self.geometry.ilines_offset
        array[:, 1] += self.geometry.xlines_offset
        array[:, 2] *= self.geometry.sample_rate
        array[:, 2] += self.geometry.delay
        return array


    # Initialization from different containers
    def from_points(self, points, transform=False, verify=True, **kwargs):
        """ Base initialization: from point cloud array of (N, 3) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: two spatial coordinates and depth.
        transform : bool
            Whether transform from line coordinates (ilines, xlines) to cubic system.
        verify : bool
            Whether to remove points outside of the cube range.
        """
        _ = kwargs

        # Transform to cubic coordinates, if needed
        if transform:
            points = self.lines_to_cubic(points)
        if verify:
            idx = np.where((points[:, 0] >= 0) &
                           (points[:, 1] >= 0) &
                           (points[:, 2] >= 0) &
                           (points[:, 0] < self.cube_shape[0]) &
                           (points[:, 1] < self.cube_shape[1]) &
                           (points[:, 2] < self.cube_shape[2]))[0]
            points = points[idx]

        if self.dtype == np.int32:
            points = np.rint(points)
        self.points = points.astype(self.dtype)

        # Collect stats on separate axes. Note that depth stats are properties
        self.reset_storage('matrix')
        if len(self.points) > 0:
            self.i_min, self.x_min, _ = np.min(self.points, axis=0).astype(np.int32)
            self.i_max, self.x_max, _ = np.max(self.points, axis=0).astype(np.int32)
            self._h_min = self.points[:, 2].min().astype(self.dtype)
            self._h_max = self.points[:, 2].max().astype(self.dtype)

            self.i_length = (self.i_max - self.i_min) + 1
            self.x_length = (self.x_max - self.x_min) + 1
            self.bbox = np.array([[self.i_min, self.i_max],
                                  [self.x_min, self.x_max],
                                  [self.h_min, self.h_max]],
                                 dtype=np.int32)


    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file. """
        _ = kwargs

        self.path = path
        self.name = os.path.basename(path) if self.name is None else self.name
        points = self.file_to_points(path)
        self.from_points(points, transform, **kwargs)

    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path) as file:
            line_len = len(file.readline().split(' '))
        if line_len == 3:
            names = Horizon.REDUCED_CHARISMA_SPEC
        elif line_len >= 9:
            names = Horizon.CHARISMA_SPEC
        else:
            raise ValueError('Horizon labels must be in CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep='\s+', names=names, usecols=Horizon.COLUMNS)
        df.sort_values(Horizon.COLUMNS, inplace=True)
        return df.values


    def from_matrix(self, matrix, i_min, x_min, length=None, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        self.matrix = matrix
        self.i_min, self.x_min = i_min, x_min
        self.i_max, self.x_max = i_min + matrix.shape[0] - 1, x_min + matrix.shape[1] - 1

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max],
                              [self.h_min, self.h_max]],
                             dtype=np.int32)

        self.reset_storage('points')
        self._len = length


    def from_full_matrix(self, matrix, **kwargs):
        """ Init from matrix that covers the whole cube. """
        kwargs = {
            'i_min': 0,
            'x_min': 0,
            **kwargs
        }
        self.from_matrix(matrix, **kwargs)


    def from_dict(self, dictionary, transform=True, **kwargs):
        """ Init from mapping from (iline, xline) to depths. """
        _ = kwargs

        points = self.dict_to_points(dictionary)
        self.from_points(points, transform=transform)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    @staticmethod
    def from_mask(mask, grid_info=None, geometry=None, shifts=None,
                  mode='mean', threshold=0.5, minsize=0, prefix='predict', **kwargs):
        """ Convert mask to a list of horizons.
        Returned list is sorted on length of horizons.

        Parameters
        ----------
        grid_info : dict
            Information about mask creation parameters. Required keys are `geom` and `range`
            to infer geometry and leftmost upper point, or they can be passed directly.
            If not provided, same entities must be passed as arguments `geometry` and `shifts`.
        threshold : float
            Parameter of mask-thresholding.
        mode : str
            Method used for finding the point of a horizon for each iline, xline.
        minsize : int
            Minimum length of a horizon to be saved.
        prefix : str
            Name of horizon to use.
        """
        _ = kwargs
        if grid_info is not None:
            geometry = grid_info['geometry']
            shifts = np.array([item[0] for item in grid_info['range']])

        if geometry is None or shifts is None:
            raise TypeError('Pass `grid_info` or `geometry` and `shifts` to `from_mask` method of Horizon creation.')

        if mode in ['mean', 'avg']:
            group_function = groupby_mean
        elif mode in ['min']:
            group_function = groupby_min
        elif mode in ['max']:
            group_function = groupby_max

        # Labeled connected regions with an integer
        labeled = label(mask >= threshold)
        objects = find_objects(labeled)

        # Create an instance of Horizon for each separate region
        horizons = []
        for i, sl in enumerate(objects):
            max_possible_length = 1
            for j in range(3):
                max_possible_length *= sl[j].stop - sl[j].start

            if max_possible_length >= minsize:
                indices = np.nonzero(labeled[sl] == i + 1)

                if len(indices[0]) >= minsize:
                    coords = np.vstack([indices[i] + sl[i].start for i in range(3)]).T

                    points = group_function(coords) + shifts
                    horizons.append(Horizon(points, geometry, name=f'{prefix}_{i}'))

        horizons.sort(key=len)
        return horizons


    # Functions to use to change the horizon
    def apply_to_matrix(self, function, **kwargs):
        """ Apply passed function to matrix storage.
        Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to matrix storage directly.
            Can return either new_matrix, new_i_min, new_x_min or new_matrix only.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        result = function(self.matrix, **kwargs)
        if isinstance(result, tuple) and len(result) == 3:
            matrix, i_min, x_min = result
        else:
            matrix, i_min, x_min = result, self.i_min, self.x_min
        self.matrix, self.i_min, self.x_min = matrix, i_min, x_min

        self.reset_storage('points') # applied to matrix, so we need to re-create points

    def apply_to_points(self, function, **kwargs):
        """ Apply passed function to points storage.
        Automatically synchronizes the instance after.

        Parameters
        ----------
        function : callable
            Applied to points storage directly.
        kwargs : dict
            Additional arguments to pass to the function.
        """
        self.points = function(self.points, **kwargs)
        self.reset_storage('matrix') # applied to points, so we need to re-create matrix


    def filter_points(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from points storage. """
        if filtering_matrix is None:
            filtering_matrix = self.geometry.zero_traces

        def filtering_function(points, **kwds):
            _ = kwds
            return _filtering_function(points, filtering_matrix)

        self.apply_to_points(filtering_function, **kwargs)

    def filter_matrix(self, filtering_matrix=None, **kwargs):
        """ Remove points that correspond to 1's in `filtering_matrix` from matrix storage. """
        if filtering_matrix is None:
            filtering_matrix = self.geometry.zero_traces

        idx_i, idx_x = np.asarray(filtering_matrix[self.i_min:self.i_max + 1,
                                                   self.x_min:self.x_max + 1] == 1).nonzero()

        def filtering_function(matrix, **kwds):
            _ = kwds
            matrix[idx_i, idx_x] = self.FILL_VALUE
            return matrix

        self.apply_to_matrix(filtering_function, **kwargs)

    filter = filter_points


    def thin_out(self, factor=1, threshold=256):
        """ Thin out the horizon by keeping only each `factor`-th line.

        Parameters
        ----------
        factor : integer or sequence of two integers
            Frequency of lines to keep along ilines and xlines direction.
        threshold : integer
            Minimal amount of points in a line to keep.
        """
        if isinstance(factor, int):
            factor = (factor, factor)

        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        mask_i = np.isin(self.points[:, 0], uniques[counts > threshold][::factor[0]])

        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        mask_x = np.isin(self.points[:, 1], uniques[counts > threshold][::factor[1]])

        self.points = self.points[mask_i + mask_x]
        self.reset_storage('matrix')

    def smooth_out(self, kernel_size=3, sigma=0.8, iters=1, preserve_borders=True, **kwargs):
        """ Convolve the horizon with gaussian kernel with special treatment to absent points:
        if the point was present in the original horizon, then it is changed to a weighted sum of all
        present points nearby;
        if the point was absent in the original horizon and there is at least one non-fill point nearby,
        then it is changed to a weighted sum of all present points nearby.

        Parameters
        ----------
        kernel_size : int
            Size of gaussian filter.
        sigma : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        iters : int
            Number of times to apply smoothing filter.
        preserve_borders : bool
            Whether or not to allow method label additional points.
        """
        def smoothing_function(src, **kwds):
            _ = kwds
            gaussian_kernel = make_gaussian_kernel(kernel_size, sigma)

            smoothed = src
            for _ in range(iters):
                smoothed = _smoothing_function(smoothed, gaussian_kernel, self.FILL_VALUE, False, np.inf)
                smoothed = np.rint(smoothed).astype(np.int32)

            if preserve_borders:
                # pylint: disable=invalid-unary-operand-type
                idx_i, idx_x = np.asarray(~self.filled_matrix).nonzero()
                smoothed[idx_i, idx_x] = self.FILL_VALUE
            return smoothed

        self.apply_to_matrix(smoothing_function, **kwargs)


    # Horizon usage: point/mask generation
    def create_sampler(self, bins=None, quality_grid=None, **kwargs):
        """ Create sampler based on horizon location.

        Parameters
        ----------
        bins : sequence
            Size of ticks alongs each respective axis.
        quality_grid : ndarray or None
            If not None, then must be a matrix with zeroes in locations to keep, ones in locations to remove.
            Applied to `points` before sampler creation.
        """
        _ = kwargs
        default_bins = self.cube_shape // np.array([5, 20, 20])
        bins = bins if bins is not None else default_bins
        quality_grid = self.geometry.quality_grid if quality_grid is True else quality_grid

        if isinstance(quality_grid, np.ndarray):
            points = _filtering_function(np.copy(self.points), 1 - quality_grid)
        else:
            points = self.points

        self.sampler = HorizonSampler(np.histogramdd(points/self.cube_shape, bins=bins))

    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : ndarray
            Where the mask is located.
        width : int
            Width of an added horizon.
        alpha : number
            Value to fill background with at horizon location.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        #TODO: add clear explanation about usage of advanced index in Horizon
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)

        if i_max > i_min and x_max > x_min:
            overlap = self.matrix[i_min - self.i_min : i_max - self.i_min,
                                  x_min - self.x_min : x_max - self.x_min]

            # Coordinates of points to use in overlap local system
            idx_i, idx_x = np.asarray((overlap != self.FILL_VALUE) &
                                      (overlap >= mask_h_min + low) &
                                      (overlap <= mask_h_max - high)).nonzero()
            heights = overlap[idx_i, idx_x]

            # Convert coordinates to mask local system
            idx_i += i_min - mask_i_min
            idx_x += x_min - mask_x_min
            heights -= (mask_h_min + low)

            for shift in range(width):
                mask[idx_i, idx_x, heights + shift] = alpha
        return mask


    def transform_where_present(self, array, normalize=None, fill_value=None, shift=None, rescale=None):
        """ Normalize array where horizon is present, fill with constant where the horizon is absent.

        Parameters
        ----------
        array : np.array
            Matrix of (cube_ilines, cube_xlines, ...) shape.
        normalize : 'min-max', 'mean-std', 'shift-rescale' or None
            Normalization mode for data where `presence_matrix` is True.
            If None, no normalization applied. Defaults to None.
        fill_value : number
            Value to fill `array` in where :attr:`.presence_matrix` is False. Must be compatible with `array.dtype`.
            If None, no filling applied. Defaults to None.
        shift, rescale : number, optional
            For 'shift-rescale` normalization mode.
        """

        if normalize is None and fill_value is None:
            return array

        values = array[self.presence_matrix]

        if normalize is None:
            pass
        elif normalize == 'min-max':
            min_, max_ = values.min(), values.max()
            array = (array - min_) / (max_ - min_)
        elif normalize == 'mean-std':
            mean, std = values.mean(), values.std()
            array = (array - mean) / std
        elif normalize == 'shift-rescale':
            array = (array + shift) * rescale
        else:
            raise ValueError('Unknown normalize mode {}'.format(normalize))

        if fill_value is not None:
            array[~self.presence_matrix] = fill_value
        return array


    @lru_cache(maxsize=1, apply_by_default=False)
    def get_cube_values(self, window=23, offset=0, chunk_size=256, **kwargs):
        """ Get values from the cube along the horizon.

        Parameters
        ----------
        window : int
            Width of data to cut.
        offset : int
            Value to add to each entry in matrix.
        scale : bool, callable
            If True, then values are scaled to [0, 1] range.
            If callable, then it is applied to data cropped along horizon.
        chunk_size : int
            Size of data along height axis processed at a time.
        nan_zero_traces : bool
            Whether fill zero traces with nans or not.
            Defaults to True.
        kwargs :
            Passed directly to :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        low = window // 2
        high = max(window - low, 0)
        chunk_size = min(chunk_size, self.h_max - self.h_min + window)

        background = np.zeros((self.geometry.ilines_len, self.geometry.xlines_len, window), dtype=np.float32)

        for h_start in range(max(low, self.h_min), self.h_max + 1, chunk_size):
            h_end = min(h_start + chunk_size, self.h_max + 1)

            # Get chunk from the cube (depth-wise)
            data_chunk = self.geometry[:, :, (h_start - low) : min(h_end + high, self.geometry.depth)]

            # Check which points of the horizon are in the current chunk (and present)
            idx_i, idx_x = np.asarray((self.matrix != self.FILL_VALUE) &
                                      (self.matrix >= h_start) &
                                      (self.matrix < h_end)).nonzero()
            heights = self.matrix[idx_i, idx_x]

            # Convert spatial coordinates to cubic, convert height to current chunk local system
            idx_i += self.i_min
            idx_x += self.x_min
            heights -= (h_start - offset)

            # Subsequently add values from the cube to background, then shift horizon 1 unit lower
            for j in range(window):
                background[idx_i, idx_x, np.full_like(heights, j)] = data_chunk[idx_i, idx_x, heights]
                heights += 1
                mask = heights < data_chunk.shape[2]
                idx_i = idx_i[mask]
                idx_x = idx_x[mask]
                heights = heights[mask]

        background[self.geometry.zero_traces == 1] = np.nan
        return self.transform_where_present(background, **transform_kwargs)


    @lru_cache(maxsize=1, apply_by_default=False)
    def get_instantaneous_amplitudes(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous amplitude along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values` and :meth:`.transform_where_present``.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_amplitudes(channels=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs) #pylint: disable=unexpected-keyword-arg
        result = np.abs(hilbert(amplitudes))[:, :, depths]
        # result[self.full_matrix == self.FILL_VALUE] = np.nan
        return self.transform_where_present(result, **transform_kwargs)


    @lru_cache(maxsize=1, apply_by_default=False)
    def get_instantaneous_phases(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous phase along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values` and :meth:`.transform_where_present`.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_phases(channels=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs) #pylint: disable=unexpected-keyword-arg
        result = np.angle(hilbert(amplitudes))[:, :, depths]
        # result[self.full_matrix == self.FILL_VALUE] = np.nan
        return self.transform_where_present(result, **transform_kwargs)


    @lru_cache(maxsize=1, apply_by_default=False)
    def get_full_matrix(self, **kwargs):
        """ Transform `matrix` attribute to match cubic coordinates.

        Parameters
        ----------
        kwargs :
            Passed directly to :meth:`.put_on_full` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        matrix = self.put_on_full(self.matrix, **kwargs)
        return self.transform_where_present(matrix, **transform_kwargs)


    @lru_cache(maxsize=1, apply_by_default=False)
    def get_full_binary_matrix(self, **kwargs):
        """ Transform `binary_matrix` attribute to match cubic coordinates.

        Parameters
        ----------
        kwargs :
            Passed directly to :meth:`.put_on_full` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        full_binary_matrix = self.put_on_full(self.binary_matrix, **kwargs)
        return self.transform_where_present(full_binary_matrix, **transform_kwargs)

    def load_attribute(self, src_attribute, location=None, **kwargs):
        """ Make crops from `src_attribute` of horizon at `location`.

        Parameters
        ----------
        src_attribute : str
            A keyword defining horizon attribute to make crops from:
            - 'cube_values' or 'amplitudes': cube values cut along the horizon;
            - 'heights' or 'full_matrix': horizon depth map in cubic coordinates;
            - 'metrics': random support metrics matrix.
            - 'instant_phases': instantaneous phase along the horizon;
            - 'instant_amplitudes': instantaneous amplitude along the horizon;
            - 'masks' or 'full_binary_matrix': mask of horizon;
        location : sequence of 3 slices or None
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is used to infer `window` parameter when `src_attribute` is 'cube_values'.
            If None, `src_attribute` is returned uncropped.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src_attribute`;
            - or attribute-transforming method :meth:`.transform_where_present`.
        Examples
        --------

        >>> horizon.load_attribute('cube_values', (x_slice, i_slice, h_slice), window=10)

        >>> horizon.load_attribute('heights')

        >>> horizon.load_attribute('metrics', metrics='hilbert', normalize='min-max')

        Notes
        -----

        Although the function can be used in a straightforward way as described above, its main purpose is to act
        as an interface for accessing :class:`.Horizon` attributes from :class:`~SeismicCropBatch` to allow calls like:

        >>> Pipeline().load_attribute('cube_values', dst='amplitudes')
        """
        x_slice, i_slice, h_slice = location if location is not None else (slice(None), slice(None), slice(None))

        default_kwargs = {'use_cache': True}
        # Update `kwargs` with additional default values depending on `src_attribute`
        if src_attribute in ['cube_values', 'amplitudes']:
            default_kwargs = {
                'nan_zero_traces': False,
                # `window` arg for `get_cube_values` can be infered from `h_slice`
                **({'window': h_slice.stop - h_slice.start} if h_slice != slice(None) else {}),
                **default_kwargs
            }
        elif src_attribute in ['masks', 'full_binary_matrix']:
            default_kwargs = {
                'fill_value': 0,
                **default_kwargs
            }
        kwargs = {**default_kwargs, **kwargs}
        func_name = self.ATTRIBUTE_TO_METHOD.get(src_attribute)
        if func_name is None:
            raise ValueError("Unknown `src_attribute` {}. Expected {}.".format(src_attribute,
                                                                               self.ATTRIBUTE_TO_METHOD.keys()))
        data = getattr(self, func_name)(**kwargs)
        return data[x_slice, i_slice]


    def get_array_values(self, array, shifts=None, grid_info=None, width=5, axes=(2, 1, 0)):
        """ Get values from an external array along the horizon.

        Parameters
        ----------
        array : np.ndarray
            A data-array to make a cut from.
        shifts : tuple or None
            an offset defining the location of given array with respect to the horizon.
            If None, `grid_info` with key `range` must be supplied.
        grid_info : dict
            Whenever passed, must contain key `range`.
            Used for infering shifts of the array with respect to horizon.
        width : int
            required width of the resulting cut.
        axes : tuple
            if not None, axes-transposition with the required axes-order is used.
        """
        if shifts is None and grid_info is None:
            raise ValueError('Either shifts or dataset with filled grid_info must be supplied!')

        if shifts is None:
            shifts = [grid_info['range'][i][0] for i in range(3)]

        shifts = np.array(shifts)
        horizon_shift = np.array((self.bbox[0, 0], self.bbox[1, 0]))

        if axes is not None:
            array = np.transpose(array, axes=axes)

        # compute start and end-points of the ilines-xlines overlap between
        # array and matrix in horizon and array-coordinates
        horizon_shift, shifts = np.array(horizon_shift), np.array(shifts)
        horizon_max = horizon_shift[:2] + np.array(self.matrix.shape)
        array_max = np.array(array.shape[:2]) + shifts[:2]
        overlap_shape = np.minimum(horizon_max[:2], array_max[:2]) - np.maximum(horizon_shift[:2], shifts[:2])
        overlap_start = np.maximum(0, horizon_shift[:2] - shifts[:2])
        heights_start = np.maximum(shifts[:2] - horizon_shift[:2], 0)

        # recompute horizon-matrix in array-coordinates
        slc_array = [slice(l, h) for l, h in zip(overlap_start, overlap_start + overlap_shape)]
        slc_horizon = [slice(l, h) for l, h in zip(heights_start, heights_start + overlap_shape)]
        overlap_matrix = np.full(array.shape[:2], fill_value=self.FILL_VALUE, dtype=np.float32)
        overlap_matrix[slc_array] = self.matrix[slc_horizon]
        overlap_matrix -= shifts[-1]

        # make the cut-array and fill it with array-data located on needed heights
        result = np.full(array.shape[:2] + (width, ), np.nan, dtype=np.float32)
        iterator = [overlap_matrix + shift for shift in range(-width // 2 + 1, width // 2 + 1)]

        for i, surface_level in enumerate(np.array(iterator)):
            mask = (surface_level >= 0) & (surface_level < array.shape[-1]) & (surface_level !=
                                                                               self.FILL_VALUE - shifts[-1])
            mask_where = np.where(mask)
            result[mask_where[0], mask_where[1], i] = array[mask_where[0], mask_where[1],
                                                            surface_level[mask_where].astype(np.int)]

        return result


    def get_cube_values_line(self, orientation='ilines', line=1, window=23, offset=0, normalize=False):
        """ Get values from the cube along the horizon on a particular line.

        Parameters
        ----------
        orientation : str
            Whether to cut along ilines ('i') or xlines ('x').
        line : int
            Number of line to cut along.
        window : int
            Width of data to cut.
        offset : int
            Value to add to each entry in matrix.
        normalize : bool, callable
            If True, then values are scaled to [0, 1] range.
            If callable, then it is applied to iline-oriented slices of data from the cube.
        chunk_size : int
            Size of data along height axis processed at a time.
        """
        low = window // 2

        # Make callable scaler
        if callable(normalize):
            pass
        elif normalize is True:
            normalize = self.geometry.scaler
        elif normalize is False:
            normalize = lambda array: array

        # Parameters for different orientation
        if orientation.startswith('i'):
            cube_hdf5 = self.geometry.file_hdf5['cube_i']
            slide_transform = lambda array: array

            hor_line = np.squeeze(self.matrix[line, :])
            background = np.zeros((self.geometry.xlines_len, window))
            idx_offset = self.x_min
            bad_traces = np.squeeze(self.geometry.zero_traces[line, :])

        elif orientation.startswith('x'):
            cube_hdf5 = self.geometry.file_hdf5['cube_x']
            slide_transform = lambda array: array.T

            hor_line = np.squeeze(self.matrix[:, line])
            background = np.zeros((self.geometry.ilines_len, window))
            idx_offset = self.i_min
            bad_traces = np.squeeze(self.geometry.zero_traces[:, line])

        # Check where horizon is
        idx = np.asarray((hor_line != self.FILL_VALUE)).nonzero()[0]
        heights = hor_line[idx]

        # Convert coordinates to cubic system
        idx += idx_offset
        heights -= (low - offset)

        slide = cube_hdf5[line, :, :]
        slide = slide_transform(slide)
        slide = normalize(slide)

        # Subsequently add values from the cube to background and shift horizon 1 unit lower
        for j in range(window):
            test = slide[idx, heights]
            background[idx, np.full_like(idx, j)] = test
            heights += 1

        idx = np.asarray((hor_line == self.FILL_VALUE)).nonzero()[0]
        idx += idx_offset
        bad_traces[idx] = 1

        bad_traces = bad_traces.reshape((1, -1) if orientation.startswith('i') else (-1, 1))
        background = background.reshape((1, -1, window) if orientation.startswith('i') else (-1, 1, window))
        return background, bad_traces


    # Horizon properties
    @property
    def cube_values(self):
        """ Values from the cube along the horizon. """
        cube_values = self.get_cube_values(window=1)
        cube_values[self.full_matrix == self.FILL_VALUE] = np.nan
        return cube_values

    @property
    def amplitudes(self):
        """ Alias for cube values. Depending on what loaded to cube geometries
        might actually not be amplitudes, so use it with caution.
        """
        return self.cube_values

    @property
    def binary_matrix(self):
        """ Matrix with ones at places where horizon is present and zeros everywhere else. """
        return (self.matrix > 0).astype(bool)

    @property
    def borders_matrix(self):
        """ Borders of horizons (borders of holes inside are not included). """
        filled_matrix = self.filled_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(filled_matrix, structure, border_value=0)
        return filled_matrix ^ eroded # binary difference operation

    @property
    def boundaries_matrix(self):
        """ Borders of horizons (borders of holes inside included). """
        binary_matrix = self.binary_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(binary_matrix, structure, border_value=0)
        return binary_matrix ^ eroded # binary difference operation

    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        return len(self) / (np.prod(self.cube_shape[:2]) - np.sum(self.geometry.zero_traces))

    @property
    def filled_matrix(self):
        """ Binary matrix with filled holes. """
        structure = np.ones((3, 3))
        filled_matrix = binary_fill_holes(self.binary_matrix, structure)
        return filled_matrix

    @property
    def full_matrix(self):
        """ Matrix in cubic coordinate system. """
        return self.get_full_matrix()

    @property
    def presence_matrix(self):
        """ Binary matrix in cubic coordinate system. """
        return self.put_on_full(self.binary_matrix, fill_value=False, dtype=bool)

    @property
    def grad_i(self):
        """ Change of heights along iline direction. """
        return self.grad_along_axis(0)

    @property
    def grad_x(self):
        """ Change of heights along xline direction. """
        return self.grad_along_axis(1)

    @property
    def hash(self):
        """ Hash on current data of the horizon. """
        return hash(self.matrix.data.tobytes())

    @property
    def horizon_metrics(self):
        """ Calculate :class:`~HorizonMetrics` on demand. """
        from .metrics import HorizonMetrics
        return HorizonMetrics(self)

    @property
    def instantaneous_phase(self):
        """ Phase along the horizon. """
        return self.horizon_metrics.evaluate('instantaneous_phase')

    @property
    def is_carcass(self):
        """ Check if the horizon is a sparse carcass. """
        return len(self) / self.filled_matrix.sum() < 0.5

    @property
    def carcass_ilines(self):
        """ Labeled inlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        return uniques[counts > 256]

    @property
    def carcass_xlines(self):
        """ Labeled xlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        return uniques[counts > 256]

    @property
    def number_of_holes(self):
        """ Number of holes inside horizon borders. """
        holes_array = self.filled_matrix != self.binary_matrix
        _, num = label(holes_array, connectivity=2, return_num=True, background=0)
        return num

    @property
    def perimeter(self):
        """ Number of points in the borders. """
        return np.sum((self.borders_matrix == 1).astype(np.int32))

    @property
    def solidity(self):
        """ Ratio of area covered by horizon to total area inside borders. """
        return len(self) / np.sum(self.filled_matrix)


    def grad_along_axis(self, axis=0):
        """ Change of heights along specified direction. """
        grad = np.diff(self.matrix, axis=axis, prepend=0)
        grad[np.abs(grad) > 10000] = self.FILL_VALUE
        grad[self.matrix == self.FILL_VALUE] = self.FILL_VALUE
        return grad

    def make_float_matrix(self, kernel=None, kernel_size=7, sigma=2., margin=5):
        """ Smooth the depth matrix to produce floating point numbers. """
        kernel = kernel if kernel is not None else make_gaussian_kernel(kernel_size, sigma)
        float_matrix = _smoothing_function(self.full_matrix, kernel,
                                           fill_value=self.FILL_VALUE,
                                           preserve=True, margin=margin)
        return float_matrix

    # Evaluate horizon on its own / against other(s)
    def evaluate(self, compute_metric=True, supports=50, plot=True, savepath=None, printer=print, **kwargs):
        """ Compute crucial metrics of a horizon.

        Parameters
        ----------
        compute_metrics : bool
            Whether to compute correlation map of a horizon.
        supports, savepath, plot, kwargs
            Passed directly to :meth:`HorizonMetrics.evaluate`.
        printer : callable
            Function to display message with metrics.
        """
        msg = f"""
        Number of labeled points:                         {len(self)}
        Number of points inside borders:                  {np.sum(self.filled_matrix)}
        Perimeter (length of borders):                    {self.perimeter}
        Percentage of labeled non-bad traces:             {self.coverage}
        Percentage of labeled traces inside borders:      {self.solidity}
        Number of holes inside borders:                   {self.number_of_holes}
        """
        printer(dedent(msg))
        if compute_metric:
            return self.horizon_metrics.evaluate('support_corrs', supports=supports, agg='nanmean',
                                                 plot=plot, savepath=savepath, **kwargs)
        return None


    @lru_cache(maxsize=1, apply_by_default=False)
    def evaluate_metric(self, metric='support_corrs', supports=50, agg='nanmean', **kwargs):
        """ Cached metrics calcucaltion with disabled plotting option.

        Parameters
        ----------
        metric, supports, agg :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        kwargs :
            Passed directly to :meth:`.HorizonMetrics.evaluate` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        metrics = self.horizon_metrics.evaluate(metric=metric, supports=supports, agg=agg,
                                                plot=False, savepath=None, **kwargs)
        metrics = np.nan_to_num(metrics)
        return self.transform_where_present(metrics, **transform_kwargs)


    def check_proximity(self, other, offset=0):
        """ Compute a number of stats on location of `self` relative to the `other` Horizons.
        This method can be used as either bound or static method.

        Parameters
        ----------
        self, other : Horizon
            Horizons to compare.
        offset : number
            Value to shift the first horizon down.

        Returns
        -------
        dictionary with following keys:
            - `mean` for average distance
            - `abs_mean` for average of absolute values of point-wise distances
            - `max`, `abs_max`, `std`, `abs_std`
            - `window_rate` for percentage of traces that are in 5ms from one horizon to the other
            - `offset_diffs` with point-wise differences
        """
        _, overlap_info = self.verify_merge(other)
        diffs = overlap_info.get('diffs', 999) + offset

        overlap_info = {
            **overlap_info,
            'mean': np.mean(diffs),
            'abs_mean': np.mean(np.abs(diffs)),
            'max': np.max(diffs),
            'abs_max': np.max(np.abs(diffs)),
            'std': np.std(diffs),
            'abs_std': np.std(np.abs(diffs)),
            'window_rate': np.mean(np.abs(diffs) < (5 / self.geometry.sample_rate)),
            'offset_diffs': diffs,
        }
        return overlap_info


    # Merge functions
    def verify_merge(self, other, mean_threshold=3.0, adjacency=0):
        """ Collect stats of overlapping of two horizons.

        Returns a number that encodes position of two horizons, as well as dictionary with collected statistics.
        If code is 0, then horizons are too far away from each other (heights-wise), and therefore are not mergeable.
        If code is 1, then horizons are too far away from each other (spatially) even with adjacency, and therefore
        are not mergeable.
        If code is 2, then horizons are close enough spatially (with adjacency), but are not overlapping, and therefore
        an additional check (`adjacent_merge`) is needed.
        If code is 3, then horizons are definitely overlapping and are close enough to meet all the thresholds, and
        therefore are mergeable without any additional checks.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to compare.
        mean_threshold : number
            Height threshold for mean distances.
        adjacency : int
            Margin to consider horizons to be close (spatially).
        """
        overlap_info = {}

        # Overlap bbox
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        i_range = overlap_i_min - overlap_i_max
        x_range = overlap_x_min - overlap_x_max

        # Simplest possible check: horizon bboxes are too far from each other
        if i_range >= adjacency or x_range >= adjacency:
            merge_code = 1
            spatial_position = 'distant'
        else:
            merge_code = 2
            spatial_position = 'adjacent'

        # Compare matrices on overlap without adjacency:
        if merge_code != 1 and i_range < 0 and x_range < 0:
            self_overlap = self.matrix[overlap_i_min - self.i_min:overlap_i_max - self.i_min,
                                       overlap_x_min - self.x_min:overlap_x_max - self.x_min]

            other_overlap = other.matrix[overlap_i_min - other.i_min:overlap_i_max - other.i_min,
                                         overlap_x_min - other.x_min:overlap_x_max - other.x_min]

            self_mask = self_overlap != self.FILL_VALUE
            other_mask = other_overlap != self.FILL_VALUE
            mask = self_mask & other_mask
            diffs_on_overlap = self_overlap[mask] - other_overlap[mask]

            if len(diffs_on_overlap) == 0:
                # bboxes are overlapping, but horizons are not
                merge_code = 2
                spatial_position = 'adjacent'
            else:
                abs_diffs = np.abs(diffs_on_overlap)
                mean_on_overlap = np.mean(abs_diffs)
                if mean_on_overlap < mean_threshold:
                    merge_code = 3
                    spatial_position = 'overlap'
                else:
                    merge_code = 0
                    spatial_position = 'separated'

                overlap_info.update({'mean': mean_on_overlap,
                                     'diffs': diffs_on_overlap})

        overlap_info['spatial_position'] = spatial_position
        return merge_code, overlap_info


    def overlap_merge(self, other, inplace=False):
        """ Merge two horizons into one.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.
        """
        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Add both horizons to the background
        background[shared_self_i_min:shared_self_i_min+self.i_length,
                   shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Correct overlapping points
        overlap_i_min, overlap_i_max = max(self.i_min, other.i_min), min(self.i_max, other.i_max) + 1
        overlap_x_min, overlap_x_max = max(self.x_min, other.x_min), min(self.x_max, other.x_max) + 1

        overlap_i_min -= shared_i_min
        overlap_i_max -= shared_i_min
        overlap_x_min -= shared_x_min
        overlap_x_max -= shared_x_min

        overlap = background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max]
        mask = overlap >= 0
        overlap[mask] //= 2
        overlap[~mask] -= self.FILL_VALUE
        background[overlap_i_min:overlap_i_max, overlap_x_min:overlap_x_max] = overlap

        background[background == 0] = self.FILL_VALUE
        length = len(self) + len(other) - mask.sum()
        # Create new instance or change `self`
        if inplace:
            # Change `self` inplace
            self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
            merged = True
        else:
            # Return a new instance of horizon
            merged = Horizon(background, self.geometry, self.name,
                             i_min=shared_i_min, x_min=shared_x_min, length=length)
        return merged


    def adjacent_merge(self, other, mean_threshold=3.0, adjacency=3, inplace=False):
        """ Check if adjacent merge (that is merge with some margin) is possible, and, if needed, merge horizons.
        Note that this function can either merge horizons in-place of the first one (`self`), or create a new instance.

        Parameters
        ----------
        self, other : :class:`.Horizon` instances
            Horizons to merge.
        mean_threshold : number
            Height threshold for mean distances.
        adjacency : int
            Margin to consider horizons close (spatially).
        inplace : bool
            Whether to create new instance or update `self`.
        """
        # Simplest possible check: horizons are too far away from one another (depth-wise)
        overlap_h_min, overlap_h_max = max(self.h_min, other.h_min), min(self.h_max, other.h_max)
        if overlap_h_max - overlap_h_min < 0:
            return False

        # Create shared background for both horizons
        shared_i_min, shared_i_max = min(self.i_min, other.i_min), max(self.i_max, other.i_max)
        shared_x_min, shared_x_max = min(self.x_min, other.x_min), max(self.x_max, other.x_max)

        background = np.zeros((shared_i_max - shared_i_min + 1, shared_x_max - shared_x_min + 1),
                              dtype=np.int32)

        # Coordinates inside shared for `self` and `other`
        shared_self_i_min, shared_self_x_min = self.i_min - shared_i_min, self.x_min - shared_x_min
        shared_other_i_min, shared_other_x_min = other.i_min - shared_i_min, other.x_min - shared_x_min

        # Put the second of the horizons on background
        background[shared_other_i_min:shared_other_i_min+other.i_length,
                   shared_other_x_min:shared_other_x_min+other.x_length] += other.matrix

        # Enlarge the image to account for adjacency
        kernel = np.ones((3, 3), np.float32)
        dilated_background = cv2.dilate(background.astype(np.float32), kernel,
                                        iterations=adjacency).astype(np.int32)

        # Make counts: number of horizons in each point; create indices of overlap
        counts = (dilated_background != 0).astype(np.int32)
        counts[shared_self_i_min:shared_self_i_min+self.i_length,
               shared_self_x_min:shared_self_x_min+self.x_length] += 1
        counts_idx = counts == 2

        # Determine whether horizon can be merged (adjacent and height-close) or not
        mergeable = False
        if counts_idx.any():
            # Put the first horizon on dilated background, compute mean
            background[shared_self_i_min:shared_self_i_min+self.i_length,
                       shared_self_x_min:shared_self_x_min+self.x_length] += self.matrix

            # Compute diffs on overlap
            diffs = background[counts_idx] - dilated_background[counts_idx]
            diffs = np.abs(diffs)
            diffs = diffs[diffs < (-self.FILL_VALUE // 2)]

            if len(diffs) != 0 and np.mean(diffs) < mean_threshold:
                mergeable = True

        if mergeable:
            background[(background < 0) & (background != self.FILL_VALUE)] -= self.FILL_VALUE
            background[background == 0] = self.FILL_VALUE

            length = len(self) + len(other) # since there is no direct overlap

            # Create new instance or change `self`
            if inplace:
                # Change `self` inplace
                self.from_matrix(background, i_min=shared_i_min, x_min=shared_x_min, length=length)
                merged = True
            else:
                # Return a new instance of horizon
                merged = Horizon(background, self.geometry, self.name,
                                 i_min=shared_i_min, x_min=shared_x_min, length=length)
            return merged
        return False


    @staticmethod
    def merge_list(horizons, mean_threshold=2.0, adjacency=3, minsize=50):
        """ Iteratively try to merge every horizon in a list to every other, until there are no possible merges.
        Parameters are passed directly to :meth:`~.verify_merge`, :meth:`~.overlap_merge` and :meth:`~.adjacent_merge`.
        """
        horizons = [horizon for horizon in horizons if len(horizon) >= minsize]

        # Iterate over the list of horizons to merge everything that can be merged
        i = 0
        flag = True
        while flag:
            # Continue while at least one pair of horizons was merged at previous iteration
            flag = False
            while True:
                if i >= len(horizons):
                    break

                j = i + 1
                while True:
                    # Attempt to merge j-th horizon to i-th horizon
                    if j >= len(horizons):
                        break

                    merge_code, _ = Horizon.verify_merge(horizons[i], horizons[j],
                                                         mean_threshold=mean_threshold,
                                                         adjacency=adjacency)
                    if merge_code == 3:
                        merged = Horizon.overlap_merge(horizons[i], horizons[j], inplace=True)
                    elif merge_code == 2:
                        merged = Horizon.adjacent_merge(horizons[i], horizons[j], inplace=True,
                                                        mean_threshold=mean_threshold,
                                                        adjacency=adjacency)
                    else:
                        merged = False

                    if merged:
                        _ = horizons.pop(j)
                        flag = True
                    else:
                        j += 1
                i += 1
        return horizons


    @staticmethod
    def average_horizons(horizons):
        """ Average list of horizons into one surface. """
        geometry = horizons[0].geometry
        horizon_matrix = np.zeros(geometry.lens, dtype=np.float32)
        std_matrix = np.zeros(geometry.lens, dtype=np.float32)
        counts_matrix = np.zeros(geometry.lens, dtype=np.int32)

        for horizon in horizons:
            fm = horizon.full_matrix
            horizon_matrix[fm != Horizon.FILL_VALUE] += fm[fm != Horizon.FILL_VALUE]
            std_matrix[fm != Horizon.FILL_VALUE] += fm[fm != Horizon.FILL_VALUE] ** 2
            counts_matrix[fm != Horizon.FILL_VALUE] += 1

        horizon_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        horizon_matrix[counts_matrix == 0] = Horizon.FILL_VALUE

        std_matrix[counts_matrix != 0] /= counts_matrix[counts_matrix != 0]
        std_matrix -= horizon_matrix ** 2
        std_matrix = np.sqrt(std_matrix)
        std_matrix[counts_matrix == 0] = np.nan

        averaged_horizon = Horizon(horizon_matrix.astype(np.int32), geometry=geometry)
        return averaged_horizon, {
            'matrix': horizon_matrix,
            'std_matrix': std_matrix,
        }


    # Save horizon to disk
    @staticmethod
    def dump_charisma(points, path, transform=None, add_height=False):
        """ Save (N, 3) array of points to disk in CHARISMA-compatible format.

        Parameters
        ----------
        points : ndarray
            Array of (N, 3) shape.
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        points = points if transform is None else transform(points)
        path = path if not add_height else f'{path}_#{round(np.mean(points[:, 2]), 1)}'

        df = pd.DataFrame(points, columns=Horizon.COLUMNS)
        df.sort_values(['iline', 'xline'], inplace=True)
        df = df.astype({'iline': np.int32, 'xline': np.int32, 'height': np.float32})
        df.to_csv(path, sep=' ', columns=Horizon.COLUMNS, index=False, header=False)

    def dump_matrix(self, matrix, path, transform=None, add_height=False):
        """ Save (N_ILINES, N_CROSSLINES) matrix in CHARISMA-compatible format.

        Parameters
        ----------
        matrix : ndarray
            Array of (N_ILINES, N_CROSSLINES) shape with depth values.
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        points = Horizon.matrix_to_points(matrix)
        points = self.cubic_to_lines(points)
        Horizon.dump_charisma(points, path, transform, add_height)

    def dump(self, path, transform=None, add_height=False):
        """ Save horizon points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        points = self.cubic_to_lines(copy(self.points))
        self.dump_charisma(points, path, transform, add_height)

    def dump_float(self, path, transform=None, kernel_size=7, sigma=2., margin=5, add_height=False):
        """ Smooth out the horizon values, producing floating-point numbers, and dump to the disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        kernel_size : int
            Size of the filtering kernel.
        sigma : number
            Standard deviation of the Gaussian kernel.
        margin : number
            During the filtering, not include in the computation all the points that are
            further away from the current, than the margin.
        add_height : bool
            Whether to concatenate average horizon height to a file name.
        """
        matrix = self.make_float_matrix(kernel_size=kernel_size, sigma=sigma, margin=margin)
        points = self.matrix_to_points(matrix)
        points = self.cubic_to_lines(points)
        self.dump_charisma(points, path, transform, add_height)

    def dump_points(self, path, fmt='npy', projections='ixh'):
        """ Dump points. """
        cube_keys = self.geometry.CUBE_PROJECTIONS
        axes = self.geometry.PROJECTION_AXES

        if fmt == 'npy':
            if os.path.exists(path):
                points = np.load(path, allow_pickle=False)
                points = np.concatenate([points, self.points], axis=0)
            else:
                points = self.points
            np.save(path, points, allow_pickle=False)
        elif fmt == 'hdf5':
            if os.path.exists(path):
                os.remove(path)
            file_hdf5 = h5py.File(path, "a")
            cube_hdf5 = dict()
            for projection in projections:
                name = cube_keys[projection]
                if name not in file_hdf5:
                    cube_hdf5[name] = file_hdf5.create_dataset(name, self.cube_shape[axes[projection]])
                else:
                    cube_hdf5[name] = file_hdf5[name]

            shape = (self.i_length, self.x_length, self.h_max - self.h_min + 1)
            fault_array = np.zeros(shape)

            points = self.points - np.array([self.i_min, self.x_min, self._h_min])
            fault_array[points[:, 0], points[:, 1], points[:, 2]] = 1

            slices = (slice(self.i_min, self.i_max+1), slice(self.x_min,self.x_max+1), slice(self.h_min, self.h_max+1))

            for projection in projections:
                name = cube_keys[projection]
                ax = axes[projection]
                cube_hdf5[name][slices[ax[0]], slices[ax[1]], slices[ax[2]]] += np.transpose(fault_array, ax)

            file_hdf5.close()
            path_meta = os.path.splitext(path)[0] + '.meta'
            self.geometry.store_meta(path_meta)
        else:
            raise ValueError('Unknown format:', fmt)


    # Methods of (visual) representation of a horizon
    def __repr__(self):
        return f"""<horizon {self.name} for {self.cube_name} at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.cube_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Depth range:       {self.h_min} to {self.h_max}
        Depth mean:        {self.h_mean:.6}
        Depth std:         {self.h_std:.6}

        Length:            {len(self)}
        Perimeter:         {self.perimeter}
        Coverage:          {self.coverage:3.5}
        Solidity:          {self.solidity:3.5}
        Num of holes:      {self.number_of_holes}
        """

        if self.is_carcass:
            msg += f"""
        Unique ilines:     {self.carcass_ilines}
        Unique xlines:     {self.carcass_xlines}
        """
        return dedent(msg)


    def put_on_full(self, matrix=None, fill_value=None, dtype=np.float32):
        """ Create a matrix in cubic coordinate system. """
        matrix = matrix if matrix is not None else self.matrix
        fill_value = fill_value if fill_value is not None else self.FILL_VALUE

        background = np.full(self.cube_shape[:-1], fill_value, dtype=dtype)
        background[self.i_min:self.i_max+1, self.x_min:self.x_max+1] = matrix
        return background


    def show(self, src='matrix', fill_value=None, on_full=True, **kwargs):
        """ Nice visualization of a horizon-related matrix. """
        matrix = getattr(self, src) if isinstance(src, str) else src
        fill_value = fill_value if fill_value is not None else self.FILL_VALUE

        if on_full:
            matrix = self.put_on_full(matrix=matrix, fill_value=fill_value)
        else:
            matrix = copy(matrix).astype(np.float32)

        # defaults for plotting if not supplied in kwargs
        kwargs = {
            'cmap': 'viridis_r',
            'title': '{} {} of `{}` on `{}`'.format(src if isinstance(src, str) else '',
                                                    'on full'*on_full, self.name, self.cube_name),
            'xlabel': self.geometry.index_headers[0],
            'ylabel': self.geometry.index_headers[1],
            **kwargs
            }
        matrix[matrix == fill_value] = np.nan
        plot_image(matrix, mode='single', **kwargs)


    def show_amplitudes_rgb(self, width=3, channel_weights=(1, 0.5, 0.25), to_uint8=True,
                            channels=None, **kwargs):
        """ Show trace values on the horizon and surfaces directly under it.

        Parameters
        ----------
        width : int
            Space between surfaces to cut.
        channel_weights : tuple
            Weights applied to rgb-channels.
        to_uint8 : bool
            Determines whether the image should be cast to uint8.
        channels : tuple
            Tuple of 3 ints. Determines channels to take from amplitudes to form rgb-image.
        backend : str
            Can be either 'matplotlib' ('plt') or 'plotly' ('go')
        """
        channels = (0, width, -1) if channels is None else channels

        # get values along the horizon and cast them to [0, 1]
        amplitudes = self.get_cube_values(window=1 + width*2, offset=width)
        amplitudes = amplitudes[:, :, channels]
        amplitudes -= np.nanmin(amplitudes, axis=(0, 1)).reshape(1, 1, -1)
        amplitudes *= 1 / np.nanmax(amplitudes, axis=(0, 1)).reshape(1, 1, -1)
        amplitudes[self.full_matrix == self.FILL_VALUE, :] = np.nan
        amplitudes = amplitudes[:, :, ::-1]
        amplitudes *= np.asarray(channel_weights).reshape(1, 1, -1)

        # cast values to uint8 if needed
        if to_uint8:
            amplitudes = (amplitudes * 255).astype(np.uint8)

        # defaults for plotting if not supplied in kwargs
        kwargs = {
            'title': 'RGB amplitudes of {} on cube {}'.format(self.name, self.cube_name),
            'xlabel': self.geometry.index_headers[0],
            'ylabel': self.geometry.index_headers[1],
            **kwargs
            }

        plot_image(amplitudes, mode='rgb', **kwargs)


    def show_3d(self, n_points=100, threshold=100., z_ratio=1., zoom_slice=None, show_axes=True,
                width=1200, height=1200, margin=(0, 0, 100), savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger height differences in vertices.
        z_ratio : number
            Aspect ratio between height axis and spatial ones.
        zoom_slice : tuple of slices
            Crop from cube to show.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : number
            Size of the image.
        margin : number
            Added margin from below and above along height axis.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Horizon `{self.name}` on `{self.cube_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.geometry.index_headers[0], self.geometry.index_headers[1], 'DEPTH')
        if zoom_slice is None:
            zoom_slice = [slice(0, i) for i in self.geometry.cube_shape]
        zoom_slice[-1] = slice(self.h_min, self.h_max)

        x, y, z, simplices = self.triangulation(n_points, threshold, zoom_slice)

        show_3d(x, y, z, simplices, title, zoom_slice, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)


    def triangulation(self, n_points, threshold, slices, **kwargs):
        """ Create triangultaion of horizon.

        Parameters
        ----------
        n_points: int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        slices : tuple
            Region to process.

        Returns
        -------
        x, y, z, simplices
            `x`, `y` and `z` are np.ndarrays of triangle vertices, `simplices` is (N, 3) array where each row
            represent triangle. Elements of row are indices of points that are vertices of triangle.
        """
        _ = kwargs
        weights_matrix = self.full_matrix

        grad_i = np.diff(weights_matrix, axis=0, prepend=0)
        grad_x = np.diff(weights_matrix, axis=1, prepend=0)
        weights_matrix = (grad_i + grad_x) / 2
        weights_matrix[np.abs(weights_matrix) > 100] = np.nan

        idx = np.stack(np.nonzero(self.full_matrix > 0), axis=0)
        mask_1 = (idx <= np.array([slices[0].stop, slices[1].stop]).reshape(2, 1)).all(axis=0)
        mask_2 = (idx >= np.array([slices[0].start, slices[1].start]).reshape(2, 1)).all(axis=0)
        mask = np.logical_and(mask_1, mask_2)
        idx = idx[:, mask]

        probs = np.abs(weights_matrix[idx[0], idx[1]].flatten())
        probs[np.isnan(probs)] = np.nanmax(probs)
        indices = np.random.choice(len(probs), size=n_points, p=probs / probs.sum())

        # Convert to meshgrid
        ilines = self.points[mask, 0][indices]
        xlines = self.points[mask, 1][indices]
        ilines, xlines = np.meshgrid(ilines, xlines)
        ilines = ilines.flatten()
        xlines = xlines.flatten()

        # Remove from grid points with no horizon in it
        heights = self.full_matrix[ilines, xlines]
        mask = (heights != self.FILL_VALUE)
        x = ilines[mask]
        y = xlines[mask]
        z = heights[mask]

        # Triangulate points and remove some of the triangles
        tri = Delaunay(np.vstack([x, y]).T)
        simplices = filter_simplices(simplices=tri.simplices, points=tri.points,
                                     matrix=self.full_matrix, threshold=threshold)
        return x, y, z, simplices

    def show_slide(self, loc, width=3, axis='i', order_axes=None, zoom_slice=None,
                   n_ticks=5, delta_ticks=100, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        width : int
            Horizon thickness.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])

        # Load seismic and mask
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis)
        mask = np.zeros(shape)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xticks = list(range(seismic_slide.shape[0]))
        yticks = list(range(seismic_slide.shape[1]))

        if zoom_slice:
            seismic_slide = seismic_slide[zoom_slice]
            mask = mask[zoom_slice]
            xticks = xticks[zoom_slice[0]]
            yticks = yticks[zoom_slice[1]]

        # defaults for plotting if not supplied in kwargs
        header = self.geometry.axis_names[axis]
        total = self.geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = self.geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.geometry.index_headers[0]
            ylabel = self.geometry.index_headers[1]
            total = self.geometry.depth

        xticks = xticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [xticks[-1]]
        xticks = sorted(list(set(xticks)))
        yticks = yticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [yticks[-1]]
        yticks = sorted(list(set(yticks)), reverse=True)

        if len(xticks) > 2 and (xticks[-1] - xticks[-2]) < delta_ticks:
            xticks.pop(-2)
        if len(yticks) > 2 and (yticks[0] - yticks[1]) < delta_ticks:
            yticks.pop(1)

        kwargs = {
            'mode': 'overlap',
            'title': f'Horizon `{self.name}` on `{self.geometry.name}`\n {header} {loc} out of {total}',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': xticks,
            'yticks': yticks,
            'y': 1.02,
            **kwargs
        }

        plot_image([seismic_slide, mask], order_axes=order_axes, **kwargs)


    def reset_cache(self):
        """ Clear cached data. """
        for attr in self._cached_attributes:
            getattr(self, attr).reset_instance(self)


    def __copy__(self):
        """ Create a shallow copy of a horizon.

        Returns
        -------
        A horizon object with new matrix object and a reference to the old geometry attribute.
        """
        return Horizon(np.copy(self.matrix), self.geometry, i_min=self.i_min, x_min=self.x_min,
                       name=f'copy_of_{self.name}')


class StructuredHorizon(Horizon):
    """ Convenient alias for :class:`.Horizon` class. """


@njit(parallel=True)
def _smoothing_function(src, kernel, fill_value, preserve=False, margin=33):
    #pylint: disable=not-an-iterable
    k1 = int(np.floor(kernel.shape[0] / 2))
    k2 = int(np.floor(kernel.shape[1] / 2))
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    dst = np.copy(src)
    for iline in range(k1, src.shape[0]-k1):
        for xline in prange(k2, src.shape[1]-k2):
            central = src[iline, xline]
            if preserve and central == fill_value:
                continue

            element = src[iline-k1:iline+k1+1, xline-k2:xline+k2+1]

            s, sum_weights = 0.0, 0.0
            for item, weight in zip(element.ravel(), raveled_kernel):
                if item != fill_value:
                    if abs(item - central) <= margin:
                        s += item * weight
                        sum_weights += weight

            if sum_weights != 0.0:
                val = s / sum_weights
                dst[iline, xline] = val
    return dst

@njit
def _filtering_function(points, filtering_matrix):
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(points), dtype=np.int32)

    for i in range(len(points)):
        il, xl = points[i, 0], points[i, 1]
        if filtering_matrix[il, xl] == 1:
            mask[i] = 0
    return points[mask == 1, :]
