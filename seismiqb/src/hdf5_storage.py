""" Class to store seismic cubes in HDF5 in multiple orientations. """
import os

import numpy as np
import h5py

from .utils import SafeIO, lru_cache, parse_axis, make_axis_grid

class StorageHDF5:
    """ Class for storing 3D data in hdf5 format. To speed up loading, the file can store multiple transposed
    copies of the same cube.

    Parameters
    ----------
    filename : str
        Path to file.
    projections : str or None, optional
        String of 'i', 'h' and 'x', by default None. Is needed to create new hdf5 file.
    shape : tuple, optional
        Shape of 3D array in original orientation (iline, xline, depth), by default None.
        Is needed to create new hdf5 file.
    mode : 'r', 'r+' or 'a', optional
        Mode to open hdf5 file, by default 'r'
    """
    STRAIGHT = {
        0: [0, 1, 2], 1: [1, 2, 0], 2: [2, 0, 1],
        'i': [0, 1, 2], 'x': [1, 2, 0], 'h': [2, 0, 1]
    }
    TRANSPOSE = {
        0: [0, 1, 2], 1: [2, 0, 1], 2: [1, 2, 0],
        'i': [0, 1, 2], 'x': [2, 0, 1], 'h': [1, 2, 0]
    }
    NAMES = {
        0: 'cube_i', 1: 'cube_x', 2: 'cube_h',
        'i': 'cube_i', 'x': 'cube_x', 'h': 'cube_h'
    }

    def __init__(self, filename, projections=None, shape=None, mode='r'):
        self.filename = filename
        if mode in ('r', 'r+'):
            self.file_hdf5 = SafeIO(filename, opener=h5py.File, mode=mode)
            projections = projections or [axis for axis in range(3) if self.NAMES[axis] in self.file_hdf5]
            self.projections = [parse_axis(item) for item in projections]
            axis = self.projections[0]
            self.shape = np.array(self.cube_orientation(axis).shape)[self.TRANSPOSE[axis]]
        elif mode == 'a':
            if os.path.exists(filename):
                os.remove(filename)
            self.file_hdf5 = h5py.File(filename, 'a')
            self.projections = projections or [0, 1, 2]
            self.projections = [parse_axis(item) for item in self.projections]
            self.shape = shape
            for p in self.projections:
                _shape = np.array(shape)[self.STRAIGHT[p]]
                self.file_hdf5.create_dataset(self.NAMES[p], _shape)

        for axis in self.projections:
            cube_name = self.NAMES[axis]
            setattr(self, cube_name, self.file_hdf5[cube_name])


    def cube_orientation(self, projection):
        """ Load the cube in the desired orientation.

        Parameters
        ----------
        projection : str ot int
            'i', 'x', 'h' or corresponding numerical index.

        Returns
        -------
        h5py._hl.dataset.Dataset
        """
        return self.file_hdf5[self.NAMES[projection]]

    def get_optimal_projection(self, locations):
        """ Choose optimal cube orientation from the existing ones to speed up crop loading.

        Parameters
        ----------
        locations : tuple of slices
            Crop position.

        Returns
        -------
        axis : int
            Optimal axis to load crop.

        Example
        -------
        self.projections = [0, 1],    locations = (slice(10), slice(100), slice(5)) -> axis = 0
        self.projections = [0, 1],    locations = (slice(100), slice(10), slice(5)) -> axis = 1
        self.projections = [0, 1, 2], locations = (slice(100), slice(10), slice(5)) -> axis = 2
        """
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)])
        indices = np.argsort(shape)
        axis = 0
        for axis in indices:
            if axis in self.projections:
                break
        return axis

    def __getitem__(self, key):
        """ Use native slicing of HDF5. """
        key, squeeze = self._process_key(key)
        projection = self.get_optimal_projection(key)
        slices = tuple([key[axis] for axis in self.STRAIGHT[projection]])

        transposed_crop = self.cube_orientation(projection)[slices]
        crop = transposed_crop.transpose(self.TRANSPOSE[projection])
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    def _process_key(self, key):
        """ Process slices for cube to put into __getitem__ and __setitem__. """
        key_ = [key] if isinstance(key, slice) else list(key)
        key, squeeze = [], []
        if len(key_) != len(self.shape):
            key_ += [slice(None)] * (len(self.shape) - len(key_))
        for i, item in enumerate(key_):
            max_size = self.shape[i]

            if isinstance(item, slice):
                slc = slice(item.start or 0, item.stop or max_size)
            elif isinstance(item, int):
                item = item if item >= 0 else max_size - item
                slc = slice(item, item + 1)
                squeeze.append(i)
            else:
                raise ValueError(f'key elements must be slices or int but {type(item)} was given.')
            key.append(slc)
        return key, squeeze

    def __setitem__(self, key, value):
        key, _ = self._process_key(key)
        for projection in self.projections:
            slices = tuple(np.array(key)[self.STRAIGHT[projection]])
            self.cube_orientation(projection)[slices] = value.transpose(self.STRAIGHT[projection])

    def load_slide(self, loc, axis=0, **kwargs):
        """ Load 2D slide from seismic cube in the fastest way by choosing the most appropriate orientation.

        Parameters
        ----------
        loc : int
            Slide position.
        axis : int, optional
            Corresponding axis, by default 0

        Returns
        -------
        np.ndarray
        """
        axis = parse_axis(axis)
        if axis in self.projections:
            cube_hdf5 = self.cube_orientation(axis)
            slide = self.load_existed_slide(cube_hdf5, loc)
            if axis == 1:
                slide = slide.T
        else:
            slide = self.construct_slide(loc, axis)
        return slide

    @lru_cache(128)
    def load_existed_slide(self, cube, loc, axis=0, **kwargs):
        """ Load one slide of data from a certain cube projection.
        Caches the result in a thread-safe manner.
        """
        _ = kwargs
        slc = [slice(None), slice(None), slice(None)]
        slc[parse_axis(axis)] = loc
        return cube[slc[0], slc[1], slc[2]]

    @lru_cache(128)
    def construct_slide(self, loc, axis=0, **kwargs):
        """ Load one slide from file without corresponding axis.
        Caches the result in a thread-safe manner.
        """
        locations = [slice(None) for _ in range(3)]
        locations[axis] = slice(loc, loc+1)
        slc = [slice(None) for _ in range(3)]
        slc[axis] = 0
        slide = self.load_crop(locations, **kwargs)[tuple(slc)]
        return slide

    def load_crop(self, locations, projection=None, **kwargs):
        """ Load crop from seismic cube in the fastest way by choosing the most appropriate orientation.

        Parameters
        ----------
        locations : tuple of slices
            Slice of cube to load.
        projection : int or None, optional
            Cube projection to load crop from, by default None. None means that projection will be selected
            automatically in the most optimal way.
        """
        projection = projection or self.get_optimal_projection(locations)
        if projection == 0:
            crop = self._load_i(*locations, **kwargs)
        elif projection == 1:
            crop = self._load_x(*locations, **kwargs)
        elif projection == 2:
            crop = self._load_h(*locations, **kwargs)
        else:
            raise ValueError('Wrong projection value:', projection)
        return crop

    def _load_i(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.cube_orientation(0)
        start, stop = ilines.start or 0, ilines.stop or cube_hdf5.shape[0]
        shape_x = (xlines.stop or cube_hdf5.shape[1]) - (xlines.start or 0)
        shape_h = (heights.stop or cube_hdf5.shape[2]) - (heights.start or 0)

        crop = np.empty((stop - start, shape_x, shape_h))
        for i, iline in enumerate(range(start, stop)):
            crop[i] = self.load_existed_slide(cube_hdf5, iline, **kwargs)[xlines, :][:, heights]
        return crop

    def _load_x(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.cube_orientation(1)
        start, stop = xlines.start or 0, xlines.stop or cube_hdf5.shape[0]
        shape_h = (heights.stop or cube_hdf5.shape[1]) - (heights.start or 0)
        shape_i = (ilines.stop or cube_hdf5.shape[2]) - (ilines.start or 0)

        crop = np.empty((stop - start, shape_h, shape_i))
        for i, xline in enumerate(range(start, stop)):
            crop[i] = self.load_existed_slide(cube_hdf5, xline, **kwargs)[heights, :][:, ilines]
        return crop.transpose(2, 0, 1)

    def _load_h(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.cube_orientation(2)
        start, stop = heights.start or 0, heights.stop or cube_hdf5.shape[0]
        shape_i = (ilines.stop or cube_hdf5.shape[1]) - (ilines.start or 0)
        shape_x = (xlines.stop or cube_hdf5.shape[2]) - (xlines.start or 0)

        crop = np.empty((stop - start, shape_i, shape_x))
        for i, height in enumerate(range(start, stop)):
            crop[i] = self.load_existed_slide(cube_hdf5, height, **kwargs)[ilines, :][:, xlines]
        return crop.transpose(1, 2, 0)

    def add_projection(self, projections, stride=100):
        """ Add additional cube orientations. To avoid load of the whole cube into memory it can be loaded
        by chunks.

        Parameters
        ----------
        projections : tuple of int or str
            Projections to add.
        stride : int, optional
            Stride for chunks, by default 100

        Notes
        -----
        New projections will be created from the first cube. Order of cubes is defined by self.projections.
        Chunks will be created along the first axis of that cube, transposed and assigned to corresponding
        region of new cube.
        """
        projections = [parse_axis(item) for item in projections]
        src_axis = self.projections[0]
        src_cube = self.cube_orientation(src_axis)
        shape = np.array(src_cube.shape)[self.TRANSPOSE[src_axis]]

        projections = [item for item in projections if item not in self.projections]
        self.projections += projections

        for i in range(0, src_cube.shape[0], stride):
            slide = src_cube[i:i+stride]
            slide = slide.transpose(self.TRANSPOSE[src_axis])
            main_axis = self.STRAIGHT[src_axis][0]
            for axis in projections:
                _shape = np.array(shape)[self.STRAIGHT[axis]]
                if self.NAMES[axis] not in self.file_hdf5:
                    self.file_hdf5.create_dataset(self.NAMES[axis], _shape)
                slices = [slice(None) for i in range(3)]
                slices[self.TRANSPOSE[axis][main_axis]] = slice(i, i+stride)
                self.cube_orientation(axis)[tuple(slices)] = slide.transpose(self.STRAIGHT[axis])

    @classmethod
    def create_file_from_iterable(cls, src, shape, window, stride, dst=None,
                                  agg=None, projection='ixh', threshold=None):
        """ Aggregate multiple chunks into file with 3D cube.

        Parameters
        ----------
        src : iterable
            Each item is a tuple (position, array) where position is a 3D coordinate of the left upper array corner.
        shape : tuple
            Shape of the resulting array.
        window : tuple
            Chunk shape.
        stride : tuple
            Stride for chunks. Values in overlapped regions will be aggregated.
        dst : str or None, optional
            Path to the resulting .hdf5. If None, function will return array with predictions
        agg : 'mean', 'min' or 'max' or None, optional
            The way to aggregate values in overlapped regions. None means that new chunk will rewrite
            previous value in cube.
        projection : str, optional
            Projections to create in hdf5 file, by default 'ixh'
        threshold : float or None, optional
            If not None, threshold to transform values into [0, 1], by default None
        """
        shape = np.array(shape)
        window = np.array(window)
        stride = np.array(stride)

        if dst is None:
            dst = np.zeros(shape)
        else:
            dst = StorageHDF5(dst, projection[0], shape=shape, mode='a')

        lower_bounds = [make_axis_grid((0, shape[i]), stride[i], shape[i], window[i]) for i in range(3)]
        lower_bounds = np.stack(np.meshgrid(*lower_bounds), axis=-1).reshape(-1, 3)
        upper_bounds = lower_bounds + window
        grid = np.stack([lower_bounds, upper_bounds], axis=-1)

        for position, chunk in src:
            slices = [slice(position[i], position[i]+chunk.shape[i]) for i in range(3)]
            _chunk = dst[slices]
            if agg in ('max', 'min'):
                chunk = np.maximum(chunk, _chunk) if agg == 'max' else np.minimum(chunk, _chunk)
            elif agg == 'mean':
                grid_mask = np.logical_and(
                    grid[..., 1] >= np.expand_dims(position, axis=0),
                    grid[..., 0] < np.expand_dims(position + window, axis=0)
                ).all(axis=1)
                agg_map = np.zeros_like(chunk)
                for chunk_slc in grid[grid_mask]:
                    slices = [slice(
                        max(chunk_slc[i, 0], position[i]) - position[i],
                        min(chunk_slc[i, 1], position[i] + window[i]) - position[i]
                    ) for i in range(3)]
                    agg_map[tuple(slices)] += 1
                chunk /= agg_map
                chunk = _chunk + chunk
            dst[slices] = chunk
        if isinstance(dst, np.ndarray):
            if threshold is not None:
                dst = (dst > threshold).astype(int)
        else:
            for i in range(0, dst.shape[0], window[0]):
                slide = dst[i:i+window[0]]
                if threshold is not None:
                    slide = (slide > threshold).astype(int)
                    dst[i:i+window[0]] = slide
            dst.add_projection(projection[1:])
        return dst

    def to_points(self, chunk_shape, threshold=None):
        """ Transform array to array of nonzero points.

        Parameters
        ----------
        chunk_shape : tuple
            shape of chunk
        threshold : float or None, optional
            Threshold to transform array into binary array, by default None

        Returns
        -------
        np.ndarray
            array of shape (N, 3) with coordinates of nonzero points.
        """
        axis = self.projections[0]
        points = []
        for start in range(0, self.shape[axis], chunk_shape[axis]):
            end = min(start + chunk_shape[axis], self.shape[axis])
            slices = [slice(None) for i in range(3)]
            slices[axis] = slice(start, end)
            chunk = self.load_crop(slices)
            if threshold:
                chunk = chunk > threshold
            points_ = np.stack(np.where(chunk), axis=-1)
            points_[:, 0] += start
            points += [points_]
        points = np.concatenate(points, axis=0)
        return points

    @classmethod
    def from_points(cls, points, path):
        """ Create hdf5 file from coordinates of nonzero points.

        Returns
        -------
            StorageHDF5
        """
        file_hdf5 = cls(path, mode='a')
        file_hdf5[points[0], points[1], points[2]] = 1
        return file_hdf5

    def reset(self):
        """ Reset cache. """
        self.load_existed_slide.reset(instance=self)
        self.construct_slide.reset(instance=self)

    @property
    def cache_length(self):
        """ Total amount of cached slides. """
        return len(self.load_existed_slide.cache()[self]) + len(self.construct_slide.cache()[self])

    @property
    def cache_items(self):
        """ Total size of cached slides. """
        return [*self.load_existed_slide.cache()[self].values(), *self.construct_slide.cache()[self].values()]
