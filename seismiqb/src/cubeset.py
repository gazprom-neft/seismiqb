""" Contains container for storing dataset of seismic crops. """
#pylint: disable=too-many-lines
import os
from glob import glob
import contextlib

import numpy as np
import h5py
from tqdm.auto import tqdm

from ..batchflow import FilesIndex, DatasetIndex, Dataset, Sampler, Pipeline
from ..batchflow import NumpySampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .horizon import Horizon, UnstructuredHorizon
from .metrics import HorizonMetrics
from .plotters import plot_image
from .utils import IndexedDict, round_to_array, gen_crop_coordinates, make_axis_grid, infer_tuple



class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures.

    Attributes
    ----------
    geometries : dict
        Mapping from cube names to instances of :class:`~.SeismicGeometry`, which holds information
        about that cube structure. :meth:`~.load_geometries` is used to infer that structure.
        Note that no more that one trace is loaded into the memory at a time.

    labels : dict
        Mapping from cube names to numba-dictionaries, which are mappings from (xline, iline) pairs
        into arrays of heights of horizons for a given cube.
        Note that this arrays preserve order: i-th horizon is always placed into the i-th element of the array.
    """
    #pylint: disable=too-many-public-methods
    def __init__(self, index, batch_class=SeismicCropBatch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes. """
        if not isinstance(index, FilesIndex):
            index = [index] if isinstance(index, str) else index
            index = FilesIndex(path=index, no_ext=True)
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.crop_index, self.crop_points = None, None

        self.geometries = IndexedDict({ix: SeismicGeometry(self.index.get_fullpath(ix), process=False)
                                       for ix in self.indices})
        self.labels = IndexedDict({ix: [] for ix in self.indices})
        self.samplers = IndexedDict({ix: None for ix in self.indices})
        self._sampler = None
        self._p, self._bins = None, None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None
        self.shapes_gen, self.orders_gen = None, None


    @classmethod
    def from_horizon(cls, horizon):
        """ Create dataset from an instance of Horizon. """
        cube_path = horizon.geometry.path
        dataset = SeismicCubeset(cube_path)
        dataset.geometries[0] = horizon.geometry
        dataset.labels[0] = [horizon]
        return dataset


    def __str__(self):
        msg = f'Seismic Cubeset with {len(self)} cube{"s" if len(self) > 1 else ""}:\n'
        for idx in self.indices:
            geometry = self.geometries[idx]
            labels = self.labels.get(idx, [])

            add = f'{repr(geometry)}' if hasattr(geometry, 'cube_shape') else f'{idx}'
            msg += f'    {add}{":" if labels else ""}\n'

            for horizon in labels:
                msg += f'        {horizon.name}\n'
        return msg


    def gen_batch(self, batch_size, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                  bar=False, bar_desc=None, iter_params=None, sampler=None):
        """ Allows to pass `sampler` directly to `next_batch` method to avoid re-creating of batch
        during pipeline run.
        """
        #pylint: disable=blacklisted-name
        if n_epochs is not None or shuffle or drop_last:
            raise ValueError('SeismicCubeset does not comply with `n_epochs`, `shuffle`\
                              and `drop_last`. Use `n_iters` instead! ')
        if sampler:
            sampler = sampler if callable(sampler) else sampler.sample
            points = sampler(batch_size * n_iters)

            self.crop_points = points
            self.crop_index = DatasetIndex(points[:, 0])
            return self.crop_index.gen_batch(batch_size, n_iters=n_iters, iter_params=iter_params,
                                             bar=bar, bar_desc=bar_desc)
        return super().gen_batch(batch_size, shuffle=shuffle, n_iters=n_iters, n_epochs=n_epochs,
                                 drop_last=drop_last, bar=bar, bar_desc=bar_desc, iter_params=iter_params)


    def load_geometries(self, logs=True, **kwargs):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        for ix in self.indices:
            self.geometries[ix].process(**kwargs)
            if logs:
                self.geometries[ix].log()

    def add_geometries_targets(self, paths, dst='geom_targets'):
        """Create targets from given cubes

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with target cubes.
        dst : str, optional
            Name of attribute to put targets in, by default 'geom_targets'
        """
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: None for ix in self.indices}))

        for ix in self.indices:
            getattr(self, dst)[ix] = SeismicGeometry(paths[ix])


    def convert_to_hdf5(self, postfix=''):
        """ Converts every cube in dataset from `.segy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_hdf5(postfix=postfix)


    def create_labels(self, paths=None, filter_zeros=True, dst='labels', labels_class=None, **kwargs):
        """ Create labels (horizons, facies, etc) from given paths.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.
        dst : str
            Name of attribute to put labels in.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded labels.
        """
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: dict() for ix in self.indices}))

        for ix in self.indices:
            if labels_class is None:
                if self.geometries[ix].structured:
                    labels_class = Horizon
                else:
                    labels_class = UnstructuredHorizon

            label_list = [labels_class(path, self.geometries[ix], **kwargs) for path in paths[ix]]
            label_list.sort(key=lambda label: label.h_mean)
            if filter_zeros:
                _ = [getattr(item, 'filter')() for item in label_list]
            getattr(self, dst)[ix] = [item for item in label_list if len(item.points) > 0]

    def dump_labels(self, path, fmt='npy', separate=False):
        """ Dump points to file. """
        for i in range(len(self.indices)):
            for label in self.labels[i]:
                dirname = os.path.dirname(self.index.get_fullpath(self.indices[i]))
                if path[0] == '/':
                    path = path[1:]
                dirname = os.path.join(dirname, path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                name = label.name if separate else 'faults'
                save_to = os.path.join(dirname, name + '.' + fmt)
                label.dump_points(save_to, fmt)


    @property
    def sampler(self):
        """ Lazily create sampler at the time of first access. """
        if self._sampler is None:
            self.create_sampler(p=self._p, bins=self._bins)
        return self._sampler

    @sampler.setter
    def sampler(self, sampler):
        self._sampler = sampler


    def create_sampler(self, mode='hist', p=None, transforms=None, dst='sampler', **kwargs):
        """ Create samplers for every cube and store it in `samplers`
        attribute of passed dataset. Also creates one combined sampler
        and stores it in `sampler` attribute of passed dataset.

        Parameters
        ----------
        mode : str or Sampler
            Type of sampler to be created.
            If 'hist' or 'horizon', then sampler is estimated from given labels.
            If 'numpy', then sampler is created with `kwargs` parameters.
            If instance of Sampler is provided, it must generate points from unit cube.
        p : list
            Weights for each mixture in final sampler.
        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from absolute coordinates (X, Y world-wise) to
            cube local specific and take array of shape (N, 4) as input.

        Notes
        -----
        Passed `dataset` must have `geometries` and `labels` attributes if you want to create HistoSampler.
        """
        #pylint: disable=cell-var-from-loop
        lowcut, highcut = [0, 0, 0], [1, 1, 1]
        transforms = transforms or dict()

        samplers = {}
        if not isinstance(mode, dict):
            mode = {ix: mode for ix in self.indices}

        for ix in self.indices:
            if isinstance(mode[ix], Sampler):
                sampler = mode[ix]

            elif mode[ix] == 'numpy':
                sampler = NumpySampler(**kwargs)

            elif mode[ix] == 'hist' or mode[ix] == 'horizon':
                sampler = 0 & NumpySampler('n', dim=3)
                for i, label in enumerate(self.labels[ix]):
                    label.create_sampler(**kwargs)
                    sampler = sampler | label.sampler
            else:
                sampler = NumpySampler('u', low=0, high=1, dim=3)

            sampler = sampler.truncate(low=lowcut, high=highcut)
            samplers.update({ix: sampler})
        self.samplers = samplers

        # One sampler to rule them all
        p = p or [1/len(self) for _ in self.indices]

        sampler = 0 & NumpySampler('n', dim=4)
        for i, ix in enumerate(self.indices):
            sampler_ = samplers[ix].apply(Modificator(cube_name=ix))
            sampler = sampler | (p[i] & sampler_)
        setattr(self, dst, sampler)

    def modify_sampler(self, dst, mode='iline', low=None, high=None,
                       each=None, each_start=None,
                       to_cube=False, post=None, finish=False, src='sampler'):
        """ Change given sampler to generate points from desired regions.

        Parameters
        ----------
        src : str
            Attribute with Sampler to change.
        dst : str
            Attribute to store created Sampler.
        mode : str
            Axis to modify: ilines/xlines/heights.
        low : float
            Lower bound for truncating.
        high : float
            Upper bound for truncating.
        each : int
            Keep only i-th value along axis.
        each_start : int
            Shift grid for previous parameter.
        to_cube : bool
            Transform sampled values to each cube coordinates.
        post : callable
            Additional function to apply to sampled points.
        finish : bool
            If False, instance of Sampler is put into `dst` and can be modified later.
            If True, `sample` method is put into `dst` and can be called via `D` named-expressions.

        Examples
        --------
        Split into train / test along ilines in 80/20 ratio:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='i', high=0.8)
        >>> cubeset.modify_sampler(dst='test_sampler', mode='i', low=0.9)

        Sample only every 50-th point along xlines starting from 70-th xline:

        >>> cubeset.modify_sampler(dst='train_sampler', mode='x', each=50, each_start=70)

        Notes
        -----
        It is advised to have gap between `high` for train sampler and `low` for test sampler.
        That is done in order to take into account additional seen entries due to crop shape.
        """

        # Parsing arguments
        sampler = getattr(self, src)

        mapping = {'ilines': 0, 'xlines': 1, 'heights': 2,
                   'iline': 0, 'xline': 1, 'i': 0, 'x': 1, 'h': 2}
        axis = mapping[mode]

        low, high = low or 0, high or 1
        each_start = each_start or each

        # Keep only points from region
        if (low != 0) or (high != 1):
            sampler = sampler.truncate(low=low, high=high, prob=high-low,
                                       expr=lambda p: p[:, axis+1])

        # Keep only every `each`-th point
        if each is not None:
            def filter_out(array):
                for cube_name in np.unique(array[:, 0]):
                    shape = self.geometries[cube_name].cube_shape[axis]
                    ticks = np.arange(each_start, shape, each)
                    name_idx = np.asarray(array[:, 0] == cube_name).nonzero()

                    arr = np.rint(array[array[:, 0] == cube_name][:, axis+1].astype(float)*shape).astype(int)
                    array[name_idx, np.full_like(name_idx, axis+1)] = round_to_array(arr, ticks).astype(float) / shape
                return array

            sampler = sampler.apply(filter_out)

        # Change representation of points from unit cube to cube coordinates
        if to_cube:
            def get_shapes(name):
                return self.geometries[name].cube_shape

            def coords_to_cube(array):
                shapes = np.array(list(map(get_shapes, array[:, 0])))
                array[:, 1:] = np.rint(array[:, 1:].astype(float) * shapes).astype(int)
                return array

            sampler = sampler.apply(coords_to_cube)

        # Apply additional transformations to points
        if callable(post):
            sampler = sampler.apply(post)

        if finish:
            setattr(self, dst, sampler.sample)
        else:
            setattr(self, dst, sampler)

    def show_slices(self, idx=0, src_sampler='sampler', n=10000, normalize=False, shape=None,
                    adaptive_slices=False, grid_src='quality_grid', side_view=False, **kwargs):
        """ Show actually sampled slices of desired shape. """
        sampler = getattr(self, src_sampler)
        if callable(sampler):
            #pylint: disable=not-callable
            points = sampler(n)
        else:
            points = sampler.sample(n)
        batch = (self.p.crop(points=points, shape=shape, side_view=side_view,
                             adaptive_slices=adaptive_slices, grid_src=grid_src)
                 .next_batch(self.size))

        unsalted = np.array([batch.unsalt(item) for item in batch.indices])
        background = np.zeros_like(self.geometries[idx].zero_traces)

        for slice_ in np.array(batch.locations)[unsalted == self.indices[idx]]:
            idx_i, idx_x, _ = slice_
            background[idx_i, idx_x] += 1

        if normalize:
            background = (background > 0).astype(int)

        kwargs = {
            'title': f'Sampled slices on {self.indices[idx]}',
            'xlabel': 'ilines', 'ylabel': 'xlines',
            'cmap': 'Reds', 'interpolation': 'bilinear',
            **kwargs
        }
        plot_image(background, **kwargs)
        return batch

    def show_points(self, idx=0, src_labels='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.geometries[idx].cube_shape[:-1])
        for label in getattr(self, src_labels)[idx]:
            map_[label.points[:, 0], label.points[:, 1]] += 1
        labels_class = type(getattr(self, src_labels)[idx][0]).__name__
        map_[map_ == 0] = np.nan
        kwargs = {
            'title': f'{labels_class} on {self.indices[idx]}',
            'xlabel': self.geometries[idx].index_headers[0],
            'ylabel': self.geometries[idx].index_headers[1],
            'cmap': 'Reds',
            **kwargs
        }
        plot_image(map_, **kwargs)


    def load(self, label_dir=None, filter_zeros=True, dst_labels='labels', p=None, bins=None, **kwargs):
        """ Load everything: geometries, point clouds, labels, samplers.

        Parameters
        ----------
        label_dir : str
            Relative path from each cube to directory with labels.
        p : sequence of numbers
            Proportions of different cubes in sampler.
        filter_zeros : bool
            Whether to remove labels on zero-traces.
        """
        _ = kwargs
        label_dir = label_dir or '/INPUTS/HORIZONS/RAW/*'

        paths_txt = {}
        for i in range(len(self)):
            dir_path = '/'.join(self.index.get_fullpath(self.indices[i]).split('/')[:-1])
            label_dir_ = label_dir if isinstance(label_dir, str) else label_dir[self.indices[i]]
            dir_ = dir_path + label_dir_
            paths_txt[self.indices[i]] = glob(dir_)

        self.load_geometries(**kwargs)
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros, dst=dst_labels, **kwargs)
        self._p, self._bins = p, bins # stored for later sampler creation


    def make_grid(self, cube_name, crop_shape, ilines=None, xlines=None, heights=None,
                  strides=None, overlap=None, overlap_factor=None,
                  batch_size=16, filtering_matrix=None, filter_threshold=0):
        """ Create regular grid of points in cube.
        This method is usually used with `assemble_predict` action of SeismicCropBatch.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.
        crop_shape : sequence
            Shape of model inputs.
        ilines : sequence of two elements
            Location of desired prediction, iline-wise.
            If None, whole cube ranges will be used.
        xlines : sequence of two elements
            Location of desired prediction, xline-wise.
            If None, whole cube ranges will be used.
        heights : sequence of two elements
            Location of desired prediction, depth-wise.
            If None, whole cube ranges will be used.
        strides : float or sequence
            Distance between grid points.
        overlap_factor : float or sequence
            Overlapping ratio of successive crops.
            Can be seen as `how many crops would cross every through point`.
            If both overlap and overlap_factor are provided, overlap_factor will be used.
        batch_size : int
            Amount of returned points per generator call.
        filtering_matrix : ndarray
            Binary matrix of (ilines_len, xlines_len) shape with ones corresponding
            to areas that can be skipped in the grid.
            E.g., a matrix with zeros at places where a horizon is present and ones everywhere else.
            If None, geometry.zero_traces matrix will be used.
        filter_threshold : int or float in [0, 1]
            Exclusive lower bound for non-gap number of points (with 0's in the filtering_matrix)
            in a crop in the grid. Default value is 0.
            If float, proportion from the total number of traces in a crop will be computed.
        """
        #pylint: disable=too-many-branches
        geometry = self.geometries[cube_name]

        if isinstance(overlap_factor, (int, float)):
            overlap_factor = [overlap_factor] * 3
        if strides is None:
            if overlap:
                strides = [c - o for c, o in zip(crop_shape, overlap)]
            elif overlap_factor:
                strides = [max(1, int(item // factor)) for item, factor in zip(crop_shape, overlap_factor)]
            else:
                strides = crop_shape

        if 0 < filter_threshold < 1:
            filter_threshold = int(filter_threshold * np.prod(crop_shape[:2]))

        filtering_matrix = geometry.zero_traces if filtering_matrix is None else filtering_matrix
        if (filtering_matrix.shape != geometry.cube_shape[:2]).all():
            raise ValueError('Filtering_matrix shape must be equal to (ilines_len, xlines_len)')

        ilines = (0, geometry.ilines_len) if ilines is None else ilines
        xlines = (0, geometry.xlines_len) if xlines is None else xlines
        heights = (0, geometry.depth) if heights is None else heights

        # Assert ranges are valid
        if ilines[0] < 0 or xlines[0] < 0 or heights[0] < 0:
            raise ValueError('Ranges must contain within the cube.')

        if ilines[1] > geometry.ilines_len or \
           xlines[1] > geometry.xlines_len or \
           heights[1] > geometry.depth:
            raise ValueError('Ranges must contain within the cube.')

        ilines_grid = make_axis_grid(ilines, strides[0], geometry.ilines_len, crop_shape[0])
        xlines_grid = make_axis_grid(xlines, strides[1], geometry.xlines_len, crop_shape[1])
        heights_grid = make_axis_grid(heights, strides[2], geometry.depth, crop_shape[2])

        # Every point in grid contains reference to cube
        # in order to be valid input for `crop` action of SeismicCropBatch
        grid = []
        for il in ilines_grid:
            for xl in xlines_grid:
                if np.prod(crop_shape[:2]) - np.sum(filtering_matrix[il: il + crop_shape[0],
                                                                     xl: xl + crop_shape[1]]) > filter_threshold:
                    for h in heights_grid:
                        point = [cube_name, il, xl, h]
                        grid.append(point)
        grid = np.array(grid, dtype=object)

        # Creating and storing all the necessary things
        # Check if grid is not empty
        shifts = np.array([ilines[0], xlines[0], heights[0]])
        if len(grid) > 0:
            grid_gen = (grid[i:i+batch_size]
                        for i in range(0, len(grid), batch_size))
            grid_array = grid[:, 1:].astype(int) - shifts
        else:
            grid_gen = iter(())
            grid_array = []

        predict_shape = (ilines[1] - ilines[0],
                         xlines[1] - xlines[0],
                         heights[1] - heights[0])

        self.grid_gen = lambda: next(grid_gen)
        self.grid_iters = - (-len(grid) // batch_size)
        self.grid_info = {
            'grid_array': grid_array,
            'predict_shape': predict_shape,
            'crop_shape': crop_shape,
            'strides': strides,
            'cube_name': cube_name,
            'geometry': geometry,
            'range': [ilines, xlines, heights],
            'shifts': shifts,
            'length': len(grid_array),
            'unfiltered_length': len(ilines_grid) * len(xlines_grid) * len(heights_grid)
        }


    def mask_to_horizons(self, src, cube_name, threshold=0.5, averaging='mean', minsize=0,
                         dst='predicted_horizons', prefix='predict', src_grid_info='grid_info'):
        """ Convert mask to a list of horizons.

        Parameters
        ----------
        src : str or array
            Source-mask. Can be either a name of attribute or mask itself.
        dst : str
            Attribute to write the horizons in.
        threshold : float
            Parameter of mask-thresholding.
        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).
        minsize : int
            Minimum length of a horizon to be saved.
        prefix : str
            Name of horizon to use.
        """
        #TODO: add `chunks` mode
        mask = getattr(self, src) if isinstance(src, str) else src

        grid_info = getattr(self, src_grid_info)

        horizons = Horizon.from_mask(mask, grid_info,
                                     threshold=threshold, averaging=averaging, minsize=minsize, prefix=prefix)
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: dict() for ix in self.indices}))

        getattr(self, dst)[cube_name] = horizons


    def merge_horizons(self, src, mean_threshold=2.0, adjacency=3, minsize=50):
        """ Iteratively try to merge every horizon in a list to every other, until there are no possible merges. """
        horizons = getattr(self, src)
        horizons = Horizon.merge_list(horizons, mean_threshold=mean_threshold, adjacency=adjacency, minsize=minsize)
        if isinstance(src, str):
            setattr(self, src, horizons)


    def compare_to_labels(self, horizon, src_labels='labels', offset=0, absolute=True,
                          printer=print, hist=True, plot=True):
        """ Compare given horizon to labels in dataset.

        Parameters
        ----------
        horizon : :class:`.Horizon`
            Horizon to evaluate.
        offset : number
            Value to shift horizon down. Can be used to take into account different counting bases.
        """
        for idx in self.indices:
            if horizon.geometry.name == self.geometries[idx].name:
                horizons_to_compare = getattr(self, src_labels)[idx]
                break
        HorizonMetrics([horizon, horizons_to_compare]).evaluate('compare', agg=None,
                                                                absolute=absolute, offset=offset,
                                                                printer=printer, hist=hist, plot=plot)


    def show_slide(self, loc, idx=0, axis='iline', zoom_slice=None, mode='overlap',
                   n_ticks=5, delta_ticks=100, **kwargs):
        """ Show full slide of the given cube on the given line.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        idx : str, int
            Number of cube in the index to use.
        mode : str
            Way of showing results. Can be either `overlap` or `separate`.
        backend : str
            Backend to use for render. Can be either 'plotly' or 'matplotlib'. Whenever
            using 'plotly', also use slices to make the rendering take less time.
        """
        components = ('images', 'masks') if list(self.labels.values())[0] else ('images',)
        cube_name = self.indices[idx]
        geometry = self.geometries[cube_name]
        crop_shape = np.array(geometry.cube_shape)

        axis = geometry.parse_axis(axis)
        point = np.array([[cube_name, 0, 0, 0]], dtype=object)
        point[0, axis + 1] = loc
        crop_shape[axis] = 1

        pipeline = (Pipeline()
                    .crop(points=point, shape=crop_shape)
                    .load_cubes(dst='images')
                    .scale(mode='q', src='images'))

        if 'masks' in components:
            indices = kwargs.pop('indices', -1)
            width = kwargs.pop('width', 5)
            labels_pipeline = (Pipeline()
                               .create_masks(dst='masks', width=width, indices=indices))

            pipeline = pipeline + labels_pipeline

        batch = (pipeline << self).next_batch(len(self), n_epochs=None)
        imgs = [np.squeeze(getattr(batch, comp)) for comp in components]
        xticks = list(range(imgs[0].shape[0]))
        yticks = list(range(imgs[0].shape[1]))

        if zoom_slice:
            imgs = [img[zoom_slice] for img in imgs]
            xticks = xticks[zoom_slice[0]]
            yticks = yticks[zoom_slice[1]]

        # Plotting defaults
        header = geometry.axis_names[axis]
        total = geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = geometry.index_headers[0]
            ylabel = geometry.index_headers[1]

        xticks = xticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [xticks[-1]]
        xticks = sorted(list(set(xticks)))
        yticks = yticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [yticks[-1]]
        yticks = sorted(list(set(yticks)), reverse=True)

        if len(xticks) > 2 and (xticks[-1] - xticks[-2]) < delta_ticks:
            xticks.pop(-2)
        if len(yticks) > 2 and (yticks[0] - yticks[1]) < delta_ticks:
            yticks.pop(1)

        kwargs = {
            'mode': mode,
            'title': f'Data slice on `{geometry.name}\n {header} {loc} out of {total}',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': xticks,
            'yticks': yticks,
            'y': 1.02,
            **kwargs
        }

        plot_image(imgs, **kwargs)
        return batch


    def make_extension_grid(self, cube_name, crop_shape, labels_src='predicted_labels',
                            stride=10, batch_size=16, coverage=True, **kwargs):
        """ Create a non-regular grid of points in a cube for extension procedure.
        Each point defines an upper rightmost corner of a crop which contains a holey
        horizon.

        Parameters
        ----------
        cube_name : str
            Reference to the cube. Should be a valid key for the `labels_src` attribute.
        crop_shape : sequence
            The desired shape of the crops.
            Note that final shapes are made in both xline and iline directions. So if
            crop_shape is (1, 64, 64), crops of both (1, 64, 64) and (64, 1, 64) shape
            will be defined.
        labels_src : str or instance of :class:`.Horizon`
            Horizon to be extended.
        stride : int
            Distance between a horizon border and a corner of a crop.
        batch_size : int
            Batch size fed to the model.
        coverage : bool or array, optional
            A boolean array of size (ilines_len, xlines_len) indicating points that will
            not be used as new crop coordinates, e.g. already covered points.
            If True then coverage array will be initialized with zeros and updated with
            covered points.
            If False then all points from the horizon border will be used.
        """
        horizon = getattr(self, labels_src)[cube_name][0] if isinstance(labels_src, str) else labels_src

        zero_traces = horizon.geometry.zero_traces
        hor_matrix = horizon.full_matrix.astype(np.int32)
        coverage_matrix = np.zeros_like(zero_traces) if isinstance(coverage, bool) else coverage

        # get horizon boundary points in horizon.matrix coordinates
        border_points = np.array(list(zip(*np.where(horizon.boundaries_matrix))))

        # shift border_points to global coordinates
        border_points[:, 0] += horizon.i_min
        border_points[:, 1] += horizon.x_min

        crops, orders, shapes = [], [], []

        for i, point in enumerate(border_points):
            if coverage_matrix[point[0], point[1]] == 1:
                continue

            result = gen_crop_coordinates(point,
                                          hor_matrix, zero_traces,
                                          stride, crop_shape, horizon.geometry.depth,
                                          horizon.FILL_VALUE, **kwargs)
            if not result:
                continue
            new_point, shape, order = result
            crops.extend(new_point)
            shapes.extend(shape)
            orders.extend(order)

            if coverage is not False:
                for _point, _shape in zip(new_point, shape):
                    coverage_matrix[_point[0]: _point[0] + _shape[0],
                                    _point[1]: _point[1] + _shape[1]] = 1

        crops = np.array(crops, dtype=np.object).reshape(-1, 3)
        cube_names = np.array([cube_name] * len(crops), dtype=np.object).reshape(-1, 1)
        shapes = np.array(shapes)
        crops = np.concatenate([cube_names, crops], axis=1)

        crops_gen = (crops[i:i+batch_size]
                     for i in range(0, len(crops), batch_size))
        shapes_gen = (shapes[i:i+batch_size]
                      for i in range(0, len(shapes), batch_size))
        orders_gen = (orders[i:i+batch_size]
                      for i in range(0, len(orders), batch_size))

        self.grid_gen = lambda: next(crops_gen)
        self.shapes_gen = lambda: next(shapes_gen)
        self.orders_gen = lambda: next(orders_gen)
        self.grid_iters = - (-len(crops) // batch_size)
        self.grid_info = {'cube_name': cube_name,
                          'geometry': horizon.geometry}


    def assemble_crops(self, crops, grid_info='grid_info', order=None, fill_value=0):
        """ Glue crops together in accordance to the grid.

        Note
        ----
        In order to use this action you must first call `make_grid` method of SeismicCubeset.

        Parameters
        ----------
        crops : sequence
            Sequence of crops.
        grid_info : dict or str
            Dictionary with information about grid. Should be created by `make_grid` method.
        order : tuple of int
            Axes-param for `transpose`-operation, applied to a mask before fetching point clouds.
            Default value of (2, 0, 1) is applicable to standart pipeline with one `rotate_axes`
            applied to images-tensor.
        fill_value : float
            Fill_value for background array if `len(crops) == 0`.

        Returns
        -------
        np.ndarray
            Assembled array of shape `grid_info['predict_shape']`.
        """
        if isinstance(grid_info, str):
            if not hasattr(self, grid_info):
                raise ValueError('Pass grid_info dictionary or call `make_grid` method to create grid_info.')
            grid_info = getattr(self, grid_info)

        # Do nothing if number of crops differ from number of points in the grid.
        if len(crops) != len(grid_info['grid_array']):
            raise ValueError('Length of crops must be equal to number of crops in a grid')
        order = order or (2, 0, 1)
        crops = np.array(crops)
        if len(crops) != 0:
            fill_value = np.min(crops)

        grid_array = grid_info['grid_array']
        crop_shape = grid_info['crop_shape']
        background = np.full(grid_info['predict_shape'], fill_value)

        for j, (i, x, h) in enumerate(grid_array):
            crop_slice, background_slice = [], []

            for k, start in enumerate((i, x, h)):
                if start >= 0:
                    end = min(background.shape[k], start + crop_shape[k])
                    crop_slice.append(slice(0, end - start))
                    background_slice.append(slice(start, end))
                else:
                    crop_slice.append(slice(-start, None))
                    background_slice.append(slice(None))

            crop = np.transpose(crops[j], order)
            crop = crop[crop_slice]
            previous = background[background_slice]
            background[background_slice] = np.maximum(crop, previous)

        return background

    def make_prediction(self, path_hdf5, pipeline, crop_shape, crop_stride,
                        idx=0, src='predictions', chunk_shape=None, chunk_stride=None, batch_size=8,
                        pbar=True):
        """ Create hdf5 file with prediction.

        Parameters
        ----------
        path_hdf5 : str

        pipeline : Pipeline
            pipeline for inference
        crop_shape : int, tuple or None
            shape of crops. Must be the same as defined in pipeline.
        crop_stride : int
            stride for crops
        idx : int
            index of cube to infer
        src : str
            pipeline variable for predictions
        chunk_shape : int, tuple or None
            shape of chunks.
        chunk_stride : int
            stride for chunks
        batch_size : int

        pbar : bool
            progress bar
        """
        geometry = self.geometries[idx]
        chunk_shape = infer_tuple(chunk_shape, geometry.cube_shape)
        chunk_stride = infer_tuple(chunk_stride, chunk_shape)

        cube_shape = geometry.cube_shape
        chunk_grid = [
            make_axis_grid((0, cube_shape[i]), chunk_stride[i], cube_shape[i], crop_shape[i])
            for i in range(2)
        ]
        chunk_grid = np.stack(np.meshgrid(*chunk_grid), axis=-1).reshape(-1, 2)

        if os.path.exists(path_hdf5):
            os.remove(path_hdf5)

        if pbar:
            total = 0
            for i_min, x_min in chunk_grid:
                i_max = min(i_min+chunk_shape[0], cube_shape[0])
                x_max = min(x_min+chunk_shape[1], cube_shape[1])
                self.make_grid(
                    self.indices[idx], crop_shape,
                    [i_min, i_max], [x_min, x_max], [0, geometry.depth-1],
                    strides=crop_stride, batch_size=batch_size
                )
                total += self.grid_iters

        with h5py.File(path_hdf5, "a") as file_hdf5:
            aggregation_map = np.zeros(cube_shape[:-1])
            cube_hdf5 = file_hdf5.create_dataset('cube', cube_shape)
            context = tqdm(total=total) if pbar else contextlib.suppress()
            with context as progress_bar:
                for i_min, x_min in chunk_grid:
                    i_max = min(i_min+chunk_shape[0], cube_shape[0])
                    x_max = min(x_min+chunk_shape[1], cube_shape[1])
                    self.make_grid(
                        self.indices[idx], crop_shape,
                        [i_min, i_max], [x_min, x_max], [0, geometry.depth-1],
                        strides=crop_stride, batch_size=batch_size
                    )
                    chunk_pipeline = pipeline << self
                    for _ in range(self.grid_iters):
                        _ = chunk_pipeline.next_batch(len(self))
                        if pbar:
                            progress_bar.update()

                    # Write to hdf5
                    slices = tuple([slice(*item) for item in self.grid_info['range']])
                    prediction = self.assemble_crops(chunk_pipeline.v(src), order=(0, 1, 2))
                    aggregation_map[tuple(slices[:-1])] += 1
                    cube_hdf5[slices[0], slices[1], slices[2]] = +prediction
                cube_hdf5[:] = cube_hdf5 / np.expand_dims(aggregation_map, axis=-1)

class Modificator:
    """ Converts array to `object` dtype and prepends the `cube_name` column.
    Picklable, unlike inline lambda function.
    """
    def __init__(self, cube_name):
        self.cube_name = cube_name

    def __call__(self, points):
        points = points.astype(np.object)
        return np.concatenate([np.full((len(points), 1), self.cube_name), points], axis=1)
