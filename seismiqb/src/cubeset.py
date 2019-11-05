""" Contains container for storing dataset of seismic crops. """
import os
from glob import glob

import dill
import numpy as np
import matplotlib.pyplot as plt

from ..batchflow import Dataset, Sampler, Pipeline
from ..batchflow import B, V, D
from ..batchflow import HistoSampler, NumpySampler, ConstantSampler
from ..batchflow.models.tf import TFModel

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch
from .utils import read_point_cloud, make_labels_dict, _filter_labels, _filter_point_cloud
from .utils import _get_horizons, compare_horizons, dump_horizon, round_to_array, convert_to_numba_dict
from .utils import labels_to_depth_map, get_cube_values, compute_corrs, FILL_VALUE_A
from .utils import update_horizon_dict, make_grid_info, compute_next_points, create_predict_ppl
from .plot_utils import show_labels, show_sampler, plot_slide, plot_from_above
from .plot_utils import plot_from_above_rgb, plot_extension_history


class SeismicCubeset(Dataset):
    """ Stores indexing structure for dataset of seismic cubes along with additional structures. """
    #pylint: disable=too-many-public-methods
    def __init__(self, index, batch_class=SeismicCropBatch, preloaded=None, *args, **kwargs):
        """ Initialize additional attributes.
        """
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)
        self.geometries = {ix: SeismicGeometry(self.index.get_fullpath(ix)) for ix in self.indices}
        self.point_clouds = {ix: np.array([]) for ix in self.indices}
        self.labels = {ix: dict() for ix in self.indices}
        self.samplers = {ix: None for ix in self.indices}
        self.sampler = None

        self.grid_gen, self.grid_info, self.grid_iters = None, None, None


    def load_geometries(self, path=None, logs=True):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        path : str
            Path to the dill-file to load geometries from.

        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        if isinstance(path, str):
            with open(path, 'rb') as file:
                self.geometries = dill.load(file)
        else:
            for ix in self.indices:
                self.geometries[ix].load()
                if logs:
                    self.geometries[ix].log()
        return self

    def convert_to_h5py(self, postfix='', dtype=np.float32):
        """ Converts every cube in dataset from `.sgy` to `.hdf5`. """
        for ix in self.indices:
            self.geometries[ix].make_h5py(postfix=postfix, dtype=dtype)
        return self

    def save_geometries(self, save_to):
        """ Save dill-serialized geometries for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.geometries, file)
        return self


    def load_point_clouds(self, paths=None, path=None, **kwargs):
        """ Load point-cloud of labels for each cube in dataset.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.

        path : str
            Path to the dill-file to load point clouds from.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded point clouds.
        """
        if isinstance(path, str):
            with open(path, 'rb') as file:
                self.point_clouds = dill.load(file)
        else:
            for ix in self.indices:
                self.point_clouds[ix] = read_point_cloud(paths[ix], **kwargs)
                self.geometries[ix].horizon_list = paths[ix]
        return self

    def filter_point_clouds(self, src='point_clouds'):
        """ Remove points corresponding to zero-traces. """
        for ix in self.indices:
            geom = getattr(self, 'geometries').get(ix)
            ilines_offset, xlines_offset = geom.ilines_offset, geom.xlines_offset
            zero_matrix = geom.zero_traces
            getattr(self, src)[ix] = _filter_point_cloud(getattr(self, src)[ix], zero_matrix,
                                                         ilines_offset, xlines_offset)

    def save_point_clouds(self, save_to):
        """ Save dill-serialized point clouds for a dataset of seismic-cubes on disk.
        """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(self.point_clouds, file)
        return self


    def create_labels(self, path=None, transforms=None, src='point_clouds', dst='labels'):
        """ Make labels in inline-xline coordinates using cloud of points and supplied transforms.

        Parameters
        ----------
        path : str
            Path to the dill-file to load labels from.

        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from (i, x, h, n) to (i, x, d, n) and take array of shape (N, 4) as input,
            where d (depth) is corrected h (height) (divided by sample rate and moved by time-delay value).

        src : str
            Attribute with saved point clouds.

        Returns
        SeismicCubeset
            Same instance with loaded labels.
        """
        point_clouds = getattr(self, src) if isinstance(src, str) else src
        transforms = transforms or dict()
        if not hasattr(self, dst):
            setattr(self, dst, {})

        if isinstance(path, str):
            try:
                with open(path, 'rb') as file:
                    setattr(self, dst, dill.load(file))
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
        else:
            for ix in self.indices:
                point_cloud = point_clouds.get(ix)
                geom = getattr(self, 'geometries').get(ix)
                transform = transforms.get(ix) or geom.height_correction
                getattr(self, dst)[ix] = make_labels_dict(transform(point_cloud))
        return self

    def filter_labels(self, src='labels'):
        """ Remove labels corresponding to zero-traces.
        Parameters
        ----------
        src : str
            Attribute with labels-dictionary.
        """
        for ix in self.indices:
            geom = getattr(self, 'geometries').get(ix)
            ilines_offset, xlines_offset = geom.ilines_offset, geom.xlines_offset
            zero_matrix = geom.zero_traces
            _filter_labels(getattr(self, src)[ix], zero_matrix, ilines_offset, xlines_offset)

    def save_labels(self, save_to, src='labels'):
        """ Save dill-serialized labels for a dataset of seismic-cubes on disk. """
        if isinstance(save_to, str):
            try:
                with open(save_to, 'wb') as file:
                    dill.dump(getattr(self, src), file)
            except TypeError:
                raise NotImplementedError("Numba dicts are yet to support serializing")
        return self

    def dump_labels(self, dir_name=None, src_labels='labels'):
        """ Dump labels-dict into separate txt files.

        Parameters
        ----------
        src_labels : str
            Attribute with dictionary: (iline, xline) -> array of heights.
            Each entry in the array is saved into separate file.

        dir_name : str
            Relative (to cube location) path to directory with saved horizons.
        """
        for ix in self.indices:
            labels = getattr(self, src_labels).get(ix)
            geom = self.geometries[ix]

            save_dir = os.path.join(os.path.dirname(geom.path), dir_name)
            try:
                os.mkdir(save_dir)
            except FileExistsError:
                pass

            for idx, path in enumerate(geom.horizon_list):
                name = os.path.basename(path)
                name = os.path.join(save_dir, os.path.basename(path))
                dump_horizon(labels, geom, name, idx=idx, offset=0)

    def show_labels(self, idx=0, hor_idx=None, src='labels', return_image=False, show_plot=True):
        """ Draw points with hand-labeled horizons from above. """
        img = show_labels(self, idx=idx, hor_idx=hor_idx, src=src, show_plot=show_plot)
        if return_image:
            return img
        return None

    def create_sampler(self, mode='hist', p=None, transforms=None, dst='sampler', **kwargs):
        """ Create samplers for every cube and store it in `samplers`
        attribute of passed dataset. Also creates one combined sampler
        and stores it in `sampler` attribute of passed dataset.

        Parameters
        ----------
        mode : str or Sampler
            Type of sampler to be created.
            If 'hist', then sampler is estimated from given labels.
            If 'numpy', then sampler is created with `kwargs` parameters.
            If instance of Sampler is provided, it must generate points from unit cube.

        p : list
            Weights for each mixture in final sampler.

        transforms : dict
            Mapping from indices to callables. Each callable should define
            way to map point from absolute coordinates (X, Y world-wise) to
            cube local specific and take array of shape (N, 4) as input.

        Note
        ----
        Passed `dataset` must have `geometries` and `labels` attributes if
        you want to create HistoSampler.
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
            elif mode[ix] == 'hist':
                point_cloud = getattr(self, 'point_clouds')[ix]

                geom = getattr(self, 'geometries')[ix]
                offsets = np.array([geom.ilines_offset, geom.xlines_offset, 0])
                cube_shape = np.array(geom.cube_shape)
                to_cube = lambda points: (points[:, :3] - offsets)/cube_shape
                default = lambda points: to_cube(geom.height_correction(points))

                transform = transforms.get(ix) or default
                cube_array = transform(point_cloud)

                # Size of ticks along each respective axis
                default_bins = cube_shape // np.array([5, 20, 20])
                bins = kwargs.get('bins') or default_bins
                sampler = HistoSampler(np.histogramdd(cube_array, bins=bins))
            else:
                sampler = NumpySampler('u', low=0, high=1, dim=3)

            sampler = sampler.truncate(low=lowcut, high=highcut)
            samplers.update({ix: sampler})
        self.samplers = samplers

        # One sampler to rule them all
        p = p or [1/len(self) for _ in self.indices]

        sampler = 0 & NumpySampler('n', dim=4)
        for i, ix in enumerate(self.indices):
            sampler_ = (ConstantSampler(ix)
                        & samplers[ix].apply(lambda d: d.astype(np.object)))
            sampler = sampler | (p[i] & sampler_)
        setattr(self, dst, sampler)
        return self

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
            def get_shape(name):
                return self.geometries[name].cube_shape[axis]

            def get_ticks(name):
                shape = self.geometries[name].cube_shape[axis]
                return np.arange(each_start, shape, each)

            def filter_out(array):
                shapes = np.array(list(map(get_shape, array[:, 0])))
                ticks = np.array(list(map(get_ticks, array[:, 0])))
                arr = (array[:, axis+1]*shapes).astype(int)
                array[:, axis+1] = round_to_array(arr, ticks) / shapes
                return array

            sampler = sampler.apply(filter_out)

        # Change representation of points from unit cube to cube coordinates
        if to_cube:
            def get_shapes(name):
                return self.geometries[name].cube_shape

            def coords_to_cube(array):
                shapes = np.array(list(map(get_shapes, array[:, 0])))
                array[:, 1:] = (array[:, 1:] * shapes).astype(int)
                return array

            sampler = sampler.apply(coords_to_cube)

        # Apply additional transformations to points
        if callable(post):
            sampler = sampler.apply(post)

        if finish:
            setattr(self, dst, sampler.sample)
        else:
            setattr(self, dst, sampler)

    def show_sampler(self, idx=0, src_sampler='sampler', n=100000, eps=3):
        """ Generate a lot of points and look at their (iline, xline) positions.

        Parameters
        ----------
        idx : int
            Number of cube to show sampler on.

        src_sampler : str
            Name of attribute with sampler in it.
            Must generate points in cubic coordinates, which can be achieved by `modify_sampler` method.

        n : int
            Number of points to generate.

        eps : int
            Window of painting.
        """
        show_sampler(self, idx=idx, src_sampler=src_sampler, n=n, eps=eps)


    def load(self, horizon_dir=None, p=None, filter_zeros=True):
        """ Load everything: geometries, point clouds, labels, samplers.

        Parameters
        ----------
        horizon_dir : str
            Relative path from each cube to directory with horizons.

        p : sequence of numbers
            Proportions of different cubes in sampler.

        filter_zeros : bool
            Whether to remove labels on zero-traces.
        """
        horizon_dir = horizon_dir or '/BEST_HORIZONS/*'

        paths_txt = {}
        for i in range(len(self)):
            dir_path = '/'.join(self.index.get_fullpath(self.indices[i]).split('/')[:-1])
            dir_ = dir_path + horizon_dir
            paths_txt[self.indices[i]] = glob(dir_)
        self.load_geometries()
        self.load_point_clouds(paths=paths_txt)
        if filter_zeros:
            self.filter_point_clouds()
        self.create_labels()
        self.create_sampler(p=p)
        return self

    def save_attr(self, name, save_to):
        """ Save attribute of dataset to disk. """
        if isinstance(save_to, str):
            with open(save_to, 'wb') as file:
                dill.dump(getattr(self, name), file)
        return self


    def make_grid(self, cube_name, crop_shape, ilines_range, xlines_range, h_range, strides=None, batch_size=16):
        """ Create regular grid of points in cube.
        This method is usually used with `assemble_predict` action of SeismicCropBatch.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.

        crop_shape : array-like
            Shape of model inputs.

        ilines_range : array-like of two elements
            Location of desired prediction, iline-wise.

        xlines_range : array-like of two elements
            Location of desired prediction, xline-wise.

        h_range : array-like of two elements
            Location of desired prediction, depth-wise.

        strides : array-like
            Distance between grid points.

        batch_size : int
            Amount of returned points per generator call.

        Returns
        -------
        SeismicCubeset
            Same instance with grid generator and grid information in attributes.

        """
        geom = self.geometries[cube_name]
        strides = strides or crop_shape

        # Assert ranges are valid
        if ilines_range[0] < 0 or \
           xlines_range[0] < 0 or \
           h_range[0] < 0:
            raise ValueError('Ranges must contain in the cube.')

        if ilines_range[1] >= geom.ilines_len or \
           xlines_range[1] >= geom.xlines_len or \
           h_range[1] >= geom.depth:
            raise ValueError('Ranges must contain in the cube.')

        # Make separate grids for every axis
        def _make_axis_grid(axis_range, stride, length, crop_shape):
            grid = np.arange(*axis_range, stride)
            grid_ = [x for x in grid if x + crop_shape < length]
            if len(grid) != len(grid_):
                grid_ += [axis_range[1] - crop_shape]
            return sorted(grid_)

        ilines = _make_axis_grid(ilines_range, strides[0], geom.ilines_len, crop_shape[0])
        xlines = _make_axis_grid(xlines_range, strides[1], geom.xlines_len, crop_shape[1])
        hs = _make_axis_grid(h_range, strides[2], geom.depth, crop_shape[2])

        # Every point in grid contains reference to cube
        # in order to be valid input for `crop` action of SeismicCropBatch
        grid = []
        for il in ilines:
            for xl in xlines:
                for h in hs:
                    point = [cube_name, il, xl, h]
                    grid.append(point)
        grid = np.array(grid, dtype=object)

        # Creating and storing all the necessary things
        grid_gen = (grid[i:i+batch_size]
                    for i in range(0, len(grid), batch_size))

        offsets = np.array([min(grid[:, 1]),
                            min(grid[:, 2]),
                            min(grid[:, 3])])

        predict_shape = (ilines_range[1] - ilines_range[0],
                         xlines_range[1] - xlines_range[0],
                         h_range[1] - h_range[0])

        grid_array = grid[:, 1:].astype(int) - offsets

        self.grid_gen = lambda: next(grid_gen)
        self.grid_iters = - (-len(grid) // batch_size)
        self.grid_info = {'grid_array': grid_array,
                          'predict_shape': predict_shape,
                          'crop_shape': crop_shape,
                          'cube_name': cube_name,
                          'range': [ilines_range, xlines_range, h_range]}
        return self

    def get_point_cloud(self, src, dst, threshold=0.5, averaging='mean', coordinates='cubic', separate=True,
                        transforms=None):
        """ Compute point cloud of horizons from a mask, save it into the 'cubeset'-attribute.

        Parameters
        ----------
        src : str or array
            Source-mask. Can be either a name of attribute or mask itself.

        dst : str
            Attribute of `cubeset` to write the horizons in.

        threshold : float
            Parameter of mask-thresholding.

        averaging : str
            Method of pandas.groupby used for finding the center of a horizon
            for each (iline, xline).

        coordinates : str
            Coordinates to use for keys of point-cloud. Can be either 'cubic'
            'lines' or None. In case of None, mask-coordinates are used. Mode 'cubic'
            requires 'grid_info'-attribute; can be run after `make_grid`-method. Mode 'lines'
            requires both 'grid_info' and 'geometries'-attributes to be loaded.

        separate : bool
            Whether to write horizonts in separate dictionaries or in one common.

        Returns
        -------
        dict or list of dict
            If separate is False, then one dictionary returned with keys being pairs of
            (iline, xline) and values being lists of heights.
            If separate is True, then list of dictionaries is returned, with every dictionary being
            mapping from pairs of (iline, xline) to height from each individual horizont.
        """
        # fetch mask-array
        mask = getattr(self, src) if isinstance(src, str) else src

        # prepare coordinate-transforms
        if coordinates is None:
            if transforms is None:
                transforms = [lambda x: x for _ in range(3)]
        elif coordinates == 'cubic':
            shifts = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda x_, shift=shift: x_ + shift for shift in shifts]
        elif coordinates == 'lines':
            geom = self.geometries[self.grid_info['cube_name']]
            i_shift, x_shift, h_shift = [axis_range[0] for axis_range in self.grid_info['range']]
            transforms = [lambda i_: geom.ilines[i_ + i_shift], lambda x_: geom.xlines[x_ + x_shift],
                          lambda h_: h_ + h_shift]

        # get horizons
        setattr(self, dst, _get_horizons(mask, threshold, averaging, transforms, separate))

        if separate:
            horizons = getattr(self, dst)
            horizons.sort(key=len, reverse=True)
            for i, horizon in enumerate(horizons):
                setattr(self, dst+'_'+str(i), horizon)
        return self

    def compare_to_labels(self, horizon, idx=0, offset=1, plot=True):
        """ Compare given horizon to labels in dataset.

        Parameters
        ----------
        horizon : dict
            Mapping from (iline, xline) to heights.

        idx : int
            Index of cube in the dataset to work with.

        offset : number
            Value to shift horizon up. Can be used to take into account different counting bases.

        plot : bool
            Whether to plot histogram of errors.
        """
        labels = self.labels[self.indices[idx]]
        sample_rate = self.geometries[self.indices[idx]].sample_rate

        compare_horizons(horizon, labels, printer=print, plot=plot,
                         sample_rate=sample_rate, offset=offset)
        return self


    def show_slide(self, idx=0, iline=0, overlap=True, **kwargs):
        """ Show full slide of the given cube on the given iline. """
        components = ('images', 'masks') if list(self.labels.values())[0] else ('images',)
        plot_slide(self, *components, idx=idx, iline=iline, overlap=overlap, **kwargs)

    def subset_labels(self, points, crop_shape=(2, 64, 64), cube_index=0, show_prior_mask=False):
        """Save prior mask to a cubeset attribute `prior_mask`.
        Parameters
        ----------
        points : tuple or list
            upper left coordinates of the starting crop in the seismic cube coordinates.
        crop_shape : tuple or list
            shape of the saved prior mask.
        cube_index : int
            index of the cube in `ds.indices` list.
        show_prior_mask : bool
            whether to show prior mask
        """
        print('hey you')
        ds_points = np.array([[self.indices[cube_index], *points, None]])[:, :4]

        start_predict_pipeline = (Pipeline()
                                  .load_component(src=[D('geometries'), D('labels')],
                                                  dst=['geometries', 'labels'])
                                  .crop(points=ds_points, shape=crop_shape)
                                  .load_cubes(dst='images')
                                  .create_masks(dst='masks', width=1, n_horizons=1, src_labels='labels')
                                  .rotate_axes(src=['images', 'masks'])
                                  .add_axis(src='masks', dst='masks')
                                  .scale(mode='normalize', src='images')) << self

        batch = start_predict_pipeline.next_batch(len(self.indices), n_epochs=None)

        if show_prior_mask:
            plt.imshow(batch.masks[0][:, :, 0, 0].T)
            plt.show()

        i_shift, x_shift, h_shift = [slices[0] for slices in batch.slices[0]]
        transforms = [lambda i_: self.geometries[self.indices[cube_index]].ilines[i_ + i_shift],
                      lambda x_: self.geometries[self.indices[cube_index]].xlines[x_ + x_shift],
                      lambda h_: h_ + h_shift]
        self.get_point_cloud(np.moveaxis(batch.masks[0][:, :, :, 0], -1, 0),
                             threshold=0.5, dst='prior_mask', coordinates=None, transforms=transforms, separate=True)
        if len(self.prior_mask[0]) == 0:
            raise ValueError("Prior mask is empty")
        numba_horizon = convert_to_numba_dict(self.prior_mask[0])
        setattr(self, 'prior_mask', {self.indices[cube_index]: numba_horizon})
        return self

    def make_slice_prediction(self, model_pipeline, points, crop_shape, max_iters=10, width=10, stride=32,
                              cube_index=0, threshold=0.02, show_count=None, slide_direction='xline', mode='right'):
        # pylint: disable=too-many-branches
        # pylint: disable=too-many-statements
        """ Extend horizon on one slice by sequential predict on overlapping crops.

        Parameters
        ----------
        ds : Cubeset
            Instance of the Cubeset. Must have non-empy attributes `predicted labels` and `labels` (for debugging plots)
        points : tuple or list
            upper left coordinates of the starting crop in the seismic cube coordinates.
        crop_shape : tuple or list
            shape of the crop fed to the model.
        max_iters : int
            max_number of extension steps. If we meet end of the cube we will make less steps.
        width : int
            width of compared windows.
        stride : int
            stride size.
        cube_index : int
            index of the cube ds.indices.
        threshold : float
            threshold for predicted mask
        show_count : int
            Number of extension steps to show
        slide_direction : str
            Either `xline` or `iline`. Direction of the predicted slice.
        mode : str
            if left increase next point's line coordinates otherwise decrease it.

        Returns
        -------
        SeismicCubeset
            Same instance with updated `predicted_labels` attribute.
            grid_info : dict
                grid info based on the grid array with upper left coordinates of the crops
        """
        show_count = max_iters if show_count is None else show_count
        geom = self.geometries[self.indices[cube_index]]
        grid_array = []
        if isinstance(points[0], (list, tuple)):
            max_iline = points[0][1] if points[0][1] is not None else geom.ilines_len
            max_xline = points[1][1] if points[1][1] is not None else geom.xlines_len
            points = [points[0][0], points[1][0], points[2][0]]
        else:
            max_iline, max_xline = geom.ilines_len, geom.xlines_len

        # compute strides for xline, iline cases
        line_stride = -stride if mode == 'left' else stride
        if slide_direction == 'iline':
            axes = (1, 0, 2)
            strides_candidates = [[line_stride, 0, -stride], [line_stride, 0, stride], [line_stride, 0, 0],
                                  [0, 0, -stride], [0, 0, stride]]
        elif slide_direction == 'xline':
            axes = (0, 1, 2)
            strides_candidates = [[0, line_stride, -stride], [0, line_stride, stride], [0, line_stride, 0],
                                  [0, 0, -stride], [0, 0, stride]]
        else:
            raise ValueError("Slide direction can be either iline or xline.")

        load_components_ppl = (Pipeline()
                               .load_component(src=[D('geometries'), D('labels')],
                                               dst=['geometries', 'labels'])
                               .add_components('predicted_labels'))

        predict_ppl = (Pipeline()
                       .load_component(src=[D('predicted_labels')], dst=['predicted_labels'])
                       .load_cubes(dst='images')
                       .create_masks(dst='masks', width=1, n_horizons=1, src_labels='labels')
                       .create_masks(dst='cut_masks', width=1, n_horizons=1, src_labels='predicted_labels')
                       .apply_transform(np.transpose, axes=axes, src=['images', 'masks', 'cut_masks'])
                       .rotate_axes(src=['images', 'masks', 'cut_masks'])
                       .analytic_transform(src='images', dst='hilbert')
                       .scale(mode='normalize', src='images')
                       .add_axis(src='masks', dst='masks')
                       .import_model('extension', model_pipeline)
                       .init_variable('result_preds', default=list())
                       .concat_components(src=('images', 'cut_masks'), dst='model_inputs')
                       .predict_model('extension', fetches='sigmoid',
                                      images=B('model_inputs'),
                                      save_to=V('result_preds', mode='e')))

        for i in range(max_iters):
            if (points[0] + crop_shape[0] > max_iline or
                    points[1] + crop_shape[1] > max_xline or points[2] + crop_shape[2] > geom.depth):
                print("End of the cube or area")
                break

            grid_array.append(points)
            ds_points = np.array([[self.indices[cube_index], *points, None]])[:, :4]
            crop_ppl = Pipeline().crop(points=ds_points, shape=crop_shape, passdown='predicted_labels')

            next_predict_pipeline = (load_components_ppl + crop_ppl + predict_ppl) << self
            btch = next_predict_pipeline.next_batch(len(self.indices), n_epochs=None)
            result = next_predict_pipeline.get_variable('result_preds')[0][..., 0]
            if np.sum(btch.images) < 1e-2:
                print('Empty traces')
                break

            # transform cube coordinates to ilines-xlines
            transforms = [lambda i_: self.geometries[self.indices[cube_index]].ilines[i_ + points[0]],
                          lambda x_: self.geometries[self.indices[cube_index]].xlines[x_ + points[1]],
                          lambda h_: h_ + points[2]]

            if slide_direction == 'iline':
                self.get_point_cloud(np.moveaxis(result, -1, 1), threshold=threshold, dst='predicted_mask',
                                     coordinates=None, separate=True, transforms=transforms)
            else:
                self.get_point_cloud(np.moveaxis(result, -1, 0), threshold=threshold, dst='predicted_mask',
                                     coordinates=None, separate=True, transforms=transforms)
            try:
                numba_horizons = convert_to_numba_dict(self.predicted_mask[0])
            except IndexError:
                print('Empty predicted mask on step %s' % i)
                plt.imshow(btch.cut_masks[0][:, :, 0].T)
                plt.show()
                break

            assembled_horizon_dict = update_horizon_dict(self.predicted_labels[self.indices[cube_index]],
                                                         numba_horizons)
            setattr(self, 'predicted_labels', {self.indices[cube_index]: assembled_horizon_dict})
            points, compared_slices_ = compute_next_points(points, result[:, :, 0].T,
                                                           crop_shape, strides_candidates, width)

            if i < show_count:
                print('----------------')
                print(i)
                print('argmax ', np.argmax(np.array(compared_slices_)))
                print('next stride ', strides_candidates[np.argmax(np.array(compared_slices_))])
                print('selected next points ', points)
                plot_extension_history(next_predict_pipeline, btch)

            if len(self.predicted_labels) == 0:
                break

        # assemble grid_info
        self.grid_info = {self.indices[cube_index]:
                          make_grid_info(grid_array, self.indices[cube_index], crop_shape)}
        return self

    def update_labels(self, src='predicted_labels', update_src='prior_mask', cube_index=0):
        """ Update dict-like component with another dict
        Parameters
        ----------
        src : str
            Component to be updated.
        update_src : str
            Component with a dict to add.
        """
        dict_update = getattr(self, update_src)[self.indices[cube_index]]
        if hasattr(self, src):
            dict_update = update_horizon_dict(dict_update, getattr(self, src)[self.indices[cube_index]])
        setattr(self, src, {self.indices[cube_index]: dict_update})
        return self

    def show_saved_horizon(self, points, shape=None, cube_index=0, width=1, show_image=True):
        """ Show saved horizon on a slice from `predicted_labels` attribute.

        Parameters
        ----------
        points : tuple or list
            Upper left coordinates of the starting crop in the seismic cube coordinates.
        shape : tuple or list, optional
            Shape of the crop to be shown
            if None, then `predict_shape` from the `grid_info` attribute will be used.
        cube_index : int
            Index of the cube ds.indices.
        width : int
            Width of the horizon
        show_image : bool
            Whether to show initial seismic image on a separate plot.
        """
        points = np.array([[self.indices[cube_index], *points, None]])[:, :4]

        if not shape:
            shape = getattr(self, 'grid_info')[self.indices[cube_index]]['predict_shape']

        load_components_ppl = (Pipeline()
                               .load_component(src=[D('geometries'), D('labels')],
                                               dst=['geometries', 'labels'])
                               .add_components('predicted_labels')
                               .crop(points=points, shape=shape, passdown='predicted_labels')
                               .load_component(src=[D('predicted_labels')],
                                               dst=['predicted_labels'])
                               .load_cubes(dst='data_crops')
                               .create_masks(dst='mask_crops', width=width, n_horizons=3, src_labels='labels')
                               .create_masks(dst='cut_mask_crops', width=width, n_horizons=1,
                                             src_labels='predicted_labels'))

        batch = (load_components_ppl << self).next_batch(len(self.indices))
        if show_image:
            plt.figure(figsize=(30, 20))
            plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
            plt.show()

        plt.figure(figsize=(30, 20))
        plt.imshow(batch.mask_crops[0][0].T, cmap="Greens")
        plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
        plt.title('True mask', fontsize=20)
        plt.show()

        plt.figure(figsize=(30, 20))
        plt.imshow(batch.cut_mask_crops[0][0].T, cmap="Blues")
        plt.imshow(batch.data_crops[0][0].T, cmap='gray', alpha=0.5)
        plt.title('Predicted mask', fontsize=20)
        plt.show()
        setattr(self, 'img', batch.data_crops[0][0].T)
        return self

    def show_depth_map(self, idx=0, labels_idx=0, labels_src=None, _return=False):
        """ Show depth map of a horizon.

        Parameters
        ----------
        idx : int
            Number of cube to use.

        labels_idx : int
            Index of used horizon from `labels` dictionary.

        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.
        """
        labels = labels_src or self.labels[self.indices[idx]]
        geom = self.geometries[self.indices[idx]]
        hor_name = os.path.basename(geom.horizon_list[labels_idx])

        depth_map = labels_to_depth_map(labels, geom, labels_idx)
        depth_map[depth_map == FILL_VALUE_A] = 0

        plot_from_above(depth_map, 'Heights {} on cube {}'.format(hor_name, self.indices[idx]), cmap='seismic')
        print('Average value of height is {}'.format(np.mean(depth_map[depth_map != FILL_VALUE_A])))

        if _return:
            return depth_map
        return None


    def show_cube_values(self, idx=0, labels_idx=0, labels_src=None, scale=False, _return=False):
        """ Show trace values on a horizon.

        Parameters
        ----------
        idx : int
            Number of cube to use.

        labels_idx : int
            Index of used horizon from `labels` dictionary.

        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.

        scale : bool, callable
            If bool, then values are scaled to [0, 1] range.
            If callable, then it is applied to iline-oriented slices of data from the cube.
        """
        labels = labels_src or self.labels[self.indices[idx]]
        geom = self.geometries[self.indices[idx]]
        hor_name = os.path.basename(geom.horizon_list[labels_idx])

        data, depth_map = get_cube_values(labels, geom, labels_idx, 1, scale=scale)

        plot_from_above(data, 'Horizon {} on cube {}'.format(hor_name, self.indices[idx]), cmap='seismic')
        print('Average value of height is {}'.format(np.mean(depth_map[depth_map != FILL_VALUE_A])))
        print('Std of amplitudes is {}'.format(np.std(data[depth_map != FILL_VALUE_A])))

        if _return:
            return data
        return None


    def show_rgb(self, idx=0, labels_idx=0, labels_src=None, width=1):
        """ Show trace values on the horizon and surfaces directly under it.

        Parameters
        ----------
        idx : int
            Number of cube to use.

        labels_idx : int
            Index of used horizon from `labels` dictionary.

        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.

        width : int
            Space between surfaces to cut.
        """
        labels = labels_src or self.labels[self.indices[idx]]
        geom = self.geometries[self.indices[idx]]
        hor_name = os.path.basename(geom.horizon_list[labels_idx])

        data, depth_map = get_cube_values(labels, geom, labels_idx, 1 + width*2, width)
        data = data[:, :, (0, width, -1)]
        data -= data.min(axis=(0, 1)).reshape(1, 1, -1)
        data *= 1 / data.max(axis=(0, 1)).reshape(1, 1, -1)
        data[depth_map == FILL_VALUE_A, :] = 0
        data = data[:, :, ::-1]
        data *= np.asarray([1, 0.5, 0.25]).reshape(1, 1, -1)

        plot_from_above_rgb(data, 'RGB horizon {} on cube {}'.format(hor_name, self.indices[idx]))
        print('AVG', np.mean(depth_map[depth_map != FILL_VALUE_A]))


    def compute_corrs(self, idx=0, labels_idx=0, labels_src=None, window=3, _return=False):
        """ Compute correlations with the nearest traces along the horizon.

        Parameters
        ----------
        idx : int
            Number of cube to use.

        labels_idx : int
            Index of used horizon from `labels` dictionary.

        labels_src : dict, optional
            If None, then horizon is taken from `labels` attribute.
            If dict, then must be a horizon.

        window : int
            Width of trace used for computing correlations.
        """
        labels = labels_src or self.labels[self.indices[idx]]
        geom = self.geometries[self.indices[idx]]
        hor_name = os.path.basename(geom.horizon_list[labels_idx])

        data, depth_map = get_cube_values(labels, geom, labels_idx, window, 0)
        corrs = compute_corrs(data)
        corrs[np.where(depth_map == FILL_VALUE_A)] = 0

        plot_from_above(corrs, 'Correlation for {} on cube {}'.format(hor_name, self.indices[idx]), cmap='seismic')
        print('Average correlation is {}'.format(np.mean(corrs[depth_map != FILL_VALUE_A])))

        if _return:
            return corrs
        return None


    def compare_horizons(self, hor_1, hor_2, hor_1_idx=0, hor_2_idx=0, idx=0, axis=-1, cmap='Set1', _return=False):
        """ Compare two horizons on l1 metric and on derivative differences.

        Parameters
        ----------
        idx : int
            Number of cube to use.

        hor_1, hor_2 : dict
            Dictionaries with labeled horizons.

        hor_1_idx, hor_2_idx : int
            Index of used horizon from each respective dictionary.

        axis : int
            Axis to take derivative along.

        cmap : str
            Colormap of showing the results.
        """
        geom = self.geometries[self.indices[idx]]

        # Create all depth maps
        depth_map_1 = labels_to_depth_map(hor_1, geom, hor_1_idx, 0)
        depth_map_2 = labels_to_depth_map(hor_2, geom, hor_2_idx, 0)
        indicator = (np.minimum(depth_map_1, depth_map_2) == FILL_VALUE_A).astype(int)
        depth_map_1[np.where(indicator == 1)] = 0
        depth_map_2[np.where(indicator == 1)] = 0

        # l1: mean absolute difference between depth maps
        metric_1 = np.abs(depth_map_1 - depth_map_2)
        metric_1[np.where(indicator == 1)] = 0
        plot_from_above(metric_1, 'l1 metric on cube {}'.format(self.indices[idx]), cmap=cmap)
        print('Average value of l1 is {}\n\n'.format(np.mean(metric_1[np.where(indicator == 0)])))

        # ~: mean absolute difference between gradients of depth maps
        metric_2 = np.abs(np.diff(depth_map_1, axis=axis, prepend=0) - np.diff(depth_map_2, axis=axis, prepend=0))
        metric_2[np.where(indicator == 1)] = 0
        plot_from_above(metric_2, '~ metric on cube {}'.format(self.indices[idx]), cmap=cmap)
        print('Average value of ~ is {}'.format(np.mean(metric_2[np.where(indicator == 0)])))

        if _return:
            return metric_1, metric_2
        return None

    def make_expand_grid(self, cube_name, crop_shape, labels_img, labels_src='labels', labels_idx=0, stride=10,
                         batch_size=16):
        """ Define crops coordinates for one step of an extension step.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.

        labels_idx : int
            To be removed after adding different directional crop
        """
        borders_img = labels_img
        border_coords = np.where(borders_img == 1)
        il_min, il_max = np.min(border_coords[0]), np.max(border_coords[0])
        x_min, x_max = np.min(border_coords[1]), np.max(border_coords[1])
        il_offset = self.geometries[cube_name].ilines_offset
        xl_offset = self.geometries[cube_name].xlines_offset

        width, line_shape, height = crop_shape
        iline_crops = []

        # sample horizontal border points
        for xline in range(x_min, x_max, width):
            non_zero = np.where(borders_img[il_min:il_max, xline] == 1)[0]
            if len(non_zero) == 0:
                continue
            _lower_il, _upper_il = np.min(non_zero) + il_min, np.max(non_zero) + il_min
            try:
                _lower_h = getattr(self, labels_src)[cube_name][(_lower_il + il_offset,
                                                                 xline + xl_offset)][labels_idx] - height // 2
            except KeyError as k:
                print(_lower_il + il_offset, xline + xl_offset, k)
            _upper_h = getattr(self, labels_src)[cube_name][(_upper_il + il_offset,
                                                             xline + xl_offset)][labels_idx] - height // 2

            _lower_il = _lower_il + stride - line_shape
            _upper_il = _upper_il - stride

            if _lower_il >= 0:
                iline_crops.append([cube_name, _lower_il, xline, _lower_h])
            if _upper_il + line_shape <= self.geometries[cube_name].ilines_len:
                iline_crops.append([cube_name, _upper_il, xline, _upper_h])

        iline_crops = np.array(iline_crops, dtype=object)
        iline_crops_gen = (iline_crops[i:i+batch_size]
                           for i in range(0, len(iline_crops), batch_size))
        offsets = np.array([np.min(iline_crops[:, 1]),
                            np.min(iline_crops[:, 2]),
                            np.min(iline_crops[:, 3])])

        ilines_range = (np.min(iline_crops[:, 1]), np.max(iline_crops[:, 1]) + line_shape)
        xlines_range = (np.min(iline_crops[:, 2]), np.max(iline_crops[:, 2]) + width)
        h_range = (np.min(iline_crops[:, 3]), np.max(iline_crops[:, 3]) + height)
        grid_array = iline_crops[:, 1:].astype(int) - offsets

        predict_shape = (ilines_range[1] - ilines_range[0],
                         xlines_range[1] - xlines_range[0],
                         h_range[1] - h_range[0])
        self.iline_crops_gen = lambda: next(iline_crops_gen)
        self.iline_crops_iters = - (-len(iline_crops) // batch_size)
        self.iline_crops_info = {'grid_array': grid_array,
                                 'predict_shape': predict_shape,
                                 'range': [ilines_range, xlines_range, h_range],
                                 'crop_shape': (crop_shape[1], crop_shape[0], crop_shape[2]),
                                 'cube_name': cube_name}

        xline_crops = []

        # sample vertical border points
        for iline in range(il_min, il_max, width):
            non_zero = np.where(borders_img[iline, x_min:x_max] == 1)[0]
            if len(non_zero) == 0:
                continue
            _lower_xl, _upper_xl = np.min(non_zero) + x_min, np.max(non_zero) + x_min
            _lower_h = getattr(self, labels_src)[cube_name][(iline + il_offset,
                                                             _lower_xl + xl_offset)][labels_idx] - height // 2
            _upper_h = getattr(self, labels_src)[cube_name][(iline + il_offset,
                                                             _upper_xl + xl_offset)][labels_idx] - height // 2

            _lower_xl = _lower_xl + stride - line_shape
            _upper_xl = _upper_xl - stride

            if _lower_xl >= 0:
                xline_crops.append([cube_name, iline, _lower_xl, _lower_h])
            if _upper_xl + line_shape <= self.geometries[cube_name].xlines_len:
                xline_crops.append([cube_name, iline, _upper_xl, _upper_h])

        xline_crops = np.array(xline_crops, dtype=object)
        xline_crops_gen = (xline_crops[i:i+batch_size]
                           for i in range(0, len(xline_crops), batch_size))


        x_offsets = np.array([np.min(xline_crops[:, 1]),
                              np.min(xline_crops[:, 2]),
                              np.min(xline_crops[:, 3])])

        x_ilines_range = (np.min(xline_crops[:, 1]), np.max(xline_crops[:, 1]) + width)
        x_xlines_range = (np.min(xline_crops[:, 2]), np.max(xline_crops[:, 2]) + line_shape)
        x_h_range = (np.min(xline_crops[:, 3]), np.max(xline_crops[:, 3]) + height)

        x_predict_shape = (x_ilines_range[1] - x_ilines_range[0],
                           x_xlines_range[1] - x_xlines_range[0],
                           x_h_range[1] - x_h_range[0])

        x_grid_array = xline_crops[:, 1:].astype(int) - x_offsets

        self.xline_crops_gen = lambda: next(xline_crops_gen)
        self.xline_crops_iters = - (-len(xline_crops) // batch_size)
        self.xline_crops_info = {'grid_array': x_grid_array,
                                 'predict_shape': x_predict_shape,
                                 'range': [x_ilines_range, x_xlines_range, x_h_range],
                                 'crop_shape': crop_shape,
                                 'cube_name': cube_name}
        return self


    def _predict_direction(self, model_pipeline, dst, direction):
        """ Awesome docstring

        Parameters
        ----------
        """
        grid_info = getattr(self, direction + '_crops_info')
        n_iters = getattr(self, direction + '_crops_iters')
        order = (2, 0, 1) if direction == 'xline' else (0, 2, 1)
        axes = (0, 1, 2) if direction == 'xline' else (1, 0, 2)

        crops_gen_name = direction + '_crops_gen'
        cube_name = grid_info['cube_name']

        predict_ppl = create_predict_ppl(model_pipeline, crops_gen_name, grid_info['crop_shape'], axes=axes) << self

        for _ in range(n_iters):
            predict_ppl.next_batch(1, n_epochs=None)

        assemble_ppl = Pipeline().assemble_crops(src=predict_ppl.v("result_preds"), dst='assembled_pred',
                                                 grid_info=grid_info, order=order) << self
        btch = assemble_ppl.next_batch(1)
        points = [rng[0] for rng in grid_info['range']]
        transforms = [lambda i_: self.geometries[cube_name].ilines[i_ + points[0]],
                      lambda x_: self.geometries[cube_name].xlines[x_ + points[1]],
                      lambda h_: h_ + points[2]]

        self.get_point_cloud(btch.assembled_pred, threshold=0.001, dst=dst,
                             coordinates=None, separate=True, transforms=transforms)
        merged_dict = {**getattr(self, dst)[0], **getattr(self, dst)[1]}
        setattr(self, dst, {cube_name: convert_to_numba_dict(merged_dict)})

    def extension_step(self, cube_name, img, model_pipeline, crop_shape=(4, 100, 100),
                       labels_src='prior_mask', stride=10, batch_size=24):
        """ One step of extension algorithm in all directions.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.

        labels_idx : int
            To be removed after adding different directional crop
        """

        self.make_expand_grid(cube_name, crop_shape=crop_shape, batch_size=batch_size, stride=stride,
                              labels_img=img, labels_idx=0, labels_src=labels_src)
        self._predict_direction(model_pipeline, 'predicted_mask_x', 'xline')
        self._predict_direction(model_pipeline, 'predicted_mask_iline', 'iline')

        expanded_labels = {**self.predicted_mask_x[cube_name], **self.predicted_mask_iline[cube_name],
                           **self.prior_mask[cube_name]}
        setattr(self, 'prior_mask', {cube_name: convert_to_numba_dict(expanded_labels)})

    def expand_cycle(self, cube_name, crop_shape, path, return_image=False, prior_src='prior_mask', labels_src='prior_mask', n_iters=10, stride=10, batch_size=32):
            """ Awesome docstring

            Parameters
            ----------
            cube_name : str
                Reference to cube. Should be valid key for `geometries` attribute.

            labels_idx : int
                To be removed after adding different directional crop 
            """

            model_pipeline = (Pipeline().load_model('static', TFModel, 'extension', path=path) << self)
            model_pipeline.next_batch(1)

            for i in range(n_iters):
                img = self.show_labels(return_image=return_image, src=prior_src)
                self.extension_step(cube_name, img, model_pipeline, crop_shape=crop_shape, labels_src=labels_src, stride=stride, batch_size=batch_size)
            return self
