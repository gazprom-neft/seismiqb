""" Contains container for storing dataset of seismic crops. """
#pylint: disable=too-many-lines
import os
from glob import glob
from warnings import warn
from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm
from scipy.special import expit

from ..batchflow import FilesIndex, DatasetIndex, Dataset, Sampler, Pipeline
from ..batchflow import NumpySampler

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .horizon import Horizon, UnstructuredHorizon
from .hdf5_storage import StorageHDF5
from .metrics import HorizonMetrics
from .plotters import plot_image, show_3d
from .utils import IndexedDict, round_to_array, gen_crop_coordinates, make_axis_grid, fill_defaults, parse_axis



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
        self._cached_attributes = {'geometries'}


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


    def __getitem__(self, key):
        """ Select attribute or its item for specific cube.

        Examples
        --------
        Get `labels` attribute for cube with 0 index:
        >>> cubeset[0, 'labels']
        Get 2nd `channels` attribute item for cube with name 'CUBE_01_XXX':
        >>> cubeset['CUBE_01_XXX', 'channels', 2]
        """
        idx, attr, *item_num = key
        item_num = item_num[0] if len(item_num) == 1 else slice(None)
        return getattr(self, attr)[idx][item_num]


    def __setitem__(self, key, value):
        """ Set attribute or its item for specific cube.

        Examples
        --------
        Set `labels` attribute for cube with 0 index to `[label_0, label_1]`:
        >>> cubeset[0, 'labels'] = [label_0, label_1]
        Set 2nd item of `channels` attribute for cube with name 'CUBE_01_XXX' to `channel_0`:
        >>> cubeset['CUBE_01_XXX', 'channels', 2] = channel_0
        """
        idx, attr, *item_num = key
        item_num = item_num[0] if len(item_num) == 1 else slice(None)
        getattr(self, attr)[idx][item_num] = value


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
        filter_zeros : bool
            Whether to remove labels on zero-traces.
        dst : str
            Name of attribute to put labels in.
        labels_class : class
            Class to use for labels creation. If None, infer from `geometries`.
            Defaults to None.
        Returns
        -------
        SeismicCubeset
            Same instance with loaded labels.
        """
        if not hasattr(self, dst):
            setattr(self, dst, IndexedDict({ix: [] for ix in self.indices}))

        for idx in self.indices:
            if labels_class is None:
                if self.geometries[idx].structured:
                    labels_class = Horizon
                else:
                    labels_class = UnstructuredHorizon
            label_list = [labels_class(path, self.geometries[idx], **kwargs) for path in paths[idx]]
            label_list.sort(key=lambda label: label.h_mean)
            if filter_zeros:
                _ = [getattr(item, 'filter')() for item in label_list]
            self[idx, dst] = [item for item in label_list if len(item.points) > 0]
            self._cached_attributes.add(dst)


    def show_labels(self, indices=None, main_labels='labels', overlay_labels=None, attributes=None, correspondence=None,
                    scale=1, colorbar=True, main_cmap='tab20b', overlay_cmap='autumn', overlay_alpha=0.7,
                    suptitle_size=20, title_size=15, transpose=True):
        """ Show specific attributes for labels of selected cubes with optional overlay by other attributes.

        Parameters
        ----------
        indices : list of int, list of str or None
            Cubes indices to show labels for. If None, show labels for all cubes. Defaults to None.
        main_labels : str
            Name of cubeset attribute to get labels from for the main image.
        overlay_labels : str
            Name of cubeset attribute to get labels from for the overlay image.
        attributes : str or list of str
            Names of label attribute to show in a row (incompatible with `correspondence` arg).
        correspondence : list of dicts
            Alternative plotting specifications allowing nuanced vizualizations (incompatible with `attributes` arg).
            Each item of a list must be a dict defining label-attribute correspondence for specific subplot in a row.
            This dict should consist of `str` keys with the names of cubeset attributes to get labels for plotting from
            and `dict` values, specifying at least an attribute to plot for selected label. This allows plotting both
            non-overlayed and overlayed by mask images of specific label attribute, e.g.:
            >>> [
            >>>     {
            >>>         'labels' : dict(attribute='amplitudes', cmap='tab20c')
            >>>     },
            >>>     {
            >>>         'labels' : dict(attribute='amplitudes', cmap='tab20c'),
            >>>         'channels': dict(attribute='masks', cmap='Blues, alpha=0.5)
            >>>     }
            >>> ]
        scale : int
            How much to scale the figure.
        colorbar : bool
            Whether plot colorbar for every subplot.
            May be overriden for specific subplot by `colorbar` from `correspondence` arg if specified there.
        main_cmap : str
            Default name of colormap for main image.
            May be overriden for specific subplot by `cmap` from `correspondence` arg if specified there.
        overlay_cmap : str
            Default name of colormap for overlay image.
            May be overriden for specific subplot by `cmap` from `correspondence` arg if specified there.
        overlay_alpha : float from [0, 1]
            Default opacity for overlay image.
            May be overriden for specific subplot by `alpha` from `correspondence` arg if specified there.
        suptitle_size : int
            Fontsize of suptitle for a row of images.
        title_size : int
            Size of titles for every subplot in a row.
        transpose : bool
            Whether plot data in `plot_image` way or not.
        """
        #pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt
        from mpl_toolkits import axes_grid1

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot. """
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        indices = indices or self.indices
        if correspondence is None:
            attributes = 'heights' if attributes is None else attributes
            attributes = [attributes] if isinstance(attributes, str) else attributes
            correspondence = [{main_labels: dict(attribute=attribute)} for attribute in attributes]
        elif attributes is not None:
            raise ValueError("Can't use both `correspondence` and `attributes`.")

        for idx in indices:
            for label_num, label in enumerate(self[idx, main_labels]):
                x, y = label.cube_shape[:-1] if transpose else label.cube_shape[1::-1]
                ratio = np.divide(x, y)
                figaspect = np.array([ratio * len(correspondence), 1]) * scale * 10
                fig, axes = plt.subplots(ncols=len(correspondence), figsize=figaspect)
                axes = axes if isinstance(axes, np.ndarray) else [axes]
                suptitle_size = int(ratio * suptitle_size)
                title_size = int(title_size * ratio)
                fig.suptitle(f"`{label.name}` on `{idx}`", size=suptitle_size)
                main_label_bounds = tuple(slice(*lims) for lims in label.bbox[:2])
                for ax, src_params in zip(axes, correspondence):
                    if overlay_labels is not None and overlay_labels not in src_params:
                        src_params[overlay_labels] = dict(attribute='masks')
                    attributes = []
                    for layer_num, (src, params) in enumerate(src_params.items()):
                        attribute = params['attribute']
                        attributes.append(attribute)
                        data = self[idx, src, label_num].load_attribute(attribute, fill_value=np.nan)

                        if layer_num == 0:
                            alpha = 1
                            cmap = params.get('cmap', main_cmap)
                        else:
                            alpha = params.get('alpha', overlay_alpha)
                            cmap = params.get('cmap', overlay_cmap)

                        if len(data.shape) > 2 and data.shape[2] != 1:
                            data = data[..., data.shape[2] // 2 + 1]
                        data = data.squeeze()
                        data = data[main_label_bounds]
                        data = data.T if transpose else data

                        im = ax.imshow(data, cmap=cmap, alpha=alpha)
                        local_colorbar = params.get('colorbar', colorbar)
                        if local_colorbar and layer_num == 0:
                            add_colorbar(im)
                    ax.set_title(', '.join(np.unique(attributes)), size=title_size)


    def reset_caches(self, attrs=None):
        """ Reset lru cache for cached class attributes.

        Parameters
        ----------
        attrs : list or tuple of str
            Class attributes to reset cache in.
            If None, reset in `geometries` and attrs added by `create_labels`.
            Defaults to None.
        """
        cached_attributes = attrs or self._cached_attributes
        for idx in self.indices:
            for attr in cached_attributes:
                cached_attr = self[idx, attr]
                cached_attr = cached_attr if isinstance(cached_attr, list) else [cached_attr]
                _ = [item.reset_cache() for item in cached_attr]


    def dump_labels(self, path, fmt='npy', separate=False, projections='ixh'):
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
                label.dump_points(save_to, fmt, projections)


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
        batch = (self.p.make_locations(points=points, shape=shape, side_view=side_view,
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

    def show_3d(self, idx=0, src='labels', aspect_ratio=None, zoom_slice=None,
                n_points=100, threshold=100, n_sticks=100, n_nodes=10,
                projections=None, margin=(0, 0, 20), colors_mapping=None, **kwargs):
        """ Interactive 3D plot for some elements of cube. Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `n_sticks` and `n_nodes` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube
        Parameters
        ----------
        idx : int, str
            Cube index.
        src : str, Horizon-instance or list
            Items to draw. If item of list (or `src` itself) is str, then all items of that dataset attribute
            will be drawn, by default, 'labels'.
        aspect_ratio : tuple of floats.
            Aspect ratio for each axis.
        zoom_slice : tuple of slices
            Crop from cube to show.
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger height differences in vertices.
        n_sticks : int
            Number of sticks for each fault.
        n_nodes : int
            Number of nodes for each stick.
        projections : list of tuples
            Each tuple is pair of location and axis to load slide from seismic cube.
        margin : tuple of ints
            Added margin for each axis, by default, (0, 0, 20).
        show_axes : bool
            Whether to show axes and their labels.
        width, height : number
            Size of the image.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        src = src if isinstance(src, (tuple, list)) else [src]
        geometry = self.geometries[idx]
        colors_mapping = colors_mapping or {'all': 'green'}
        coords = []
        simplices = []
        colors = []

        if zoom_slice is None:
            zoom_slice = [slice(0, geometry.cube_shape[i]) for i in range(3)]
        else:
            zoom_slice = [
                slice(item.start or 0, item.stop or stop) for item, stop in zip(zoom_slice, geometry.cube_shape)
            ]
        zoom_slice = tuple(zoom_slice)
        triangulation_kwargs = {
            'n_points': n_points,
            'threshold': threshold,
            'n_sticks': n_sticks,
            'n_nodes': n_nodes,
            'slices': zoom_slice
        }

        labels = [getattr(self, src_)[idx] if isinstance(src_, str) else [src_] for src_ in src]
        labels = sum(labels, [])
        for label in labels:
            x, y, z, simplices_ = label.triangulation(**triangulation_kwargs)
            if x is not None:
                simplices += [simplices_ + sum([len(item) for item in coords])]
                color = colors_mapping.get(type(label).__name__, colors_mapping.get('all', 'green'))
                colors += [[color] * len(simplices_)]
                coords += [np.stack([x, y, z], axis=1)]

        simplices = np.concatenate(simplices, axis=0)
        coords = np.concatenate(coords, axis=0)
        colors = np.concatenate(colors)
        title = geometry.name

        default_aspect_ratio = (geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (geometry.index_headers[0], geometry.index_headers[1], 'DEPTH')

        images = []
        if projections is not None:
            for loc, axis in projections:
                image = geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom_slice[1:]]
                elif axis == 1:
                    image = image[zoom_slice[0], zoom_slice[-1]]
                else:
                    image = image[zoom_slice[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom_slice, colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)

    def show_points(self, idx=0, src_labels='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.geometries[idx].cube_shape[:-1])
        denum = np.zeros(self.geometries[idx].cube_shape[:-1])
        for label in self[idx, src_labels]:
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src_labels)[idx][0]).__name__
        kwargs = {
            'title': f'{labels_class} on {self.indices[idx]}',
            'xlabel': self.geometries[idx].index_headers[0],
            'ylabel': self.geometries[idx].index_headers[1],
            'cmap': 'Reds',
            **kwargs
        }
        plot_image(map_, **kwargs)


    def apply_to_attributes(self, function, indices, attributes, **kwargs):
        """ Call specific function for all attributes of specific cubes.

        function : str or callable
            If str, name of the function or method to call from the attribute.
            If callable, applied directly to each item of cubeset attribute from `attributes`.
        indices : sequence of str
            For the attributes of which cubes to call `function`.
        attributes : sequence of str
            For what cube attributes to call `function`.
        kwargs :
            Passed directly to `function`.

        Examples
        --------
        >>> cubeset.apply_to_attributes('smooth_out', ['CUBE_01_XXX', 'CUBE_02_YYY'], ['horizons', 'fans'}, iters=3])
        """
        for idx in indices:
            if idx not in self.indices:
                warn(f"Can't call `{function} for {attributes} of cube {idx}, since it is not in index.")
            else:
                for attribute in attributes:
                    for item in self[idx, attribute]:
                        res = function(item, **kwargs) if callable(function) else getattr(item, function)(**kwargs)
                        if res is not None:
                            warn(f"Call for {item} returned not None, which is not expected.")


    def make_grid(self, cube_name, crop_shape, ilines=None, xlines=None, heights=None, mode='3d',
                  strides=None, overlap=None, overlap_factor=None,
                  batch_size=16, filtering_matrix=None, filter_threshold=0):
        """ Create regular grid of points in cube.
        This method is usually used with :meth:`.assemble_predict`.

        Parameters
        ----------
        cube_name : str
            Reference to cube. Should be valid key for `geometries` attribute.
        crop_shape : sequence
            Shape of model inputs.
        ilines : sequence of two int
            Location of desired prediction, iline-wise.
            If None, whole cube ranges will be used.
        xlines : sequence of two int
            Location of desired prediction, xline-wise.
            If None, whole cube ranges will be used.
        heights : sequence of two int or a single number
            If sequence, location of desired prediction, depth-wise.
            If single number, a height to make grid along when `mode` is '2d'.
            In this case height will be corrected by half of crop height.
            If None, whole cube ranges will be used.
        mode : '3d' or '2d'
            Mode to generate grid coordinates.
            If '3d', in volume defined by `ilines`, `xlines`, `heights`.
            If '2d', on area defined by `ilines`, `xlines`.
            Defaults to '3d'.
        strides : float or sequence
            Distance between grid points.
        overlap : float or sequence
            Distance between grid points.
        overlap_factor : float or sequence
            Overlapping ratio of successive crops.
            Can be seen as `how many crops would cross every through point`.
            If both overlap and overlap_factor are provided,
            only overlap_factor will be used.
        batch_size : int
            Amount of returned points per generator call.
        filtering_matrix : ndarray
            Binary matrix of (ilines_len, xlines_len) shape with ones
            corresponding to areas that can be skipped in the grid.
            E.g., a matrix with zeros at places where a horizon is present
            and ones everywhere else.
            If None, geometry.zero_traces matrix will be used.
        filter_threshold : int or float in [0, 1]
            Exclusive lower bound for non-gap number of points (with 0's in the
            filtering_matrix) in a crop in the grid. Default value is 0.
            If float, proportion from the total number of traces in a crop will
            be computed.
        """
        if mode == '2d':
            if isinstance(heights, (int, float)):
                height = int(heights) - crop_shape[2] // 2 # start for heights slices made by `crop` action
                heights = (height, height + 1)
            else:
                raise ValueError("`heights` should be a single `int` value when `mode` is '2d'")
        elif mode != '3d':
            raise ValueError("`mode` can either be '3d' or '2d'.")
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
        #pylint: disable=too-many-branches
        ilines_grid, xlines_grid, heights_grid, grid = self._crop_positions(cube_name, crop_shape, ilines, xlines,
                                                                            heights, strides, overlap, overlap_factor,
                                                                            filtering_matrix, filter_threshold)
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

    def _crop_positions(self, cube_name, crop_shape, ilines=None, xlines=None, heights=None,
                       strides=None, overlap=None, overlap_factor=None,
                       filtering_matrix=None, filter_threshold=0):
        """ Create grid for each axis and array of crop positions. """
        geometry = self.geometries[cube_name]

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
        return ilines_grid, xlines_grid, heights_grid, np.array(grid, dtype=object)

    def show_grid(self, src_labels='labels', labels_indices=None, attribute='cube_values', plot_dict=None):
        """ Plot grid over selected surface to visualize how it overlaps data.

        Parameters
        ----------
        src_labels : str
            Labels to show below the grid.
            Defaults to `labels`.
        labels_indices : str
            Indices of items from `src_labels` to show below the grid.
        attribute : str
            Alias from :attr:`~Horizon.FUNC_BY_ATTR` to show below the grid.
        plot_dict : dict, optional
            Dict of plot parameters, such as:
                figsize : tuple
                    Size of resulted figure.
                title_fontsize : int
                    Font size of title over the figure.
                attr_* : any parameter for `plt.imshow`
                    Passed to attribute plotter
                grid_* : any parameter for `plt.hlines` and `plt.vlines`
                    Passed to grid plotter
                crop_* : any parameter for `plt.hlines` and `plt.vlines`
                    Passed to corners crops plotter
        """
        #pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt

        labels_indices = labels_indices if isinstance(labels_indices, (tuple, list)) else [labels_indices]
        labels_indices = slice(None) if labels_indices[0] is None else labels_indices
        labels = self[self.grid_info['cube_name'], src_labels, labels_indices]

        # Calculate grid lines coordinates
        (x_min, x_max), (y_min, y_max) = self.grid_info['range'][:2]
        x_stride, y_stride = self.grid_info['strides'][:2]
        x_crop, y_crop = self.grid_info['crop_shape'][:2]
        x_lines = list(np.arange(0, x_max, x_stride)) + [x_max - x_crop]
        y_lines = list(np.arange(0, y_max, y_stride)) + [y_max - y_crop]

        default_plot_dict = {
            'figsize': (20 * x_max // y_max, 10),
            'title_fontsize': 18,
            'attr_cmap' : 'tab20b',
            'grid_color': 'darkslategray',
            'grid_linestyle': 'dashed',
            'crop_color': 'crimson',
            'crop_linewidth': 3
        }
        plot_dict = default_plot_dict if plot_dict is None else {**default_plot_dict, **plot_dict}
        attr_plot_dict = {k.split('attr_')[-1]: v for k, v in plot_dict.items() if k.startswith('attr_')}
        attr_plot_dict['zorder'] = 0
        grid_plot_dict = {k.split('grid_')[-1]: v for k, v in plot_dict.items() if k.startswith('grid_')}
        grid_plot_dict['zorder'] = 1
        crop_plot_dict = {k.split('crop_')[-1]: v for k, v in plot_dict.items() if k.startswith('crop_')}
        crop_plot_dict['zorder'] = 2

        _fig, axes = plt.subplots(ncols=len(labels), figsize=plot_dict['figsize'])
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, label in zip(axes, labels):
            # Plot underlaying attribute
            underlay = label.load_attribute(attribute, transform={'fill_value': np.nan})
            if len(underlay.shape) == 3:
                underlay = underlay[:, :, underlay.shape[2] // 2].squeeze()
            underlay = underlay.T
            ax.imshow(underlay, **attr_plot_dict)
            ax.set_title("Grid over `{}` on `{}`".format(attribute, label.name), fontsize=plot_dict['title_fontsize'])

            # Set limits
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_max, y_min])

            # Plot grid
            ax.vlines(x_lines, y_min, y_max, **grid_plot_dict)
            ax.hlines(y_lines, x_min, x_max, **grid_plot_dict)

            # Plot first crop
            ax.vlines(x=x_lines[0] + x_crop, ymin=y_min, ymax=y_crop, **crop_plot_dict)
            ax.hlines(y=y_lines[0] + y_crop, xmin=x_min, xmax=x_crop, **crop_plot_dict)

            # Plot last crop
            ax.vlines(x=x_lines[-1], ymin=y_max - x_crop, ymax=y_max, **crop_plot_dict)
            ax.hlines(y=y_lines[-1], xmin=x_max - y_crop, xmax=x_max, **crop_plot_dict)


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
            setattr(self, dst, IndexedDict({ix: [] for ix in self.indices}))

        self[cube_name, dst] = horizons


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
                horizons_to_compare = self[idx, src_labels]
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

        axis = parse_axis(axis, geometry.index_headers)
        point = np.array([[cube_name, 0, 0, 0]], dtype=object)
        point[0, axis + 1] = loc
        crop_shape[axis] = 1

        pipeline = (Pipeline()
                    .make_locations(points=point, shape=crop_shape)
                    .load_cubes(dst='images')
                    .normalize(mode='q', src='images'))

        if 'masks' in components:
            use_labels = kwargs.pop('use_labels', -1)
            width = kwargs.pop('width', 5)
            labels_pipeline = (Pipeline()
                               .create_masks(dst='masks', width=width, use_labels=use_labels))

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
        horizon = self[cube_name, labels_src, 0] if isinstance(labels_src, str) else labels_src

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
            crop = crop[tuple(crop_slice)]
            previous = background[tuple(background_slice)]
            background[tuple(background_slice)] = np.maximum(crop, previous)

        return background

    def make_prediction(self, dst, pipeline, crop_shape, crop_stride, locations=None,
                        idx=0, src='predictions', chunk_shape=None, chunk_stride=None, batch_size=8,
                        agg='max', projection='ixh', threshold=0.5, pbar=True):
        """ Infer, assemble and dump predictions from pipeline.

        Parameters
        ----------
        dst : str or None
            Path to save predictions. If None, function returns`np.ndarray` with predictions.
        pipeline : Pipeline
            Pipeline for inference, `run_later` action must be provided.
        crop_shape : tuple
            Shape of crops. Must be the same as defined in pipeline. Is needed to create grid for each
            chunk of prediction.
        crop_stride : tuple or None
            Stride for crops, by default None (crop_stride is equal to crop_shape).
        locations : tuple of slices or None, optional
            Region of cube to infer, by default None. None means that prediction will be infered for the whole cube.
        idx : int, optional
            Index of the cube in dataset to infer, by default 0.
        src : str, optional
            Variable of pipeline which stores predictions, by default 'predictions'.
        chunk_shape : tuple or None, optional
            Shape of chunk to split initial cube, by default None. Pipeline will be executed chunk-wise,
            then prediction will be aggregated and stored to `'dst'`. None means that chunk has shape of
            the whole cube.
        chunk_stride : tuple or None, optional
            Stride for crops, by default None (chunk_stride is equal to chunk_shape).
        batch_size : int, optional
            Batch size for `make_grid`, by default 8
        agg : str, optional
            Aggregation for chunks, by default 'max'
        projection : str, optional
            Projections to create in hdf5 file, by default 'ixh'
        threshold : float, optional
            Threshold to transform predictions to 'points' format, by default 0.5
        pbar : bool, optional
            Progress bar, by default True
        """
        cube_shape = self.geometries[idx].cube_shape

        if locations is None:
            locations = [(0, s) for s in cube_shape]
        else:
            locations = [(item.start or 0, item.stop or stop) for item, stop in zip(locations, cube_shape)]
        locations = np.array(locations)
        output_shape = locations[:, 1] - locations[:, 0]

        chunk_shape = fill_defaults(chunk_shape, output_shape)
        chunk_shape = np.minimum(np.array(chunk_shape), np.array(output_shape))
        chunk_stride = fill_defaults(chunk_stride, chunk_shape)

        predictions_generator = self._predictions_generator(idx, pipeline, locations, output_shape,
                                                            chunk_shape, chunk_stride, crop_shape, crop_stride,
                                                            batch_size, src, pbar)

        return StorageHDF5.create_file_from_iterable(predictions_generator, output_shape,
                                                     chunk_shape, chunk_stride, dst, agg, projection, threshold)

    def _predictions_generator(self, idx, pipeline, locations, output_shape, chunk_shape, chunk_stride,
                               crop_shape, crop_stride, batch_size, src, pbar):
        """ Apply inference pipeline to each chunk. Returns position of predictions and corresponding array. """
        geometry = self.geometries[idx]
        cube_shape = geometry.cube_shape

        chunk_grid = self._crop_positions(idx, chunk_shape, ilines=locations[0], xlines=locations[1],
                                          heights=locations[2], filtering_matrix=geometry.zero_traces,
                                          strides=chunk_stride)[-1][:, 1:]

        if pbar:
            total = self._compute_total_batches_in_all_chunks(idx, chunk_grid, chunk_shape,
                                                              crop_shape, crop_stride, batch_size)
            progress_bar = tqdm(total=total)

        for lower_bound in chunk_grid:
            upper_bound = np.minimum(lower_bound + chunk_shape, cube_shape)
            self.make_grid(
                self.indices[idx], crop_shape,
                *list(zip(lower_bound, upper_bound)),
                strides=crop_stride, batch_size=batch_size
            )
            chunk_pipeline = pipeline << self
            for _ in range(self.grid_iters):
                _ = chunk_pipeline.next_batch(len(self))
                if pbar:
                    progress_bar.update(1)
            prediction = self.assemble_crops(chunk_pipeline.v(src), order=(0, 1, 2))
            prediction = prediction[:output_shape[0], :output_shape[1], :output_shape[2]]
            position = lower_bound - np.array([locations[i][0] for i in range(3)])
            yield position, prediction
        if pbar:
            progress_bar.close()

    def add_geometries_targets(self, paths, dst='geom_targets'):
        """ Create targets from given cubes.

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

    def _compute_total_batches_in_all_chunks(self, idx, chunk_grid, chunk_shape, crop_shape, crop_stride, batch_size):
        """ Is needed to use progress bar in `make_prediction`. """
        total = 0
        for lower_bound in chunk_grid:
            upper_bound = np.minimum(lower_bound + chunk_shape, self.geometries[idx].cube_shape)
            self.make_grid(
                self.indices[idx], crop_shape,
                *list(zip(lower_bound, upper_bound)),
                strides=crop_stride, batch_size=batch_size
            )
            total += self.grid_iters
        return total


    def make_labels_prediction(self, pipeline, crop_shape, overlap_factor,
                               src_labels='horizons', dst_labels='predictions', bar='n',
                               pipeline_var='predictions', order=(1, 2, 0), binarize=True):
        """
        Make predictions and put them into dataset attribute.

        Parameters
        ----------
        pipeline : Pipeline
            Inference pipeline.
        crop_shape : sequence
            Passed directly to :meth:`.make_grid`.
        overlap_factor : float or sequence
            Passed directly to :meth:`.make_grid`.
        src_labels : str
            Name of dataset component with items to make grid for.
        dst_labels : str
            Name of dataset component to put predictions into.
        pipeline_var : str
            Name of pipeline variable to get predictions for assemble from.
        order : tuple of int
            Passed directly to :meth:`.assemble_crops`.
        binarize : bool
            Whether convert probability to class label or not.
        """
        # pylint: disable=blacklisted-name
        setattr(self, dst_labels, IndexedDict({ix: [] for ix in self.indices}))
        for idx, labels in getattr(self, src_labels).items():
            for label in labels:
                self.make_grid(cube_name=idx, crop_shape=crop_shape, overlap_factor=overlap_factor,
                               heights=int(label.h_mean), mode='2d')
                pipeline = pipeline << self
                pipeline.run(batch_size=self.size, n_iters=self.grid_iters, bar=bar)
                prediction = self.assemble_crops(pipeline.v(pipeline_var), order=(1, 2, 0)).squeeze()
                prediction = expit(prediction)
                prediction = prediction.round() if binarize else prediction
                prediction_name = "{}_predicted".format(label.name)
                self[idx, dst_labels] += [Horizon(prediction, label.geometry, prediction_name)]


    # Task-specific loaders

    def load(self, label_dir=None, filter_zeros=True, dst_labels='labels',
             labels_class=None, p=None, bins=None, **kwargs):
        """ Load everything: geometries, point clouds, labels, samplers.

        Parameters
        ----------
        label_dir : str
            Relative path from each cube to directory with labels.
        filter_zeros : bool
            Whether to remove labels on zero-traces.
        dst_labels : str
            Class attribute to put loaded data into.
        labels_class : class
            Class to use for labels creation.
            See details in :meth:`.create_labels`.
        p : sequence of numbers
            Proportions of different cubes in sampler.
        bins : TODO
        """
        _ = kwargs
        label_dir = label_dir or '/INPUTS/HORIZONS/RAW/*'

        paths_txt = {}
        for idx in self.indices:
            dir_path = '/'.join(self.index.get_fullpath(idx).split('/')[:-1])
            label_dir_ = label_dir if isinstance(label_dir, str) else label_dir[idx]
            dir_ = glob(dir_path + label_dir_)
            if len(dir_) == 0:
                warn("No labels in {}".format(dir_path))
            paths_txt[idx] = dir_
        self.load_geometries(**kwargs)
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros,
                           dst=dst_labels, labels_class=labels_class, **kwargs)
        self._p, self._bins = p, bins # stored for later sampler creation


    def load_corresponding_labels(self, info, dst_labels=None, main_labels=None, **kwargs):
        """ Load corresponding labels into corresponding dataset attributes.

        Parameters
        ----------
        correspondence : dict
            Correspondence between cube name and a list of patterns for its labels.
        labels_dirs : sequence
            Paths to folders to look corresponding labels with patterns from `correspondence` values for.
            Paths must be relative to cube location.
        dst_labels : sequence
            Names of dataset components to load corresponding labels into.
        main_labels : str
            Which dataset attribute assign to `self.labels`.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        The following argument values may be used to load for labels for 'CUBE_01_XXX':
        - from 'INPUTS/FACIES/FANS_HORIZONS/horizon_01_corrected.char' into `horizons` component;
        - from 'INPUTS/FACIES/FANS/fans_on_horizon_01_corrected_v8.char' into `fans` component,
        and assign `self.horizons` to `self.labels`.

        >>> correspondence = {'CUBE_01_XXX' : ['horizon_01']}
        >>> labels_dirs = ['INPUTS/FACIES/FANS_HORIZONS', 'INPUTS/FACIES/FANS']
        >>> dst_labels = ['horizons', 'fans']
        >>> main_labels = 'horizons'
        """
        self.load_geometries()

        label_dir = info['PATHS']["LABELS"]
        labels_subdirs = info['PATHS']["SUBDIRS"]
        dst_labels = dst_labels or [labels_subdir.lower() for labels_subdir in labels_subdirs]
        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in info['CUBES'].items():
                full_cube_name = f"amplitudes_{cube_name}"
                cube_path = self.index.get_fullpath(full_cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_mask = '/'.join([cube_dir, label_dir, labels_subdir, f"*{label}"])
                    label_path = glob(label_mask)
                    if len(label_path) > 1:
                        raise ValueError('Multiple files match pattern')
                    paths[full_cube_name].append(label_path[0])
            self.create_labels(paths=paths, dst=dst_label, labels_class=Horizon, **kwargs)
        if main_labels is None:
            main_labels = dst_labels[0]
            warn("""Cubeset `labels` point now to `{}`.
                    To suppress this warning, explicitly pass value for `main_labels`.""".format(main_labels))
        self.labels = getattr(self, main_labels)


class Modificator:
    """ Converts array to `object` dtype and prepends the `cube_name` column.
    Picklable, unlike inline lambda function.
    """
    def __init__(self, cube_name):
        self.cube_name = cube_name

    def __call__(self, points):
        points = points.astype(np.object)
        return np.concatenate([np.full((len(points), 1), self.cube_name), points], axis=1)
