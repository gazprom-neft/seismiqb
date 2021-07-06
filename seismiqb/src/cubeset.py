""" Container for storing seismic data and labels. """
#pylint: disable=too-many-lines, too-many-arguments
import os
from glob import glob
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from ..batchflow import FilesIndex, Dataset, Pipeline

from .geometry import SeismicGeometry
from .crop_batch import SeismicCropBatch

from .horizon import Horizon, UnstructuredHorizon
from .metrics import HorizonMetrics
from .plotters import plot_image, show_3d
from .utils import fill_defaults
from .utility_classes import IndexedDict


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
        return msg[:-1]


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


    def gen_batch(self, batch_size=None, shuffle=False, n_iters=None, n_epochs=None, drop_last=False,
                  bar=False, iter_params=None, **kwargs):
        """ Remove `n_epochs`, `shuffle` and `drop_last` from passed arguments. """
        #pylint: disable=blacklisted-name
        batch_size = batch_size or len(self)
        if (n_epochs is not None and n_epochs != 1) or shuffle or drop_last:
            raise TypeError(f'`SeismicCubeset` does not work with `n_epochs`, `shuffle` or `drop_last`!'
                            f'`{n_epochs}`, `{shuffle}`, `{drop_last}`')
        return super().gen_batch(batch_size, n_iters=n_iters, bar=bar, iter_params=iter_params, **kwargs)


    def load_geometries(self, logs=True, collect_stats=True, spatial=True, **kwargs):
        """ Load geometries into dataset-attribute.

        Parameters
        ----------
        logs : bool
            Whether to create logs. If True, .log file is created next to .sgy-cube location.
        collect_stats : bool
            Whether to collect stats for cubes in SEG-Y format.
        spatial : bool
            Whether to collect additional stats for POST-STACK cubes.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded geometries.
        """
        for ix in self.indices:
            self.geometries[ix].process(collect_stats=collect_stats, spatial=spatial, **kwargs)
            if logs:
                self.geometries[ix].log()


    def create_labels(self, paths=None, filter_zeros=True, dst='labels', labels_class=None,
                      sort=False, bar=False, **kwargs):
        """ Create labels (horizons, facies, etc) from given paths and optionaly sort them.

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
        sort : 'h_min', 'h_mean', 'h_max' or False
            Whether sort loaded labels by one of its attributes or not.
        bar : bool
            Progress bar for labels loading. Defaults to False.
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
            pbar = tqdm(paths[idx], disable=(not bar))
            label_list = []
            for path in pbar:
                if path.endswith('.dvc'):
                    continue
                pbar.set_description(os.path.basename(path))
                label_list += [labels_class(path, self.geometries[idx], **kwargs)]
            if sort:
                label_list.sort(key=lambda label: getattr(label, sort))
            if filter_zeros:
                _ = [getattr(item, 'filter')() for item in label_list]
            self[idx, dst] = [item for item in label_list if len(item.points) > 0]
            self._cached_attributes.add(dst)

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


    def dump_labels(self, path, fmt='npy', separate=False):
        """ Dump points to file. """
        for idx, labels_list in self.labels.items():
            for label in labels_list:
                dirname = os.path.dirname(self.index.get_fullpath(idx))
                if path[0] == '/':
                    path = path[1:]
                dirname = os.path.join(dirname, path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                name = label.name if separate else 'faults'
                save_to = os.path.join(dirname, name + '.' + fmt)
                label.dump_points(save_to, fmt)


    def show_3d(self, idx=0, src='labels', aspect_ratio=None, zoom_slice=None,
                 n_points=100, threshold=100, n_sticks=100, n_nodes=10,
                 slides=None, margin=(0, 0, 20), colors=None, **kwargs):
        """ Interactive 3D plot for some elements of cube. Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `n_sticks` and `n_nodes` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube if needed
        Parameters
        ----------
        idx : int, str
            Cube index.
        src : str, Horizon-instance or list
            Items to draw, by default, 'labels'. If item of list (or `src` itself) is str, then all items of
            that dataset attribute will be drawn.
        aspect_ratio : None, tuple of floats or Nones
            Aspect ratio for each axis. Each None in the resulting tuple will be replaced by item from
            `(geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)`.
        zoom_slice : tuple of slices or None
            Crop from cube to show. By default, the whole cube volume will be shown.
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger height differences in vertices.
        n_sticks : int
            Number of sticks for each fault.
        n_nodes : int
            Number of nodes for each stick.
        slides : list of tuples
            Each tuple is pair of location and axis to load slide from seismic cube.
        margin : tuple of ints
            Added margin for each axis, by default, (0, 0, 20).
        colors : dict or list
            Mapping of label class name to color defined as str, by default, all labels will be shown in green.
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
        coords = []
        simplices = []

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

        if isinstance(colors, dict):
            colors = [colors.get(type(label).__name__, colors.get('all', 'green')) for label in labels]

        simplices_colors = []
        for label, color in zip(labels, colors):
            x, y, z, simplices_ = label.make_triangulation(**triangulation_kwargs)
            if x is not None:
                simplices += [simplices_ + sum([len(item) for item in coords])]
                simplices_colors += [[color] * len(simplices_)]
                coords += [np.stack([x, y, z], axis=1)]

        simplices = np.concatenate(simplices, axis=0)
        coords = np.concatenate(coords, axis=0)
        simplices_colors = np.concatenate(simplices_colors)
        title = geometry.displayed_name

        default_aspect_ratio = (geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (geometry.index_headers[0], geometry.index_headers[1], 'DEPTH')

        images = []
        if slides is not None:
            for loc, axis in slides:
                image = geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom_slice[1:]]
                elif axis == 1:
                    image = image[zoom_slice[0], zoom_slice[-1]]
                else:
                    image = image[zoom_slice[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom_slice, simplices_colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)

    def show_points(self, idx=0, src_labels='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.geometries[idx].cube_shape[:-1])
        denum = np.zeros(self.geometries[idx].cube_shape[:-1])
        for label in getattr(self, src_labels)[idx]:
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src_labels)[idx][0]).__name__
        kwargs = {
            'title_label': f'{labels_class} on {self.indices[idx]}',
            'xlabel': self.geometries[idx].index_headers[0],
            'ylabel': self.geometries[idx].index_headers[1],
            'cmap': 'Reds',
            **kwargs
        }
        return plot_image(map_, **kwargs)


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
        # TODO: move to `Horizon` class
        for idx in self.indices:
            if horizon.geometry.name == self.geometries[idx].name:
                horizons_to_compare = self[idx, src_labels]
                break
        HorizonMetrics([horizon, horizons_to_compare]).evaluate('compare', agg=None,
                                                                absolute=absolute, offset=offset,
                                                                printer=printer, hist=hist, plot=plot)


    def show_slide(self, loc, idx=0, axis='iline', zoom_slice=None,
                   n_ticks=5, delta_ticks=100, src_labels='labels', **kwargs):
        """ Show full slide of the given cube on the given line.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        src_labels : str
            Dataset components to show as labels.
        idx : str, int
            Number of cube in the index to use.
        backend : str
            Backend to use for render. Can be either 'plotly' or 'matplotlib'. Whenever
            using 'plotly', also use slices to make the rendering take less time.
        """
        components = ('images', 'masks') if getattr(self, src_labels)[idx] else ('images',)
        cube_name = self.indices[idx]
        geometry = self.geometries[cube_name]
        crop_shape = np.array(geometry.cube_shape)

        axis = geometry.parse_axis(axis)
        point = np.array([[cube_name, 0, 0, 0]], dtype=object)
        point[0, axis + 1] = loc
        crop_shape[axis] = 1

        pipeline = (Pipeline()
                    .make_locations(points=point, shape=crop_shape)
                    .load_cubes(dst='images', src_labels=src_labels)
                    .normalize(src='images'))

        if 'masks' in components:
            use_labels = kwargs.pop('use_labels', 'all')
            width = kwargs.pop('width', 5)
            labels_pipeline = (Pipeline()
                               .create_masks(src_labels=src_labels, dst='masks', width=width, use_labels=use_labels))

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
            'title_label': f'Data slice on cube `{geometry.displayed_name}`\n {header} {loc} out of {total}',
            'title_y': 1.01,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'xticks': tuple(xticks),
            'yticks': tuple(yticks),
            'legend': False, # TODO: Make every horizon mask creation individual to allow their distinction while plot.
            **kwargs
        }

        plot_image(imgs, **kwargs)
        return batch

    def assemble_crops(self, crops, grid_info='grid_info', order=(0, 1, 2), fill_value=None):
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

        if fill_value is None and len(crops) != 0:
            fill_value = np.min(crops)

        grid_array = grid_info['grid_array']
        crop_shape = grid_info['crop_shape']
        background = np.full(grid_info['predict_shape'], fill_value, dtype=crops[0].dtype)

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

            crop = crops[j]
            crop = np.transpose(crop, order)
            crop = crop[tuple(crop_slice)]
            previous = background[tuple(background_slice)]
            background[tuple(background_slice)] = np.maximum(crop, previous)

        return background

    def make_prediction(self, dst, pipeline, crop_shape, crop_stride, locations=None,
                        idx=0, src='predictions', chunk_shape=None, chunk_stride=None, batch_size=8,
                        agg='max', projection='ixh', threshold=0.5, pbar=True, order=(0, 1, 2)):
        """ Infer, assemble and dump predictions from pipeline.

        Parameters
        ----------
        dst : str or None
            Path to save predictions. If None, function returns `np.ndarray` with predictions.
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
        order : tuple of int
            Passed directly to :meth:`.assemble_crops`.
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
                                                            batch_size, src, pbar, order)

        return SeismicGeometry.create_file_from_iterable(predictions_generator, output_shape,
                                                         chunk_shape, chunk_stride, dst, agg, projection, threshold)

    def _predictions_generator(self, idx, pipeline, locations, output_shape, chunk_shape, chunk_stride,
                               crop_shape, crop_stride, batch_size, src, pbar, order):
        """ Apply inference pipeline to each chunk. Returns position of predictions and corresponding array. """
        geometry = self.geometries[idx]
        cube_shape = geometry.cube_shape

        chunk_grid = self._make_regular_grid(idx, chunk_shape, ilines=locations[0], xlines=locations[1],
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
            prediction = self.assemble_crops(chunk_pipeline.v(src), order=order)
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
        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros, dst=dst_labels,
                           labels_class=labels_class, **kwargs)
        self._p, self._bins = p, bins # stored for later sampler creation
