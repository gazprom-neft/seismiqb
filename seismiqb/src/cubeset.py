""" Container for storing seismic data and labels. """
#pylint: disable=too-many-lines, too-many-arguments
import os
from glob import glob
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from ..batchflow import FilesIndex, Dataset, Pipeline

from .geometry import SeismicGeometry
from .horizon import Horizon
from .plotters import plot_image, show_3d
from .crop_batch import SeismicCropBatch
from .utility_classes import IndexedDict


class SeismicCubeset(Dataset):
    """ A common container for data entities: usually, seismic cubes and some type of labels.
    Entities are stored in dict-like objects, which can be indexed with either cube names or ordinal integer.

    Can be initialized with either:
        - `FilesIndex` instance
        - path(s) to cubes
        - from an instance of Horizon

    For batch generation and pipeline usage, we use `:meth:~SeismicCropBatch.make_locations` to convert batch indices
    from individual cubes to crops.

    Attributes
    ----------
    geometries : IndexedDict
        Storage of geometries, where keys are cube names and values are `SeismicGeometry` instances.
    labels : IndexedDict
        Nested storage of labels, where keys are cube names and values are sequences of labels.
    """
    #pylint: disable=keyword-arg-before-vararg
    def __init__(self, index, batch_class=SeismicCropBatch, preloaded=None, *args, **kwargs):
        # Wrap with `FilesIndex`, if needed
        if not isinstance(index, FilesIndex):
            index = [index] if isinstance(index, str) else index
            index = FilesIndex(path=index, no_ext=True)
        super().__init__(index, batch_class=batch_class, preloaded=preloaded, *args, **kwargs)

        # Initialize basic containers
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


    # Inner workings
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

    def gen_batch(self, batch_size=None, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, **kwargs):
        """ Remove `n_epochs`, `shuffle` and `drop_last` from passed arguments.
        Set default value `batch_size` to the size of current dataset, removing the need to
        pass it to `next_batch` and `run` methods.
        """
        if (n_epochs is not None and n_epochs != 1) or shuffle or drop_last:
            raise TypeError(f'`SeismicCubeset` does not work with `n_epochs`, `shuffle` or `drop_last`!'
                            f'`{n_epochs}`, `{shuffle}`, `{drop_last}`')

        batch_size = batch_size or len(self)
        return super().gen_batch(batch_size, n_iters=n_iters, **kwargs)


    # Create and manage data attributes
    def load_geometries(self, logs=True, collect_stats=True, spatial=True, **kwargs):
        """ Load geometries into dataset attribute.

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

    def create_labels(self, paths=None, filter_zeros=True, dst='labels', labels_class=Horizon,
                      sort=True, bar=False, **kwargs):
        """ Create labels (horizons, facies, etc) from given paths.
        Optionally, sorts and filters loaded labels.

        Parameters
        ----------
        paths : dict
            Mapping from indices to txt paths with labels.
        filter_zeros : bool
            Whether to remove labels on zero-traces.
        dst : str
            Name of attribute to put labels in.
        labels_class : class
            Class to use for labels creation.
        sort : bool or str
            Whether to sort loaded labels. If True, then sort based on average depth.
            If string, then name of the attribute to use as sorting key.
        bar : bool
            Progress bar for labels loading. Defaults to False.

        Returns
        -------
        SeismicCubeset
            Same instance with loaded labels.
        """
        labels = IndexedDict({ix: [] for ix in self.indices})

        for idx in self.indices:
            pbar = tqdm(paths[idx], disable=(not bar))

            label_list = []
            for path in pbar:
                if path.endswith('.dvc'):
                    continue
                pbar.set_description(os.path.basename(path))
                label_list.append(labels_class(path, geometry=self.geometries[idx], **kwargs))

            if sort:
                sort = sort if isinstance(sort, str) else 'h_mean'
                label_list.sort(key=lambda label: getattr(label, sort))
            if filter_zeros:
                _ = [label.filter() for label in label_list]

            labels[idx] = [label for label in label_list if len(label) > 0]

        setattr(self, dst, labels)
        self._cached_attributes.add(dst)


    def dump_labels(self, path, name='points', separate=True):
        """ Dump label points to file. """
        for idx, labels_list in self.labels.items():
            for label in labels_list:
                dirname = os.path.dirname(self.index.get_fullpath(idx))
                if path[0] == '/':
                    path = path[1:]
                dirname = os.path.join(dirname, path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                name = label.name if separate else name
                save_to = os.path.join(dirname, name + '.npz')
                label.dump_points(save_to)

    def reset_caches(self, attributes=None):
        """ Reset lru cache for cached class attributes.

        Parameters
        ----------
        attributes : sequence of str
            Class attributes to reset cache in.
            If not supplied, reset in `geometries` and attributes added by `create_labels`.
        """
        cached_attributes = attributes or self._cached_attributes

        for attr in cached_attributes:
            for idx in self.indices:
                cached_attr = getattr(self, attr)[idx]
                cached_attr = cached_attr if isinstance(cached_attr, list) else [cached_attr]
                _ = [item.reset_cache() for item in cached_attr]


    # Default pipeline and batch for fast testing / introspection
    def data_pipeline(self, sampler, batch_size=4):
        """ Pipeline with default actions of creating locations, loading seismic images and corresponding masks. """
        return (self.p
                .make_locations(generator=sampler, batch_size=batch_size)
                .create_masks(dst='masks', width=4)
                .load_cubes(dst='images')
                .adaptive_reshape(src=['images', 'masks'])
                .normalize(src='images'))

    def data_batch(self, sampler, batch_size=4):
        """ Get one batch of `:meth:.data_pipeline` with `images` and `masks`. """
        return self.data_pipeline(sampler=sampler, batch_size=batch_size).next_batch()


    # Textual and visual representation of dataset contents
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

    def show_slide(self, loc, idx=0, axis='iline', zoom_slice=None, src_labels='labels', **kwargs):
        """ Show slide of the given cube on the given line.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int or str
            Number or name of axis to load slide along.
        zoom_slice : tuple of slices
            Tuple of slices to apply directly to 2d images.
        idx : str, int
            Number of cube in the index to use.
        src_labels : str
            Dataset components to show as labels.
        """
        components = ('images', 'masks') if getattr(self, src_labels)[idx] else ('images',)
        cube_name = self.indices[idx]
        geometry = self.geometries[cube_name]
        crop_shape = np.array(geometry.cube_shape)

        axis = geometry.parse_axis(axis)
        crop_shape[axis] = 1

        location = np.zeros((1, 9), dtype=np.int32)
        location[0, axis + 3] = loc
        location[0, axis + 6] = loc
        location[0, [6, 7, 8]] += crop_shape

        # Fake generator with one point only
        generator = lambda batch_size: location
        generator.to_names = lambda array: np.array([[cube_name, 'unknown']])

        pipeline = (Pipeline()
                    .make_locations(generator=generator)
                    .load_cubes(dst='images', src_labels=src_labels)
                    .normalize(src='images'))

        if 'masks' in components:
            use_labels = kwargs.pop('use_labels', 'all')
            width = kwargs.pop('width', crop_shape[-1] // 100)
            labels_pipeline = (Pipeline()
                               .create_masks(src_labels=src_labels, dst='masks', width=width, use_labels=use_labels))

            pipeline = pipeline + labels_pipeline

        batch = (pipeline << self).next_batch()
        imgs = [np.squeeze(getattr(batch, comp)) for comp in components]
        xmin, xmax, ymin, ymax = 0, imgs[0].shape[0], imgs[0].shape[1], 0

        if zoom_slice:
            imgs = [img[zoom_slice] for img in imgs]
            xmin = zoom_slice[0].start or xmin
            xmax = zoom_slice[0].stop or xmax
            ymin = zoom_slice[1].stop or ymin
            ymax = zoom_slice[1].start or ymax

        # Plotting defaults
        header = geometry.axis_names[axis]
        total = geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = geometry.index_headers[0]
            ylabel = geometry.index_headers[1]

        kwargs = {
            'title_label': f'Data slice on cube `{geometry.displayed_name}`\n {header} {loc} out of {total}',
            'title_y': 1.01,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': False, # TODO: Make every horizon mask creation individual to allow their distinction while plot.
            **kwargs
        }
        return plot_image(imgs, **kwargs)


    # Predictions
    # TODO: no longer needed, remove
    def assemble_crops(self, crops, grid_info='grid_info', order=(0, 1, 2), fill_value=None):
        """ #TODO: no longer needed, remove. """
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

    def _compute_total_batches_in_all_chunks(self, idx, chunk_grid, chunk_shape, crop_shape, crop_stride, batch_size):
        """ #TODO: no longer needed, remove. """
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


    # Convenient loader
    def load(self, label_dir=None, filter_zeros=True, dst_labels='labels',
             labels_class=None, direction=None, **kwargs):
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
        direction : int or None
            Faults direction, 0 or 1. If None, will be infered automatically.
        """
        self.load_geometries(**kwargs)

        # Create suitable data structure for `create_labels`
        label_dir = label_dir or '/INPUTS/HORIZONS/RAW/*'

        paths_txt = {}
        for idx in self.indices:
            dir_path = '/'.join(self.index.get_fullpath(idx).split('/')[:-1])
            label_dir_ = label_dir if isinstance(label_dir, str) else label_dir[idx]
            dir_ = glob(dir_path + label_dir_)
            if len(dir_) == 0:
                warn("No labels in {}".format(dir_path))
            paths_txt[idx] = dir_

        self.create_labels(paths=paths_txt, filter_zeros=filter_zeros, dst=dst_labels,
                           labels_class=labels_class, direction=direction, **kwargs)
