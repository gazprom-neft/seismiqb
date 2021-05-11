""" Container for storing seismic data and labels with facies-specific interaction model. """
from copy import copy
from warnings import warn
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from .facies import Facies
from .batch import FaciesBatch
from ..cubeset import SeismicCubeset
from ..utility_classes import IndexedDict
from ..utils import to_list



class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    """
    def __init__(self, *args, batch_class=FaciesBatch, **kwargs):
        super().__init__(*args, batch_class=batch_class, **kwargs)


    def load_labels(self, label_dir, labels_subdirs, linkage, dst_labels=None,
                    base_labels='horizons', add_subsets=True, **kwargs):
        """ Load corresponding labels into given dataset attributes.
        Optionally add secondary labels as subsets into base labels.
        Adress `Facies` docs for details on subsets implementation.

        Parameters
        ----------
        label_dir : str
            Path to folder with corresponding labels subfolders.
            Must be relative to loaded geometry location.
        labels_subdirs : sequence
            Path to folders containing corresponding labels.
            Must be relative to `labels_dir` folder.
        linkage : dict
            Correspondence between cube name and a list of patterns for its labels.
        dst_labels : sequence
            Names of dataset components to load corresponding labels into.
        base_labels : str
            Which dataset attribute assign to `self.labels`.
        add_subsets : bool
            Whether add corresponding labels as subset to base labels or not.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        Given following arguments:

        >>> label_dir = 'INPUTS/FACIES'
        >>> labels_subdirs = ['FANS_HORIZON', 'FANS']
        >>> linkage = {'CUBE_01_AAA' : ['horizon_1.char'], 'CUBE_01_BBB' : ['horizon_2.char']}
        >>> dst_labels = ['horizons', 'fans']
        >>> base_labels = 'horizons'

        Following actions will be performed:
        >>> load labels into `self.horizon` component from:
            - CUBE_01_AAA/INPUTS/FACIES/FANS_HORIZONS/horizon_1.char
            - CUBE_02_BBB/INPUTS/FACIES/FANS_HORIZONS/horizon_2.char
        >>> load labels into `self.fans` component from:
            - CUBE_01_AAA/INPUTS/FACIES/FANS/horizon_1.char
            - CUBE_02_BBB/INPUTS/FACIES/FANS/horizon_2.char
        >>> assign `self.horizons` to `self.labels`.
        """
        self.load_geometries()

        default_dst_labels = [labels_subdir.lower() for labels_subdir in labels_subdirs]
        dst_labels = to_list(dst_labels, default=default_dst_labels)

        if base_labels not in dst_labels:
            alt_base_labels = dst_labels[0]
            msg = f"""
            Provided `base_labels={base_labels}` are not in `dst_labels` and set automatically to `{alt_base_labels}`.
            That means, that dataset `labels` attribute will point to `{alt_base_labels}`.
            To override this behaviour provide `base_labels` from `dst_labels`.
            """
            warn(msg)
            base_labels = alt_base_labels

        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in linkage.items():
                cube_path = self.index.get_fullpath(cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_path = f"{cube_dir}/{label_dir}/{labels_subdir}/{label}"
                    paths[cube_name].append(label_path)
            self.create_labels(paths=paths, dst=dst_label, labels_class=Facies, **kwargs)
            if add_subsets and (dst_label != base_labels):
                self.add_subsets(subset_labels=dst_label, base_labels=base_labels)

        setattr(self, 'labels', getattr(self, base_labels))

    def add_subsets(self, subset_labels, base_labels='labels'):
        """ Add nested labels. """
        flatten_base_labels = self.flatten_labels(src_labels=base_labels)
        flatten_subset_labels = self.flatten_labels(src_labels=subset_labels)
        if len(flatten_base_labels) != len(flatten_subset_labels):
            raise ValueError(f"Labels `{subset_labels}` and `{base_labels}` have different lengths.")
        for base_label, subset_label in zip(flatten_base_labels, flatten_subset_labels):
            base_label.add_subset(subset_labels, subset_label)

    def flatten_labels(self, src_labels='labels', indices=None):
        """ Convert given labels attribute from `IndexedDict` to `list`. Optionally filter out required indices. """
        indices = to_list(indices, default=self.indices)
        indices = [idx if isinstance(idx, str) else self.indices[idx] for idx in indices]
        labels_lists = [self[idx, src_labels] for idx in indices]
        return sum(labels_lists, [])

    @property
    def flat_labels(self):
        """ Return flatten base labels. """
        return self.flatten_labels()

    def apply_to_labels(self, function, indices=None, src_labels='labels', **kwargs):
        """ Call specific function for labels attributes of specific cubes.

        function : str or callable
            If str, name of the function or method to call from the attribute.
            If callable, applied directly to each item of cubeset attribute from `attributes`.
        indices : sequence of str
            For the attributes of which cubes to call `function`.
        src_labels : sequence of str
            For what cube label to call `function`.
        kwargs :
            Passed directly to `function`.

        Examples
        --------
        >>> cubeset.apply_to_labels('smooth_out', ['CUBE_01_XXX', 'CUBE_02_YYY'], ['horizons', 'fans'}, iters=3])
        """
        src_labels = to_list(src_labels)
        results = {}
        for src in src_labels:
            results[src] = {}
            for label in self.flatten_labels(src_labels=src, indices=indices):
                res = function(label, **kwargs) if callable(function) else getattr(label, function)(**kwargs)
                results[src][label.short_name] = res
        return results

    def show(self, attributes=None, src_labels='labels', indices=None, linkage=None,
                    plot=True, save=None, return_figures=False, **figure_params):
        """ Show attributes of multiple dataset labels. """
        figures = self.apply_to_labels(function='show', src_labels=src_labels, indices=indices,
                                       attributes=attributes, linkage=linkage, plot=plot, save=save,
                                       return_figure=return_figures, **figure_params)
        return figures if return_figures else None

    def invert_subsets(self, subset, indices=None, src_labels='labels', dst_labels=None, add_subsets=True):
        """ Apply `invert_subset` for every given label and put it into cubeset. """
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.apply_to_labels(function='invert_subset', indices=indices, src_labels=src_labels, subset=subset)
        results = IndexedDict({idx: [] for idx in self.indices})
        for _, label in inverted[src_labels].items():
            results[label.cube_name].append(label)
        setattr(self, dst_labels, results)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, base_labels=src_labels)

    def add_merged_labels(self, src_labels, dst_labels, indices=None, add_subsets_to='labels'):
        """ Merge given labels and put result into cubeset. """
        results = IndexedDict({idx: [] for idx in self.indices})
        indices = to_list(indices, default=self.indices)
        for idx in indices:
            to_merge = self[idx, src_labels]
            # since `merge_list` merges all horizons into first object from the list,
            # make a copy of first horizon in list to save merge into its instance
            container = copy(to_merge[0])
            container.name = f"Merged {'/'.join([horizon.short_name for horizon in to_merge])}"
            _ = [container.adjacent_merge(horizon, inplace=True, mean_threshold=999, adjacency=999)
                 for horizon in to_merge]
            container.reset_cache()
            results[idx].append(container)
        setattr(self, dst_labels, results)
        if add_subsets_to:
            self.add_subsets(subset_labels=dst_labels, base_labels=add_subsets_to)

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
        labels = getattr(self, src_labels)[self.grid_info['cube_name']]
        if labels_indices is not None:
            labels_indices = [labels_indices] if isinstance(labels_indices, int) else labels_indices
            labels = [labels[i] for i in labels_indices]

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
            underlay = underlay[label.bbox]
            underlay = underlay.T
            ax.imshow(underlay, **attr_plot_dict)
            ax.set_title(f"Grid over `{attribute}` on `{label.short_name}`", fontsize=plot_dict['title_fontsize'])

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

    def make_predictions(self, pipeline, crop_shape, overlap_factor, order=(1, 2, 0), src_labels='labels',
                         dst_labels='predictions', add_subsets=True, pipeline_variable='predictions', bar='n'):
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
        """
        # pylint: disable=blacklisted-name
        results = IndexedDict({ix: [] for ix in self.indices})
        pbar = tqdm(self.flatten_labels(src_labels))
        pbar.set_description("General progress")
        for label in pbar:
            prediction = copy(label)
            prediction.name = label.name
            cube_name = label.geometry.short_name
            self.make_grid(cube_name=cube_name, crop_shape=crop_shape, overlap_factor=overlap_factor,
                           heights=int(label.h_mean), mode='2d')
            pipeline = pipeline << self
            pipeline.update_config({'src_labels': src_labels, 'base_horizon': label})
            pipeline.run(batch_size=self.size, n_iters=self.grid_iters, bar=bar)
            predicted_matrix = expit(self.assemble_crops(pipeline.v(pipeline_variable), order=order).squeeze())
            prediction.filter_matrix(~(predicted_matrix.round().astype(bool)))
            setattr(prediction, "probability_matrix", predicted_matrix)
            results[cube_name].append(prediction)
        setattr(self, dst_labels, results)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, base_labels=src_labels)

    def evaluate(self, true_src, pred_src, metrics_fn, output_format='df'):
        """ TODO """
        pd.options.display.float_format = '{:,.3f}'.format
        if output_format == 'dict':
            results = {}
            for idx in self.indices:
                results[idx] = []
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    result = metrics_fn(true_mask, pred_mask)
                    results[idx].append(result)
        elif output_format == 'df':
            columns = ['cube', 'horizon', 'metrics']
            rows = []
            for idx in self.indices:
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    metrics_value = metrics_fn(true_mask, pred_mask)
                    row = np.array([idx, true.short_name, metrics_value])
                    rows.append(row)
            data = np.stack(rows)
            results = pd.DataFrame(data=data, columns=columns)
            results['metrics'] = results['metrics'].values.astype(float)
        return results

    def dump_labels(self, path, src_labels, postfix=None, indices=None):
        """ TODO """
        postfix = src_labels if postfix is None else postfix
        timestamp = datetime.now().strftime('%b-%d_%H-%M-%S')
        path = f"{path}/{timestamp}_{postfix}/"
        self.apply_to_labels(function='dump', indices=indices, src_labels=src_labels, path=path)

    def make_grid(self, label, **kwargs):
        """ Create regular grid of points in cube.
        This method is usually used with :meth:`.assemble_predict`.

        Parameters
        ----------
        label : Facies
            label to make grid for
        kwargs : for `Horizon.make_grid`
        """
        # pylint: disable=too-many-statements
        height = int(label.h_mean) - kwargs['crop_shape'][2] // 2 # start for heights slices made by `crop` action
        kwargs['heights'] = (height, height + 1)
        super().make_grid(**kwargs)
        label.grid_info = self.grid_info
