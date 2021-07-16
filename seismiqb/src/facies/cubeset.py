""" Container for storing seismic data and labels with facies-specific interaction model. """
from copy import copy
from warnings import warn
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm.notebook import tqdm

from .facies import Facies
from .batch import FaciesBatch
from ..samplers import RegularGrid
from ..cubeset import SeismicCubeset
from ..utility_classes import IndexedDict, Accumulator3D
from ..utils import to_list



class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    """
    # pylint: disable=useless-super-delegation
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
        dst_labels = to_list(dst_labels or default_dst_labels)

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
        flat_base_labels = getattr(self, base_labels).flat
        flat_subset_labels = getattr(self, subset_labels).flat
        if len(flat_base_labels) != len(flat_subset_labels):
            raise ValueError(f"Labels `{subset_labels}` and `{base_labels}` have different lengths.")
        for base_label, subset_label in zip(flat_base_labels, flat_subset_labels):
            base_label.add_subset(subset_labels, subset_label)

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
            for label in getattr(self, src).flatten(keys=indices):
                res = function(label, **kwargs) if callable(function) else getattr(label, function)(**kwargs)
                results[src][label.short_name] = res
        return results

    def show(self, load='depths', src_labels='labels', indices=None, **kwargs):
        """ Show attributes of multiple dataset labels. """
        res = self.apply_to_labels(function='show', src_labels=src_labels, indices=indices, load=load, **kwargs)
        return res if kwargs.get('return_figure') else None

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
        indices = to_list(indices or self.indices)
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

    def make_predictions(self, pipeline, crop_shape, overlap_factor, aggregation='mean', src_labels='labels',
                         dst_labels='predictions', add_subsets=True, bar='n'):
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
        """
        # pylint: disable=blacklisted-name
        results = IndexedDict({ix: [] for ix in self.indices})
        labels = getattr(self, src_labels)
        pbar = tqdm(labels.flat)
        pbar.set_description("General progress")

        for label in pbar:
            label_id = np.where([item.name == label.name for item in labels[label.geometry.short_name]])[0][0]
            grid = RegularGrid(geometry=label.geometry, label_id=label_id, label_name=label.name,
                               ranges=[None, None, (0, 1)], crop_shape=crop_shape, overlap_factor=overlap_factor)
            accumulator = Accumulator3D.from_aggregation(aggregation=aggregation, origin=grid.origin,
                                                         shape=grid.shape, fill_value=None)

            pipeline = pipeline << self << {'grid': grid, 'accumulator': accumulator}
            pipeline.run(n_iters=grid.n_iters, bar=bar)

            predicted_matrix = expit(accumulator.data).squeeze()
            predicted_matrix[label.geometry.zero_traces] = 0
            filtering_matrix = np.invert(predicted_matrix > 0.5).astype(int)

            prediction = copy(label)
            prediction.name = label.name
            prediction.filter_matrix(filtering_matrix)

            setattr(prediction, "probability_matrix", predicted_matrix)
            results[label.geometry.short_name].append(prediction)

        setattr(self, dst_labels, results)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, base_labels=src_labels)

    def evaluate(self, src_true, src_pred, metrics_fn, indices=None, src_labels='labels'):
        """ TODO """
        metrics_values = self.apply_to_labels(function='evaluate', src_labels=src_labels, indices=indices,
                                              src_true=src_true, src_pred=src_pred, metrics_fn=metrics_fn)

        results = pd.concat(metrics_values[src_labels].values())

        return results

    def dump_labels(self, path, src_labels, postfix=None, indices=None):
        """ TODO """
        postfix = src_labels if postfix is None else postfix
        timestamp = datetime.now().strftime('%b-%d_%H-%M-%S')
        path = f"{path}/{timestamp}_{postfix}/"
        self.apply_to_labels(function='dump', indices=indices, src_labels=src_labels, path=path)