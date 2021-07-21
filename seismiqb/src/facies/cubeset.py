""" Container for storing seismic data and labels with facies-specific interaction model. """
from copy import copy
from datetime import datetime

import pandas as pd

from .batch import FaciesBatch
from ..cubeset import SeismicCubeset
from ..utility_classes import IndexedDict
from ..utils import to_list



class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    """
    # pylint: disable=useless-super-delegation
    def __init__(self, *args, batch_class=FaciesBatch, **kwargs):
        super().__init__(*args, batch_class=batch_class, **kwargs)

    def add_subsets(self, subset_labels, base_labels='labels'):
        """ Add nested labels. """
        flat_base_labels = getattr(self, base_labels).flat
        flat_subset_labels = getattr(self, subset_labels).flat
        if len(flat_base_labels) != len(flat_subset_labels):
            raise ValueError(f"Labels `{subset_labels}` and `{base_labels}` have different lengths.")
        for base_label, subset_label in zip(flat_base_labels, flat_subset_labels):
            base_label.add_subset(subset_labels, subset_label)

    def map_labels(self, function, indices=None, src_labels='labels', **kwargs):
        """ Call function for every item from labels list of requested cubes and return produced results.

        Parameters
        ----------
        function : str or callable
            If str, name of label method to call.
            If callable, applied to labels of chosen cubes.
        indices : sequence of str
            Which cubes' labels to map.
        src_labels : str
            Attribute with labels to map.
        kwargs :
            Passed directly to `function`.

        Returns
        -------
        IndexedDict where keys are cubes names and values are lists of results obtained by applied map.
        If all lists in result values are empty, None is returned instead.

        Examples
        --------
        >>> cubeset.map_labels('smooth_out', ['CUBE_01_AAA', 'CUBE_02_BBB'], 'horizons', iters=3)
        """
        results = IndexedDict({idx: [] for idx in self.indices})
        for label in getattr(self, src_labels).flatten(keys=indices):
            if isinstance(function, str):
                res = getattr(label, function)(**kwargs)
            elif callable(function):
                res = function(label, **kwargs)
            if res is not None:
                results[label.geometry.short_name].append(res)
        return results if len(results.flat) > 0 else None

    def show(self, attributes='depths', src_labels='labels', indices=None, **kwargs):
        """ Show attributes of multiple dataset labels. """
        return self.map_labels(function='show', src_labels=src_labels, indices=indices, attributes=attributes, **kwargs)

    def invert_subsets(self, subset, src_labels='labels', dst_labels=None, add_subsets=True):
        """ Apply `invert_subset` for every given label and put it into cubeset. """
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.map_labels(function='invert_subset', indices=None, src_labels=src_labels, subset=subset)

        setattr(self, dst_labels, inverted)
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

    def evaluate(self, src_true, src_pred, metrics_fn, indices=None, src_labels='labels'):
        """ TODO """
        metrics_values = self.map_labels(function='evaluate', src_labels=src_labels, indices=indices,
                                              src_true=src_true, src_pred=src_pred, metrics_fn=metrics_fn)
        return pd.concat(metrics_values.flat)

    def dump_labels(self, path, src_labels, postfix=None, indices=None):
        """ TODO """
        postfix = src_labels if postfix is None else postfix
        timestamp = datetime.now().strftime('%b-%d_%H-%M-%S')
        path = f"{path}/{timestamp}_{postfix}/"
        self.map_labels(function='dump', indices=indices, src_labels=src_labels, path=path)
