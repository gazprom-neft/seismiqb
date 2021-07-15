""" Metrics for seismic objects: cubes and horizons. """
from copy import copy
from textwrap import dedent
from tqdm.auto import tqdm

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

import cv2

from ..batchflow.notifier import Notifier

from .horizon import Horizon
from .utility_classes import Accumulator
from .functional import to_device, from_device
from .functional import correlation, crosscorrelation, btch, kl, js, hellinger, tv, hilbert
from .functional import smooth_out, digitize, gridify, perturb, histo_reduce
from .plotters import plot_image



class BaseMetrics:
    """ Base class for seismic metrics.
    Child classes have to implement access to `data`, `probs`, `bad_traces` attributes.
    """
    # pylint: disable=attribute-defined-outside-init, blacklisted-name
    PLOT_DEFAULTS = {
        'cmap': 'Metric',
        'fill_color': 'black'
    }

    LOCAL_DEFAULTS = {
        'kernel_size': 3,
        'agg': 'nanmean',
        'device': 'gpu',
        'amortize': True,
    }

    SUPPORT_DEFAULTS = {
        'supports': 100,
        'safe_strip': 50,
        'agg': 'nanmean',
        'device': 'gpu',
        'amortize': True,
    }

    SMOOTHING_DEFAULTS = {
        'kernel_size': 21,
        'sigma': 10.0,
    }

    EPS = 0.00001


    def evaluate(self, metric, plot=False, plot_supports=False, enlarge=True, width=5, **kwargs):
        """ Calculate desired metric, apply aggregation, then plot resulting metric-map.
        To plot the results, set `plot` argument to True.

        Parameters
        ----------
        metric : str
            Name of metric to evaluate.
        enlarge : bool
            Whether to apply `:meth:.Horizon.enlarge_carcass_image` to the result.
        width : int
            Widening for the metric. Works only if `enlarge` set to True.
        plot : bool
            Whether to use `:func:.plot_image` to show the result.
        plot_supports : bool
            Whether to show support traces on resulting image. Works only if `plot` set to True.
        kwargs : dict
            Arguments to be passed in metric-calculation methods
            (see `:meth:.compute_local` and `:meth:.compute_support`),
            as well as plotting arguments (see `:func:.plot_image`).
        """
        if 'support' in metric:
            kwargs = {**self.SUPPORT_DEFAULTS, **kwargs}
        elif 'local' in metric:
            kwargs = {**self.LOCAL_DEFAULTS, **kwargs}

        self._last_evaluation = {**kwargs}
        metric_fn = getattr(self, metric)
        metric_val, plot_dict = metric_fn(**kwargs)

        if cp is not np and cp.cuda.is_available():
            # pylint: disable=protected-access
            cp._default_memory_pool.free_all_blocks()

        if hasattr(self, 'horizon') and self.horizon.is_carcass and enlarge:
            metric_val = self.horizon.enlarge_carcass_image(metric_val, width)

        if plot:
            plot_dict = {**self.PLOT_DEFAULTS, **plot_dict}
            figure = plot_image(metric_val, **plot_dict, return_figure=True)

            if 'support' in metric and plot_supports:
                support_coords = self._last_evaluation['support_coords']
                figure.axes[0].scatter(support_coords[:, 0],
                                       support_coords[:, 1], s=33, marker='.', c='blue')

            # Store for debug / introspection purposes
            self._last_evaluation['plot_dict'] = plot_dict
            self._last_evaluation['figure'] = figure
        return metric_val


    def compute_local(self, function, data, bad_traces, kernel_size=3,
                      normalize=True, agg='mean', amortize=False, axis=0, device='cpu', pbar=None):
        """ Compute metric in a local fashion, using `function` to compare nearest traces.
        Under the hood, each trace is compared against its nearest neighbours in a square window
        of `kernel_size` size. Results of comparisons are aggregated via `agg` function.

        Works on both `cpu` (via standard `NumPy`) and GPU (with the help of `cupy` library).
        The returned array is always on CPU.

        Parameters
        ----------
        function : callable
            Function to compare two arrays. Must have the following signature:
            `(array1, array2, std1, std2)`, where `std1` and `std2` are pre-computed standard deviations.
            In order to work properly on GPU, must be device-agnostic.
        data : ndarray
            3D array of data to evaluate on.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        kernel_size : int
            Window size for comparison traces.
        normalize : bool
            Whether the data should be zero-meaned before computing metric.
        agg : str
            Function to aggregate values for each trace. See :class:`.Accumulator` for details.
        amortize : bool
            Whether the aggregation should be sequential or by stacking all the matrices.
            See :class:`.Accumulator` for details.
        axis : int
            Axis to stack arrays on. See :class:`.Accumulator` for details.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        pbar : type or None
            Progress bar to use.
        """
        i_range, x_range = data.shape[:2]
        k = kernel_size // 2 + 1

        # Transfer to GPU, if needed
        data = to_device(data, device)
        bad_traces = to_device(bad_traces, device)
        xp = cp.get_array_module(data) if (cp is not np) else np

        # Compute data statistics
        data_stds = data.std(axis=-1)
        bad_traces[data_stds == 0.0] = 1
        if normalize:
            data_n = data - data.mean(axis=-1, keepdims=True)
        else:
            data_n = data

        # Pad everything
        padded_data = xp.pad(data_n, ((0, k), (k, k), (0, 0)), constant_values=xp.nan)
        padded_stds = xp.pad(data_stds, ((0, k), (k, k)), constant_values=0.0)
        padded_bad_traces = xp.pad(bad_traces, k, constant_values=1)

        # Compute metric by shifting arrays
        total = kernel_size * kernel_size - 1
        pbar = Notifier(pbar, total=total) if pbar else None

        accumulator = Accumulator(agg=agg, amortize=amortize, axis=axis, total=total)
        for i in range(k):
            for j in range(-k+1, k):
                # Comparison between (x, y) and (x+i, y+j) vectors is the same as comparison between (x+i, y+j)
                # and (x, y). So, we can compare (x, y) with (x+i, y+j) and save computed result twice:
                # matrix associated with vector (x, y) and matrix associated with (x+i, y+j) vector.
                if (i == 0) and (j <= 0):
                    continue
                shifted_data = padded_data[i:i+i_range, k+j:k+j+x_range]
                shifted_stds = padded_stds[i:i+i_range, k+j:k+j+x_range]
                shifted_bad_traces = padded_bad_traces[k+i:k+i+i_range, k+j:k+j+x_range]

                computed = function(data, shifted_data, data_stds, shifted_stds)
                # Using symmetry property:
                symmetric_bad_traces = padded_bad_traces[k-i:k-i+i_range, k-j:k-j+x_range]
                symmetric_computed = computed[:i_range-i, max(0, -j):min(x_range, x_range-j)]
                symmetric_computed = xp.pad(symmetric_computed,
                                            ((i, 0), (max(0, j), -min(0, j))),
                                            constant_values=xp.nan)

                computed[shifted_bad_traces == 1] = xp.nan
                symmetric_computed[symmetric_bad_traces == 1] = xp.nan
                accumulator.update(computed)
                accumulator.update(symmetric_computed)
                if pbar:
                    pbar.update(2)

        if pbar:
            pbar.close()

        result = accumulator.get(final=True)
        return from_device(result)

    def compute_support(self, function, data, bad_traces, supports, safe_strip=0,
                        normalize=True, agg='mean', amortize=False, axis=0, device='cpu', pbar=None):
        """ Compute metric in a support fashion, using `function` to compare all the traces
        against a set of (randomly chosen or supplied) reference ones.
        Results of comparisons are aggregated via `agg` function.

        Works on both `cpu` (via standard `NumPy`) and GPU (with the help of `cupy` library).
        The returned array is always on CPU.

        Parameters
        ----------
        function : callable
            Function to compare two arrays. Must have the following signature:
            `(array1, array2, std1, std2)`, where `std1` and `std2` are pre-computed standard deviations.
            In order to work properly on GPU, must be device-agnostic.
        data : ndarray
            3D array of data to evaluate on.
        bad_traces : ndarray
            2D matrix of traces where the metric should not be computed.
        supports : int or ndarray
            If int, then number of supports to generate randomly from non-bad traces.
            If ndarray, then should be of (N, 2) shape and contain coordinates of reference traces.
        normalize : bool
            Whether the data should be zero-meaned before computing metric.
        agg : str
            Function to aggregate values for each trace. See :class:`.Accumulator` for details.
        amortize : bool
            Whether the aggregation should be sequential or by stacking all the matrices.
            See :class:`.Accumulator` for details.
        axis : int
            Axis to stack arrays on. See :class:`.Accumulator` for details.
        device : str
            Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
        pbar : type or None
            Progress bar to use.
        """
        # Transfer to GPU, if needed
        data = to_device(data, device)
        bad_traces = to_device(bad_traces, device)
        xp = cp.get_array_module(data) if (cp is not np) else np

        # Compute data statistics
        data_stds = data.std(axis=-1)
        bad_traces[data_stds == 0.0] = 1
        if normalize:
            data_n = data - data.mean(axis=-1, keepdims=True)
        else:
            data_n = data

        # Generate support coordinates
        if isinstance(supports, int):
            if safe_strip:
                bad_traces_ = bad_traces.copy()
                bad_traces_[:, :safe_strip], bad_traces_[:, -safe_strip:] = 1, 1
                bad_traces_[:safe_strip, :], bad_traces_[-safe_strip:, :] = 1, 1
            else:
                bad_traces_ = bad_traces

            valid_traces = xp.where(bad_traces_ == 0)
            indices = xp.random.choice(len(valid_traces[0]), supports)
            support_coords = xp.asarray([valid_traces[0][indices], valid_traces[1][indices]]).T

        elif isinstance(supports, (tuple, list, np.ndarray)):
            support_coords = xp.asarray(supports)

        # Save for plot and introspection
        self._last_evaluation['support_coords'] = from_device(support_coords)

        # Generate support traces
        support_traces = data_n[support_coords[:, 0], support_coords[:, 1]]
        support_stds = data_stds[support_coords[:, 0], support_coords[:, 1]]

        # Compute metric
        pbar = Notifier(pbar, total=len(support_traces)) if pbar else None
        accumulator = Accumulator(agg=agg, amortize=amortize, axis=axis, total=len(support_traces))
        for i, _ in enumerate(support_traces):
            computed = function(data_n, support_traces[i], data_stds, support_stds[i])
            computed[bad_traces == 1] = xp.nan
            accumulator.update(computed)
            if pbar:
                pbar.update()
        if pbar:
            pbar.close()

        result = accumulator.get(final=True)
        return from_device(result)


    def local_corrs(self, kernel_size=3, normalize=True, agg='mean', amortize=False,
                    device='cpu', pbar=None, **kwargs):
        """ Compute correlation in a local fashion. """
        metric = self.compute_local(function=correlation, data=self.data, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local correlation, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': -1.0, 'zmax': 1.0,
            **kwargs
        }
        return metric, plot_dict

    def support_corrs(self, supports=100, safe_strip=0, normalize=True, agg='mean', amortize=False,
                      device='cpu', pbar=None, **kwargs):
        """ Compute correlation against reference traces. """
        metric = self.compute_support(function=correlation, data=self.data, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, device=device, amortize=amortize,
                                      pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support correlation with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': -1.0, 'zmax': 1.0,
            'colorbar': True,
            'bad_color': 'k',
            **kwargs
        }
        return metric, plot_dict


    def local_crosscorrs(self, kernel_size=3, normalize=False, agg='mean', amortize=False,
                         device='cpu', pbar=None, **kwargs):
        """ Compute cross-correlation in a local fashion. """
        metric = self.compute_local(function=crosscorrelation, data=self.data, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)
        zvalue = np.nanquantile(np.abs(metric), 0.98).astype(np.int32)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local cross-correlation, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'cmap': 'seismic_r',
            'zmin': -zvalue, 'zmax': zvalue,
            **kwargs
        }
        return metric, plot_dict

    def support_crosscorrs(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                           device='cpu', pbar=None, **kwargs):
        """ Compute cross-correlation against reference traces. """
        metric = self.compute_support(function=crosscorrelation, data=self.data, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize, device=device, pbar=pbar)
        zvalue = np.nanquantile(np.abs(metric), 0.98).astype(np.int32)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support cross-correlation with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'cmap': 'seismic_r',
            'zmin': -zvalue, 'zmax': zvalue,
            **kwargs
        }
        return metric, plot_dict


    def local_btch(self, kernel_size=3, normalize=False, agg='mean', amortize=False,
                   device='cpu', pbar=None, **kwargs):
        """ Compute Bhattacharyya divergence in a local fashion. """
        metric = self.compute_local(function=btch, data=self.probs, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local Bhattacharyya divergence, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': 0.0, 'zmax': 1.0,
            **kwargs
        }
        return metric, plot_dict

    def support_btch(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                     device='cpu', pbar=None, **kwargs):
        """ Compute Bhattacharyya divergence against reference traces. """
        metric = self.compute_support(function=btch, data=self.probs, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize, device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support Bhattacharyya divergence with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': 0.0, 'zmax': 1.0,
            **kwargs
        }
        return metric, plot_dict


    def local_kl(self, kernel_size=3, normalize=False, agg='mean', amortize=False,
                 device='cpu', pbar=None, **kwargs):
        """ Compute Kullback-Leibler divergence in a local fashion. """
        metric = self.compute_local(function=kl, data=self.probs, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local KL divergence, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict

    def support_kl(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                   device='cpu', pbar=None, **kwargs):
        """ Compute Kullback-Leibler divergence against reference traces. """
        metric = self.compute_support(function=kl, data=self.probs, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize,
                                      device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support KL divergence with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict


    def local_js(self, kernel_size=3, normalize=False, agg='mean', amortize=False, device='cpu', pbar=None, **kwargs):
        """ Compute Jensen-Shannon divergence in a local fashion. """
        metric = self.compute_local(function=js, data=self.probs, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local JS divergence, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict

    def support_js(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                   device='cpu', pbar=None, **kwargs):
        """ Compute Jensen-Shannon divergence against reference traces. """
        metric = self.compute_support(function=js, data=self.probs, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize,
                                      device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support JS divergence with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict


    def local_hellinger(self, kernel_size=3, normalize=False, agg='mean', amortize=False,
                        device='cpu', pbar=None, **kwargs):
        """ Compute Hellinger distance in a local fashion. """
        metric = self.compute_local(function=hellinger, data=self.probs, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local Hellinger distance, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict

    def support_hellinger(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                          device='cpu', pbar=None, **kwargs):
        """ Compute Hellinger distance against reference traces. """
        metric = self.compute_support(function=hellinger, data=self.probs, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize,
                                      device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support Hellinger distance with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict


    def local_tv(self, kernel_size=3, normalize=False, agg='mean', amortize=False, device='cpu', pbar=None, **kwargs):
        """ Compute total variation in a local fashion. """
        metric = self.compute_local(function=tv, data=self.probs, bad_traces=self.bad_traces,
                                    kernel_size=kernel_size, normalize=normalize, agg=agg, amortize=amortize,
                                    device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        title = f'Local total variation, k={kernel_size}, with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict

    def support_tv(self, supports=100, safe_strip=0, normalize=False, agg='mean', amortize=False,
                   device='cpu', pbar=None, **kwargs):
        """ Compute total variation against reference traces. """
        metric = self.compute_support(function=tv, data=self.probs, bad_traces=self.bad_traces,
                                      supports=supports, safe_strip=safe_strip,
                                      normalize=normalize, agg=agg, amortize=amortize,
                                      device=device, pbar=pbar)

        title, plot_defaults = self.get_plot_defaults()
        n_supports = supports if isinstance(supports, int) else len(supports)
        title = f'Support total variation with {n_supports} supports with `{agg}` aggregation\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'zmin': None, 'zmax': None,
            **kwargs
        }
        return metric, plot_dict


    def quality_map(self, quantiles, metric_names=None, computed_metrics=None,
                    agg='mean', amortize=False, axis=0, apply_smoothing=False,
                    smoothing_params=None, local_params=None, support_params=None, **kwargs):
        """ Create a quality map based on number of metrics.

        Parameters
        ----------
        quantiles : sequence of floats
            Quantiles for computing hardness thresholds. Must be in (0, 1) ranges.
        metric_names : sequence of str
            Which metrics to use to assess hardness of data.
        reduce_func : str
            Function to reduce multiple metrics into one spatial map.
        smoothing_params, local_params, support_params : dicts
            Additional parameters for smoothening, local metrics, support metrics.
        """
        _ = kwargs
        computed_metrics = computed_metrics or []
        smoothing_params = smoothing_params or self.SMOOTHING_DEFAULTS
        local_params = local_params or self.LOCAL_DEFAULTS
        support_params = support_params or self.SUPPORT_DEFAULTS

        smoothing_params = {**self.SMOOTHING_DEFAULTS, **smoothing_params, **kwargs}
        local_params = {**self.LOCAL_DEFAULTS, **local_params, **kwargs}
        support_params = {**self.SUPPORT_DEFAULTS, **support_params, **kwargs}

        if metric_names:
            for metric_name in metric_names:
                if 'local' in metric_name:
                    kwds = copy(local_params)
                elif 'support' in metric_name:
                    kwds = copy(support_params)

                metric = self.evaluate(metric_name, plot=False, **kwds)
                computed_metrics.append(metric)

        accumulator = Accumulator(agg=agg, amortize=amortize, axis=axis)
        for metric_matrix in computed_metrics:
            if apply_smoothing:
                metric_matrix = smooth_out(metric_matrix, **smoothing_params)
            digitized = digitize(metric_matrix, quantiles)
            accumulator.update(digitized)
        quality_map = accumulator.get(final=True)

        if apply_smoothing:
            quality_map = smooth_out(quality_map, **smoothing_params)

        title, plot_defaults = self.get_plot_defaults()
        plot_dict = {
            **plot_defaults,
            'title_label': f'Quality map for {title}',
            'cmap': 'Reds',
            'zmin': 0.0, 'zmax': np.nanmax(quality_map),
            **kwargs
        }
        return quality_map, plot_dict

    def make_grid(self, quality_map, frequencies, iline=True, xline=True, full_lines=True, margin=0, **kwargs):
        """ Create grid with various frequencies based on quality map. """
        _ = kwargs
        if margin:
            bad_traces = np.copy(self.geometry.zero_traces)
            bad_traces[:, 0] = 1
            bad_traces[:, -1] = 1
            bad_traces[0, :] = 1
            bad_traces[-1, :] = 1

            kernel = np.ones((2 + 2*margin, 2 + 2*margin), dtype=np.uint8)
            bad_traces = cv2.dilate(bad_traces.astype(np.uint8), kernel, iterations=1).astype(bad_traces.dtype)
            quality_map[(bad_traces - self.geometry.zero_traces) == 1] = 0.0

        pre_grid = np.rint(quality_map)
        grid = gridify(pre_grid, frequencies, iline, xline, full_lines)

        if margin:
            grid[(bad_traces - self.geometry.zero_traces) == 1] = 0
        return grid



class HorizonMetrics(BaseMetrics):
    """ Evaluate metric(s) on horizon(s).
    During initialization, data along the horizon is cut with the desired parameters.
    To get the value of a particular metric, use :meth:`.evaluate`::
        HorizonMetrics(horizon).evaluate('support_corrs', supports=20, agg='mean')

    To plot the results, set `plot` argument of :meth:`.evaluate` to True.

    Parameters
    horizons : :class:`.Horizon` or sequence of :class:`.Horizon`
        Horizon(s) to evaluate.
        Can be either one horizon, then this horizon is evaluated on its own,
        or sequence of two horizons, then they are compared against each other,
        or nested sequence of horizon and list of horizons, then the first horizon is compared against the
        best match from the list.
    other parameters
        Passed direcly to :meth:`.Horizon.get_cube_values` or :meth:`.Horizon.get_cube_values_line`.
    """
    AVAILABLE_METRICS = [
        'local_corrs', 'support_corrs',
        'local_btch', 'support_btch',
        'local_kl', 'support_kl',
        'local_js', 'support_js',
        'local_hellinger', 'support_hellinger',
        'local_tv', 'support_tv',
        'instantaneous_phase',
    ]

    def __init__(self, horizons, window=23, offset=0, normalize=False, chunk_size=256):
        super().__init__()
        horizons = list(horizons) if isinstance(horizons, tuple) else horizons
        horizons = horizons if isinstance(horizons, list) else [horizons]
        self.horizons = horizons

        # Save parameters for later evaluation
        self.window, self.offset, self.normalize, self.chunk_size = window, offset, normalize, chunk_size

        # The first horizon is used to evaluate metrics
        self.horizon = horizons[0]
        self.name = self.horizon.short_name

        # Properties
        self._data = None
        self._probs = None
        self._bad_traces = None


    def get_plot_defaults(self):
        """ Axis labels and horizon/cube names in the title. """
        title = f'horizon `{self.name}` on cube `{self.horizon.geometry.displayed_name}`'
        return title, {
            'xlabel': self.horizon.geometry.axis_names[0],
            'ylabel': self.horizon.geometry.axis_names[1],
        }

    @property
    def data(self):
        """ Create `data` attribute at the first time of evaluation. """
        if self._data is None:
            self._data = self.horizon.get_cube_values(window=self.window, offset=self.offset,
                                                      normalize=self.normalize, chunk_size=self.chunk_size)
            self._data[self._data == Horizon.FILL_VALUE] = np.nan
        return self._data

    @property
    def probs(self):
        """ Probabilistic interpretation of `data`. """
        if self._probs is None:
            hist_matrix = histo_reduce(self.data, self.horizon.geometry.bins)
            self._probs = hist_matrix / np.sum(hist_matrix, axis=-1, keepdims=True) + self.EPS
        return self._probs

    @property
    def bad_traces(self):
        """ Traces to fill with `nan` values. """
        if self._bad_traces is None:
            self._bad_traces = self.horizon.geometry.zero_traces.copy()
            self._bad_traces[self.horizon.full_matrix == Horizon.FILL_VALUE] = 1
        return self._bad_traces


    def perturbed(self, n=5, scale=2.0, clip=3, window=None, kernel_size=3, agg='nanmean', device='cpu', **kwargs):
        """ Evaluate horizon by:
            - compute the `local_corrs` metric
            - perturb the horizon `n` times by random shifts, generated from normal
            distribution of `scale` std and clipping of `clip` size
            - compute the `local_corrs` metric for each of the perturbed horizons
            - get a mean and max value of those metrics: they correspond to the `averagely shifted` and
            `best generated shifts` horizons
            - use difference between horizon metric and mean/max metrics of perturbed as a final assesment maps

        Parameters
        ----------
        n : int
            Number of perturbed horizons to generate.
        scale : number
            Standard deviation (spread or “width”) of the distribution. Must be non-negative.
        clip : number
            Maximum size of allowed shifts
        window : int or None
            Size of the data along the height axis to evaluate perturbed horizons.
            Note that due to shifts, it must be smaller than the original data by atleast 2 * `clip` units.
        kernel_size, agg, device
            Parameters of individual metric evaluation
        """
        w = self.data.shape[2]
        window = window or w - 2 * clip - 1

        # Compute metrics for multiple perturbed horizons: generate shifts, apply them to data,
        # evaluate metric on the produced array
        acc_mean = Accumulator('nanmean')
        acc_max = Accumulator('nanmax')
        for _ in range(n):
            shifts = np.random.normal(scale=2., size=self.data.shape[:2])
            shifts = np.rint(shifts).astype(np.int32)
            shifts = np.clip(shifts, -clip, clip)

            shifts[self.horizon.full_matrix == self.horizon.FILL_VALUE] = 0
            pb = perturb(self.data, shifts, window)

            pb_metric = self.compute_local(function=correlation, data=pb, bad_traces=self.bad_traces,
                                           kernel_size=kernel_size, normalize=True, agg=agg, device=device)
            acc_mean.update(pb_metric)
            acc_max.update(pb_metric)

        pb_mean = acc_mean.get(final=True)
        pb_max = acc_max.get(final=True)

        # Subtract mean/max maps from the horizon metric
        horizon_metric = self.compute_local(function=correlation, data=self.data, bad_traces=self.bad_traces,
                                            kernel_size=kernel_size, normalize=True, agg=agg, device=device)
        diff_mean = horizon_metric - pb_mean
        diff_max = horizon_metric - pb_max

        title, plot_defaults = self.get_plot_defaults()
        title = f'Perturbed metrics\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'figsize': (20, 7),
            'separate': True,
            'suptitle_label': title,
            'title_label': ['mean', 'max'],
            'cmap': 'Reds_r',
            'zmin': [0.0, -0.5], 'zmax': 0.5,
            **kwargs
        }
        return (diff_mean, diff_max), plot_dict


    def instantaneous_phase(self, device='cpu', **kwargs):
        """ Compute instantaneous phase via Hilbert transform. """
        #pylint: disable=unexpected-keyword-arg
        # Transfer to GPU, if needed
        data = to_device(self.data, device)
        xp = cp.get_array_module(data) if (cp is not np) else np

        # Compute hilbert transform and scale to 2pi range
        analytic = hilbert(data, axis=2)

        phase = xp.angle(analytic)

        phase_slice = phase[:, :, phase.shape[-1] // 2]
        phase_slice[np.isnan(xp.std(data, axis=-1))] = xp.nan
        phase_slice[self.horizon.full_matrix == self.horizon.FILL_VALUE] = xp.nan

        # Evaluate mode value
        values = phase_slice[~xp.isnan(phase_slice)].round(2)
        uniques, counts = xp.unique(values, return_counts=True)
        # 3rd most frequent value is chosen to skip the first two (they are highly likely -pi/2 and pi/2)
        mode = uniques[xp.argpartition(counts, -3)[-3]]

        shifted_slice = phase_slice - mode
        shifted_slice[shifted_slice >= xp.pi] -= 2 * xp.pi

        # Re-norm so that mode value is at zero point
        if xp.nanmin(shifted_slice) < -xp.pi:
            shifted_slice[shifted_slice < -xp.pi] += 2 * xp.pi
        if xp.nanmax(shifted_slice) > xp.pi:
            shifted_slice[shifted_slice > xp.pi] -= 2 * xp.pi

        title, plot_defaults = self.get_plot_defaults()
        title = f'Instantaneous phase\nfor {title}'
        plot_dict = {
            **plot_defaults,
            'title_label': title,
            'cmap': 'seismic',
            'zmin': -np.pi, 'zmax': np.pi,
            'colorbar': True,
            'bad_color': 'k',
            **kwargs
        }
        return from_device(shifted_slice), plot_dict


    def find_best_match(self, offset=0, **kwargs):
        """ Find the closest horizon to the first one in the list of passed at initialization. """
        _ = kwargs
        if isinstance(self.horizons[1], Horizon):
            self.horizons[1] = [self.horizons[1]]

        lst = []
        for hor in self.horizons[1]:
            if hor.geometry.name == self.horizon.geometry.name:
                overlap_info = Horizon.check_proximity(self.horizon, hor, offset=offset)
                lst.append((hor, overlap_info))
        lst.sort(key=lambda x: abs(x[1].get('mean', 999999)))
        other, overlap_info = lst[0]
        return (other, overlap_info), {} # actual return + fake plot dict


    def compare(self, offset=0, absolute=True, hist=True, printer=print, **kwargs):
        """ Compare horizons on against the best match from the list of horizons.

        Parameters
        ----------
        offset : number
            Value to shift horizon down. Can be used to take into account different counting bases.
        absolute : bool
            Whether to use absolute values for differences.
        hist : bool
            Whether to plot histogram of differences.
        printer : callable
            Function to print results, for example `print` or any other callable that can log data.
        """
        if len(self.horizons) != 2:
            raise ValueError('Can compare two horizons exactly or one to the best match from list of horizons. ')
        _ = kwargs
        (other, oinfo), _ = self.find_best_match(offset=offset)

        self_full_matrix = self.horizon.full_matrix
        other_full_matrix = other.full_matrix
        metric = np.where((self_full_matrix != other.FILL_VALUE) & (other_full_matrix != other.FILL_VALUE),
                          offset + self_full_matrix - other_full_matrix, np.nan)
        if absolute:
            metric = np.abs(metric)

        at_1 = len(np.asarray((self_full_matrix != other.FILL_VALUE) &
                              (other_full_matrix == other.FILL_VALUE)).nonzero()[0])
        at_2 = len(np.asarray((self_full_matrix == other.FILL_VALUE) &
                              (other_full_matrix != other.FILL_VALUE)).nonzero()[0])

        if printer is not None:
            msg = f"""
            Comparing horizons:
            {self.horizon.name.rjust(45)}
            {other.name.rjust(45)}
            {'—'*45}
            Rate in 5ms:                         {oinfo['window_rate']:8.3f}
            Mean/std of errors:               {oinfo['mean']:4.2f} / {oinfo['std']:4.2f}
            Mean/std of abs errors:           {oinfo['abs_mean']:4.2f} / {oinfo['abs_std']:4.2f}
            Max error/abs error:              {oinfo['max']:4} / {oinfo['abs_max']:4}
            {'—'*45}
            Lengths of horizons:                 {len(self.horizon):8}
                                                 {len(other):8}
            {'—'*45}
            Average heights of horizons:         {(offset + self.horizon.h_mean):8.2f}
                                                 {other.h_mean:8.2f}
            {'—'*45}
            Coverage of horizons:                {self.horizon.coverage:8.4f}
                                                 {other.coverage:8.4f}
            {'—'*45}
            Solidity of horizons:                {self.horizon.solidity:8.4f}
                                                 {other.solidity:8.4f}
            {'—'*45}
            Number of holes in horizons:         {self.horizon.number_of_holes:8}
                                                 {other.number_of_holes:8}
            {'—'*45}
            Additional traces labeled:           {at_1:8}
            (present in one, absent in other)    {at_2:8}
            {'—'*45}
            """
            printer(dedent(msg))

        if hist:
            hist_dict = {
                'bins': 100,
                'xlabel': 'l1-values',
                'ylabel': 'N',
                'title_label': 'Histogram of l1 differences',
            }
            plot_image(metric, mode='hist', **hist_dict)

        title = 'Height differences between {} and {}'.format(self.horizon.name, other.name)
        plot_dict = {
            'spatial': True,
            'title_label': '{} on cube {}'.format(title, self.horizon.cube_name),
            'cmap': 'Reds',
            'zmin': 0, 'zmax': np.nanmax(metric),
            'ignore_value': np.nan,
            'xlabel': 'INLINE_3D', 'ylabel': 'CROSSLINE_3D',
            'bad_color': 'black',
            'colorbar': True,
            **kwargs
        }
        return metric, plot_dict




class GeometryMetrics(BaseMetrics):
    """ Metrics to asses cube quality. """
    AVAILABLE_METRICS = [
        'local_corrs', 'support_corrs',
        'local_btch', 'support_btch',
        'local_kl', 'support_kl',
        'local_js', 'support_js',
        'local_hellinger', 'support_hellinger',
        'local_tv', 'support_tv',
    ]


    def __init__(self, geometries):
        super().__init__()

        geometries = list(geometries) if isinstance(geometries, tuple) else geometries
        geometries = geometries if isinstance(geometries, list) else [geometries]
        self.geometries = geometries

        self.geometry = geometries[0]
        self._data = None
        self._probs = None
        self._bad_traces = None

        self.name = 'hist_matrix'

    def get_plot_defaults(self):
        """ Axis labels and horizon/cube names in the title. """
        title = f'`{self.name}` on cube `{self.geometry.displayed_name}`'
        return title, {
            'xlabel': self.geometry.axis_names[0],
            'ylabel': self.geometry.axis_names[1],
        }

    @property
    def data(self):
        """ Histogram of values for every trace in the cube. """
        if self._data is None:
            self._data = self.geometry.hist_matrix
        return self._data

    @property
    def bad_traces(self):
        """ Traces to exclude from metric evaluations: bad traces are marked with `1`s. """
        if self._bad_traces is None:
            self._bad_traces = self.geometry.zero_traces
            self._bad_traces[self.data.max(axis=-1) == self.data.sum(axis=-1)] = 1
        return self._bad_traces

    @property
    def probs(self):
        """ Probabilistic interpretation of `data`. """
        if self._probs is None:
            self._probs = self.data / np.sum(self.data, axis=-1, keepdims=True) + self.EPS
        return self._probs


    def tracewise(self, func, l=3, pbar=True, **kwargs):
        """ Apply `func` to compare two cubes tracewise. """
        pbar = tqdm if pbar else lambda iterator, *args, **kwargs: iterator
        metric = np.full((*self.geometry.lens, l), np.nan)

        indices = [geometry.dataframe['trace_index'] for geometry in self.geometries]

        for idx, _ in pbar(indices[0].iteritems(), total=len(indices[0])):
            trace_indices = [ind[idx] for ind in indices]

            header = self.geometries[0].segyfile.header[trace_indices[0]]
            keys = [header.get(field) for field in self.geometries[0].byte_no]
            store_key = [self.geometries[0].uniques_inversed[i][item] for i, item in enumerate(keys)]
            store_key = tuple(store_key)

            traces = [geometry.load_trace(trace_index) for
                      geometry, trace_index in zip(self.geometries, trace_indices)]

            metric[store_key] = func(*traces, **kwargs)

        title = f"tracewise {func}"
        plot_dict = {
            'title_label': f'{title} for `{self.name}` on cube `{self.geometry.displayed_name}`',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            'xlabel': 'INLINE_3D', 'ylabel': 'CROSSLINE_3D',
            **kwargs
        }
        return metric, plot_dict

    def tracewise_unsafe(self, func, l=3, pbar=True, **kwargs):
        """ Apply `func` to compare two cubes tracewise in an unsafe way:
        structure of cubes is assumed to be identical.
        """
        pbar = tqdm if pbar else lambda iterator, *args, **kwargs: iterator
        metric = np.full((*self.geometry.lens, l), np.nan)

        for idx in pbar(range(len(self.geometries[0].dataframe))):
            header = self.geometries[0].segyfile.header[idx]
            keys = [header.get(field) for field in self.geometries[0].byte_no]
            store_key = [self.geometries[0].uniques_inversed[i][item] for i, item in enumerate(keys)]
            store_key = tuple(store_key)

            traces = [geometry.load_trace(idx) for geometry in self.geometries]
            metric[store_key] = func(*traces, **kwargs)

        title = f"tracewise unsafe {func}"
        plot_dict = {
            'title_label': f'{title} for {self.name} on cube {self.geometry.displayed_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            'xlabel': 'INLINE_3D', 'ylabel': 'CROSSLINE_3D',
            **kwargs
        }
        return metric, plot_dict


    def blockwise(self, func, l=3, pbar=True, kernel=(5, 5), block_size=(1000, 1000),
                  heights=None, prep_func=None, **kwargs):
        """ Apply function to all traces in lateral window """

        window = np.array(kernel)
        low = window // 2
        high = window - low

        total = np.product(self.geometries[0].lens - window)
        prep_func = prep_func if prep_func else lambda x: x

        pbar = tqdm if pbar else lambda iterator, *args, **kwargs: iterator
        metric = np.full((*self.geometries[0].lens, l), np.nan)

        heights = slice(0, self.geometries[0].depth) if heights is None else slice(*heights)

        with pbar(total=total) as prog_bar:
            for il_block in np.arange(0, self.geometries[0].cube_shape[0], block_size[0]-window[0]):
                for xl_block in np.arange(0, self.geometries[0].cube_shape[1], block_size[1]-window[1]):
                    block_len = np.min((np.array(self.geometries[0].lens) - (il_block, xl_block),
                                        block_size), axis=0)
                    locations = [slice(il_block, il_block + block_len[0]),
                                 slice(xl_block, xl_block + block_len[1]),
                                 heights]

                    blocks = [prep_func(geometry.load_crop(locations)) for geometry in self.geometries]

                    for il_kernel in range(low[0], blocks[0].shape[0] - high[0]):
                        for xl_kernel in range(low[1], blocks[0].shape[1] - high[1]):

                            il_from, il_to = il_kernel - low[0], il_kernel + high[0]
                            xl_from, xl_to = xl_kernel - low[1], xl_kernel + high[1]

                            subsets = [b[il_from:il_to, xl_from:xl_to, :].reshape((-1, b.shape[-1])) for b in blocks]
                            metric[il_block + il_kernel, xl_block + xl_kernel, :] = func(*subsets, **kwargs)
                            prog_bar.update(1)

        title = f"Blockwise {func}"
        plot_dict = {
            'title_label': f'{title} for {self.name} on cube {self.geometry.displayed_name}',
            'cmap': 'seismic',
            'zmin': None, 'zmax': None,
            'ignore_value': np.nan,
            **kwargs
        }
        return metric, plot_dict

class FaultsMetrics:
    """ Faults metric class. """
    SHIFTS = [-20, -15, -5, 5, 15, 20]

    def similarity_metric(self, semblance, masks, threshold=None):
        """ Compute similarity metric for faults mask. """
        if threshold:
            masks = masks > threshold
        if semblance.ndim == 2:
            semblance = np.expand_dims(semblance, axis=0)
        if semblance.ndim == 3:
            semblance = np.expand_dims(semblance, axis=0)

        if masks.ndim == 2:
            masks = np.expand_dims(masks, axis=0)
        if masks.ndim == 3:
            masks = np.expand_dims(masks, axis=0)

        res = []
        m = self.sum_with_axes(masks * (1 - semblance), axes=[1,2,3])
        weights = np.ones((len(self.SHIFTS), 1))
        weights = weights / weights.sum()
        for i in self.SHIFTS:
            random_mask = self.make_shift(masks, shift=i)
            rm = self.sum_with_axes(random_mask * (1 - semblance), axes=[1,2,3])
            ratio = m/rm
            res += [np.log(ratio)]
        res = np.stack(res, axis=0)
        res = (res * weights).sum(axis=0)
        res = np.clip(res, -2, 2)
        return res

    def sum_with_axes(self, array, axes=None):
        """ Sum for several axes. """
        if axes is None:
            return array.sum()
        if isinstance(axes, int):
            axes = [axes]
        res = array
        axes = sorted(axes)
        for i, axis in enumerate(axes):
            res = res.sum(axis=axis-i)
        return res

    def make_shift(self, array, shift=20):
        """ Make shifts for mask. """
        result = np.zeros_like(array)
        for i, _array in enumerate(array):
            if shift > 0:
                result[i][:, shift:] = _array[:, :-shift]
            elif shift < 0:
                result[i][:, :shift] = _array[:, -shift:]
            else:
                result[i] = _array
        return result
