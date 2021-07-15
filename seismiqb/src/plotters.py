""" Plot functions. """
# pylint: disable=too-many-statements
from numbers import Number
from copy import copy
import colorsys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.cm import get_cmap, register_cmap
from matplotlib.patches import Patch
from matplotlib.colors import ColorConverter, ListedColormap, LinearSegmentedColormap, is_color_like
from mpl_toolkits import axes_grid1

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import to_list
from ..batchflow import deprecated



def plot_image(data=None, mode='imshow', backend='matplotlib', **kwargs):
    """ Overall plotter function, converting kwarg-names to match chosen backend and redirecting
    plotting task to one of the methods of backend-classes.
    """
    if backend in ('matplotlib', 'plt'):
        return MatplotlibPlotter.plot(data=data, mode=mode, **kwargs)
    if backend in ('plotly', 'go'):
        return getattr(PlotlyPlotter, mode)(data, **kwargs)
    raise ValueError('{} backend is not supported!'.format(backend))


def plot_loss(data, title=None, **kwargs):
    """ Shorthand for loss plotting. """
    kwargs = {
        'xlabel': 'Iterations',
        'ylabel': 'Loss',
        'label': title or 'Loss graph',
        'xlim': (0, None),
        'rolling_mean': 10,
        'final_mean': 100,
        **kwargs
    }
    return plot_image(data, mode='curve', backend='matplotlib', **kwargs)


def filter_parameters(params, keys=None, prefix='', index=None, index_condition=None):
    """ Make a subdictionary of parameters with required keys.

    Parameter are retrieved if:
    a. It is explicitly requested (via `keys` arg).
    b. Its name starts with given prefix (defined by `prefix` arg).

    Parameters
    ----------
    params : dict
        Arguments to filter.
    keys : sequence
        Keys to retrieve.
    prefix : str, optional
        Arguments with keys starting with given prefix will also be retrieved.
        Defaults to `''`, i.e. no prefix used.
    index : int
        Index of argument value to retrieve.
        If none provided, get whole argument value.
        If value is non-indexable, get it without indexing.
    index_condition : callable
        Function that takes indexed argument value and returns a bool specifying whether should it be really indexed.
    """
    result = {}

    keys = keys or list(params.keys())
    if prefix:
        keys += [key.split(prefix)[1] for key in params if key.startswith(prefix)]

    for key in keys:
        value = params.get(prefix + key, params.get(key))
        if value is None:
            continue
        # check if parameter value indexing is requested and possible
        if index is not None and isinstance(value, list):
            # check if there is no index condition or there is one and it is satisfied
            if index_condition is None or index_condition(value[index]):
                value = value[index]
        result[key] = value
    return result


class MatplotlibPlotter:
    """ Plotting backend for matplotlib.

    Basic class logic
    -----------------
    Simply provide data, plot mode and parameters to the `plot_image` call.
    `MatplotlibPlotter` takes care of redirecting params to methods they are meant for.

    The logic behind the process is the following:
    1. Convert some provided parameters from 'plotly' to 'matplotlib' naming convention.
    2. Obtain default params for chosen mode and merge them with provided params.
    3. Put data into a double-nested list (via `make_nested_data`).
       Nestedness levels define subplot and layer data order correspondingly.
    4. Parse axes or create them if none provided via `make_or_parse_axes`.
    5. For every axis-data pair:
       a. Filter params relevant for ax (via `filter_parameters`).
       b. Call chosen plot method (one of `imshow`, `wiggle`, `hist` or `curve`) with ax params.
       c. Apply all annotations with ax params (via `annotate_axis`).
    6. Show and save figure (via `show_and_save`).

    Data display scenarios
    ----------------------
    1. The simplest one if when one provide a single data array.
    2. A more advanced use case is when one provide a list of arrays:
       a. Images are put on same axis and overlaid: data=[image_1, image_2];
       b. Images are put on separate axes: data=[image_1, image_2], separate=True;
    3. The most complex scenario is when one wish to display images in a 'mixed' manner.
       For example, to overlay first two images but to display the third one separately, one must use:
       data = [[image_1, image_2], image_3];

    The order of arrrays inside the double-nested structure basically declares, which of them belong to the same axis
    and therefore should be rendered one over another, and which must be displayed separately.

    Note that in general parameters should resemble data nestedness level.
    That allows binding axes and parameters that correspond each other.
    However, it's possible for parameter to be a single item — in that case it's shared across all subplots and layers.

    Advanced parameters managing
    ----------------------------
    The list of parameters expected by specific plot method is rather short.
    But there is a way to provide parameter to a plot method, even if it's not hard-coded.
    One must use specific prefix for that.
    Address docs of `imshow`, `wiggle`, `hist`, `curve` and `annotate_axis` for details.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `fontsize` argument.
    Providing `{'fontsize': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.
    """
    @classmethod
    def plot(cls, data, mode='imshow', separate=False, **kwargs):
        """ Plot manager.

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.

        Parameters
        ----------
        data : np.ndarray or a list of np.ndarray objects (possibly nested)
            If list has level 1 nestedness, 'overlaid/separate' logic is handled via `separate` parameter.
            If list has level 2 nestedness, outer level defines subplots order while inner one defines layers order.
            Shape of data items depends on chosen mode (see below).
        mode : 'imshow', 'wiggle', 'hist', 'curve'
            If 'imshow' plot given arrays as images.
            If 'wiggle' plot 1d subarrays of given array as signals.
            Subarrays are extracted from given data with fixed step along vertical axis.
            If 'hist' plot histogram of flattened array.
            If 'curve' plot given arrays as curves.
        separate : bool
            Whether plot images on separate axes instead of putting them all together on a single one.
            Incompatible with 'wiggle' mode.
        kwargs :
            - For one of `imshow`, 'wiggle`, `hist` or `curve` (depending on chosen mode).
              Parameters and data nestedness levels must match.
              Every param with 'imshow_', 'wiggle_', 'hist_' or 'curve_' prefix is redirected to corresponding method.
            - For `annotate_axis`.
              Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'ticks_', 'xlim_',
              'ylim_', colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
              Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.
        """
        if mode == 'wiggle' and separate:
            raise ValueError("Can't use `separate` option with `wiggle` mode.")

        PLOTLY_TO_PYPLOT = {'zmin': 'vmin', 'zmax': 'vmax', 'xaxis': 'xlabel', 'yaxis': 'ylabel'}
        # pylint: disable=expression-not-assigned
        [kwargs.update({new: kwargs[old]}) for old, new in PLOTLY_TO_PYPLOT.items() if old in kwargs]

        mode_defaults = getattr(cls, f"{mode.upper()}_DEFAULTS")
        all_params = {**mode_defaults, **kwargs}

        data = cls.make_nested_data(data=data, separate=separate)
        axes = cls.make_or_parse_axes(mode=mode, n_subplots=len(data), all_params=all_params)
        for ax_num, (ax_data, ax) in enumerate(zip(data, axes)):
            index_condition = None if separate else lambda x: isinstance(x, list)
            ax_params = filter_parameters(all_params, index=ax_num, index_condition=index_condition)
            ax_params = getattr(cls, mode)(ax=ax, data=ax_data, **ax_params)
            cls.annotate_axis(ax=ax, ax_num=ax_num, ax_params=ax_params, all_params=all_params, mode=mode)

        [ax.set_axis_off() for ax in axes[len(data):]] # pylint: disable=expression-not-assigned

        return cls.save_and_show(fig=axes[0].figure, **kwargs)

    # Action methods

    @staticmethod
    def make_nested_data(data, separate):
        """ Construct nested list of data arrays for plotting. """
        if data is None:
            return []
        if isinstance(data, np.ndarray):
            return [[data]]
        if isinstance(data[0], Number):
            return [[np.array(data)]]
        if all(isinstance(item, np.ndarray) for item in data):
            return [[item] for item in data] if separate else [data]
        if separate:
            raise ValueError("Arrays list must be flat, when `separate` option is True.")
        return [[item] if isinstance(item, np.ndarray) else item for item in data]


    @classmethod
    def make_or_parse_axes(cls, mode, n_subplots, all_params):
        """ Create figure and axes if needed, else use provided. """
        MODE_TO_FIGSIZE = {'imshow' : (12, 12),
                           'hist' : (8, 5),
                           'wiggle' : (12, 7),
                           'curve': (15, 5)}

        axes = all_params.pop('axis', None)
        axes = all_params.pop('axes', axes)
        axes = all_params.pop('ax', axes)

        if axes is None:
            FIGURE_KEYS = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'constrained_layout']
            params = filter_parameters(all_params, FIGURE_KEYS, prefix='figure_')
            params['figsize'] = params.get('figsize', MODE_TO_FIGSIZE[mode])
            if ('ncols' not in params) and ('nrows' not in params):
                params['ncols'] = n_subplots
            _, axes = plt.subplots(**params)

        axes = to_list(axes)
        n_axes = len(axes)
        if n_axes < n_subplots:
            raise ValueError(f"Not enough axes provided ({n_axes}) for {n_subplots} subplots.")

        return axes


    @classmethod
    def annotate_axis(cls, ax, ax_num, ax_params, all_params, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        TEXT_KEYS = ['fontsize', 'family', 'color']

        # title
        keys = ['title', 'label', 'y'] + TEXT_KEYS
        params = filter_parameters(ax_params, keys, prefix='title_', index=ax_num)
        params['label'] = params.pop('title', None) or params.get('label')
        if params:
            ax.set_title(**params)

        # suptitle
        keys = ['t', 'y'] + TEXT_KEYS
        params = filter_parameters(ax_params, keys, prefix='suptitle_')
        params['t'] = params.get('t') or params.get('suptitle') or params.get('label')
        if params:
            ax.figure.suptitle(**params)

        # xlabel
        keys = ['xlabel'] + TEXT_KEYS
        params = filter_parameters(ax_params, keys, prefix='xlabel_', index=ax_num)
        if params:
            ax.set_xlabel(**params)

        # ylabel
        keys = ['ylabel'] + TEXT_KEYS
        params = filter_parameters(ax_params, keys, prefix='ylabel_', index=ax_num)
        if params:
            ax.set_ylabel(**params)

        # aspect
        params = filter_parameters(ax_params, ['aspect'], prefix='aspect_', index=ax_num)
        if params:
            ax.set_aspect(**params)

        # xticks
        params = filter_parameters(ax_params, ['xticks'], prefix='xticks_', index=ax_num)
        if 'xticks' in params:
            params['ticks'] = params.get('ticks', params.pop('xticks'))
        if params:
            ax.set_xticks(**params)

        # yticks
        params = filter_parameters(ax_params, ['yticks'], prefix='yticks_', index=ax_num)
        if 'yticks' in params:
            params['ticks'] = params.get('ticks', params.pop('yticks'))
        if params:
            ax.set_yticks(**params)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        params = filter_parameters(ax_params, keys, prefix='tick_', index=ax_num)
        if params:
            ax.tick_params(**params)

        # xlim
        params = filter_parameters(ax_params, ['xlim'], prefix='xlim_', index=ax_num)
        if 'xlim' in params:
            params['left'] = params.get('left', params.pop('xlim'))
        if params:
            ax.set_xlim(**params)

        # ylim
        params = filter_parameters(ax_params, ['ylim'], prefix='ylim_', index=ax_num)
        if 'ylim' in params:
            params['bottom'] = params.get('bottom', params.pop('ylim'))
        if params:
            ax.set_ylim(**params)

        # colorbar
        if all_params.get('colorbar', False) and mode == 'imshow':
            keys = ['colorbar', 'fraction', 'aspect', 'fake', 'ax_image']
            params = filter_parameters(ax_params, keys, prefix='colorbar_', index=ax_num)
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            params['fake'] = not params.pop('colorbar', True)
            cls.add_colorbar(**params)

        # legend
        keys = ['label', 'size', 'cmap', 'color', 'loc']
        params = filter_parameters(ax_params, keys, prefix='legend_')
        params['color'] = params.pop('cmap', None) or params.get('color')
        if params.get('label') is not None:
            cls.add_legend(ax, **params)

        # grid
        keys = ['grid', 'b', 'which', 'axis']
        params = filter_parameters(ax_params, keys, prefix='grid_', index=ax_num)
        params['b'] = params.pop('grid', params.pop('b', 'False'))
        if params:
            ax.grid(**params)

        if ax_params.get('facecolor'):
            ax.set_facecolor(ax_params['facecolor'])

        ax.set_axisbelow(ax_params.get('set_axisbelow', False))

        if ax_params.get('disable_axes'):
            ax.set_axis_off()
        elif not ax.axison:
            ax.set_axis_on()


    @staticmethod
    def save_and_show(fig, show=True, savepath=None, return_figure=False, pyqt=False, **kwargs):
        """ Save and show plot if needed. """
        if pyqt:
            return None
        save_kwargs = dict(bbox_inches='tight', pad_inches=0, dpi=100)
        save_kwargs.update(kwargs.get('save') or dict())

        # save if necessary and render
        if savepath is not None:
            fig.savefig(savepath, **save_kwargs)
        if show:
            fig.show()
        else:
            plt.close()

        plot_image.last_figure = fig
        if return_figure:
            return fig
        return None

    # Rendering methods

    IMSHOW_DEFAULTS = {
        # image
        'cmap': ['Greys_r', 'firebrick', 'forestgreen', 'royalblue', 'sandybrown', 'darkorchid'],
        'facecolor': 'white',
        # axis labels
        'xlabel': '', 'ylabel': '',
        # colorbar
        'colorbar_fraction': 3.0,
        'colorbar_aspect': 30,
        # ticks
        'labeltop': True,
        'labelright': True,
        'direction': 'inout',
        # legend
        'legend_loc': 0,
        'legend_size': 10,
        'legend_label': None,
        # common
        'fontsize': 20,
        # grid
        'grid': False,
        # other
        'order_axes': (1, 0, 2),
        'bad_color': (.0,.0,.0,.0),
        'transparize_masks': False,
    }

    @classmethod
    def imshow(cls, ax, data, **kwargs):
        """ Plot arrays as images one over another on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : list of np.ndarray
            Every item must be a valid matplotlib image.
        kwargs :
            order_axes : tuple of ints
                Order of image axes.
            bad_values : list of numbers
                Data values that should be displayed with 'bad_color'.
            transparize_masks : bool, optional
                Whether treat zeros in binary masks as bad values or not.
                If True, make zero values in all binary masks transparent on display.
                If False, do not make zero values in any binary masks transparent on display.
                If not provided, make zero values transparent in all masks that overlay an image.
            params for images drawn by `plt.imshow`:
                - 'cmap', 'vmin', 'vmax', 'interpolation', 'alpha', 'extent'
                - params with 'imshow_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, image in enumerate(data):
            image = np.transpose(image, axes=kwargs['order_axes'][:image.ndim]).astype(np.float32)

            # fill some values with nans to display them with `bad_color`
            bad_values = filter_parameters(kwargs, ['bad_values'], index=image_num).get('bad_values', [])
            if kwargs.get('transparize_masks', image_num > 0):
                unique_values = tuple(np.unique(image))
                if unique_values == (0,) or unique_values == (0, 1): # pylint: disable=consider-using-in
                    kwargs['vmin'] = params.get('vmin', 0)
                    bad_values = [0]
            for bad_value in bad_values:
                image[image == bad_value] = np.nan

            keys = ['cmap', 'vmin', 'vmax', 'interpolation', 'alpha', 'extent']
            params = filter_parameters(kwargs, keys, prefix='imshow_', index=image_num)
            params['cmap'] = cls.make_cmap(params.pop('cmap'), kwargs['bad_color'])
            params['extent'] = params.get('extent') or [0, image.shape[1], image.shape[0], 0]
            ax_image = ax.imshow(image, **params)
            if image_num == 0:
                kwargs['ax_image'] = ax_image

        return kwargs


    WIGGLE_DEFAULTS = {
        # main
        'step': 15,
        'width_multiplier': 1,
        'curve_width': 1,
        # wiggle
        'wiggle_color': 'k',
        'wiggle_linestyle': '-',
        # curve
        'color': 'r',
        'marker': 'o',
        'linestyle': '',
        # suptitle
        'suptitle_color': 'k',
        # title
        'title_color': 'k',
        # axis labels
        'xlabel': '', 'ylabel': '',
        'xlabel_color': 'k', 'ylabel_color': 'k',
        # ticks
        'labeltop': True,
        'labelright': True,
        'direction': 'inout',
        # legend
        'legend_loc': 1,
        'legend_size': 15,
        # grid
        'grid_axis': 'y',
        # common
        'set_axisbelow': True,
        'fontsize': 20, 'label': ''
    }

    @classmethod
    def wiggle(cls, ax, data, **kwargs):
        """ Make wiggle plot of signals array. Optionally overlap it with a curve.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : np.ndarray or list of np.ndarray
            If array, must be 2d.
            If list, must contain image and curve arrays.
            Curve, in turn must be either 1d array of heights or 2d array mask.
                If 1d heights, its shape must match correposnding image dimension.
                If 2d mask, its shape must match image shape.
                In both cases it is expected, that there must be `np.nan` where curve is not defined.
        kwargs :
            step : int, optional
                Step to take signals from the array with.
            width_multiplier : float, optional
                Scale factor for signals amplitudes.
            color : matplotlib color
                Wiggle lines color.
            fill_color : matplotlib color
                Wiggle fill color.
            xlim, ylims : tuples of int
                Define displayed data limits.

            params for overlaid curves drawn by `plt.plot`:
                - 'color', 'linestyle', 'marker', 'markersize'
                - params with 'curve_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        image, *curves = data

        offsets = np.arange(0, image.shape[0], kwargs['step'])
        y_range = np.arange(0, image.shape[1])

        x_range = [] # accumulate traces to draw curve above them if needed
        for offset in offsets:
            x = offset + kwargs['width_multiplier'] * image[offset] / np.std(image)
            params = filter_parameters(kwargs, ['color'], prefix='wiggle_')
            ax.plot(x, y_range, **params)

            fill_color = kwargs.get('fill_color') or params['color']
            ax.fill_betweenx(y_range, offset, x, where=(x > offset), color=fill_color)
            x_range.append(x)
        x_range = np.r_[x_range]

        if 'xlim' not in kwargs:
            kwargs['xlim'] = (x_range[0].min(), x_range[-1].max())
        if 'ylim' not in kwargs:
            kwargs['ylim'] = (y_range.max(), y_range.min())

        for curve_num, curve in enumerate(curves):
            keys = ['color', 'linestyle', 'marker', 'markersize']
            params = filter_parameters(kwargs, keys, prefix='curve_', index=curve_num)
            width = params.pop('width', 1)

            curve = curve[offsets]
            if curve.ndim == 1:
                curve_x = (~np.isnan(curve)).nonzero()[0]
                curve_y = curve[curve_x]
            # transform height-mask to heights if needed
            elif curve.ndim == 2:
                curve = (~np.isnan(curve)).nonzero()
                curve_x = curve[0][(width // 2)::width]
                curve_y = curve[1][(width // 2)::width]

            ax.plot(x_range[curve_x, curve_y], curve_y, **params)

        return kwargs


    HIST_DEFAULTS = {
        # hist
        'bins': 50,
        'color': ['firebrick', 'forestgreen', 'royalblue', 'sandybrown', 'darkorchid'],
        'alpha': 0.8,
        'facecolor': 'white',
        # suptitle
        'suptitle_color': 'k',
        'suptitle_y': 1.01,
        # title
        'title_color' : 'k',
        # axis labels
        'xlabel': '', 'ylabel': '',
        'xlabel_color' : 'k', 'ylabel_color' : 'k',
        # legend
        'legend_size': 10,
        'legend_label': None,
        'legend_loc': 0,
        # grid
        'grid': True,
        # common
        'set_axisbelow': True,
        'fontsize': 20
    }

    @classmethod
    def hist(cls, ax, data, **kwargs):
        """ Plot histograms on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : np.ndarray or list of np.ndarray
            Arrays to build histograms. Can be of any shape since they are flattened.
        kwargs :
            params for overlaid histograms drawn by `plt.hist`:
                - 'bins', 'color', 'alpha'
                - params with 'hist_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, array in enumerate(data):
            array = array.flatten()
            params = filter_parameters(kwargs, ['bins', 'color', 'alpha'], prefix='hist_', index=image_num)
            ax.hist(array, **params)

        return kwargs


    CURVE_DEFAULTS = {
        # main
        'rolling_mean': None,
        'rolling_final': None,
        # curve
        'color': ['skyblue', 'sandybrown', 'lightcoral'],
        'facecolor': 'white',
        # title
        'title_color': 'k',
        # axis labels
        'xlabel': 'x', 'ylabel': 'y',
        'xlabel_color': 'k', 'ylabel_color': 'k',
        # legend
        'legend_loc': 0,
        'legend_size': 10,
        'legend_label': None,
        # common
        'fontsize': 20,
        'grid': True
    }

    @classmethod
    def curve(cls, ax, data, **kwargs):
        """ Plot curves on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : np.ndarray or list of np.ndarray
            Arrays to plot. Must be 1d.
        kwargs :
            rolling_mean : int or None
                If int, calculate and display rolling mean with window `rolling_mean` size.
            rolling_final : int or None
                If int, calculate an display mean over last `rolling_final` array elements.
            params for overlaid curves drawn by `plt.plot`:
                - 'color', 'linestyle', 'alpha'
                - params with 'curve_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, array in enumerate(data):
            keys = ['color', 'linestyle', 'alpha']
            params = filter_parameters(kwargs, keys, prefix='curve_', index=image_num)
            ax.plot(array, **params)

            mean_color = cls.scale_lightness(params['color'], scale=.5)

            rolling_mean = kwargs.get('rolling_mean')
            if rolling_mean:
                averaged = array.copy()
                window = min(10 if rolling_mean is True else rolling_mean, len(array))
                averaged[(window // 2):(-window // 2 + 1)] = np.convolve(array, np.ones(window) / window, mode='valid')
                ax.plot(averaged, color=mean_color, linestyle='--')

            final_mean = kwargs.get('final_mean')
            if final_mean:
                window = 100 if final_mean is True else final_mean
                mean = np.mean(array[-window:])

                line_len = len(array) // 20
                curve_len = len(array)
                line_x = np.arange(line_len) + curve_len
                line_y = [mean] * line_len
                ax.plot(line_x, line_y, linestyle='--', linewidth=1.2, color=mean_color)

                fontsize = 10
                text_x = curve_len + line_len
                text_y = mean - fontsize / 300
                text = ax.text(text_x, text_y, f"{mean:.3f}", fontsize=fontsize)
                text.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white'), patheffects.Normal()])

                kwargs['xlim'] = (0, text_x)

        return kwargs

    # Predefined colormaps

    METRIC_CDICT = {
        'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
        'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
        'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
    }
    METRIC_CMAP = LinearSegmentedColormap('Metric', METRIC_CDICT)
    METRIC_CMAP.set_bad(color='black')
    register_cmap(name='Metric', cmap=METRIC_CMAP)

    DEPTHS_CMAP = ListedColormap(get_cmap('viridis_r')(np.linspace(0.0, 0.5, 100)))
    register_cmap(name='Depths', cmap=DEPTHS_CMAP)

    SAMPLER_CMAP = ListedColormap([ColorConverter().to_rgb('blue'),
                                   ColorConverter().to_rgb('red'),
                                   ColorConverter().to_rgb('purple')])
    register_cmap(name='Sampler', cmap=SAMPLER_CMAP)

    # Supplementary methods

    @staticmethod
    def make_cmap(color, bad_color=None):
        """ Make listed colormap from 'white' and provided color. """
        try:
            cmap = copy(plt.get_cmap(color))
        except ValueError: # if not a valid cmap name, expect it to be a matplotlib color
            if isinstance(color, str):
                color = ColorConverter().to_rgb(color)
            cmap = ListedColormap([(1, 1, 1, 1), color])

        if bad_color is not None:
            cmap.set_bad(color=bad_color)
        return cmap

    @staticmethod
    def scale_lightness(color, scale):
        """ Make new color with modified lightness from existing. """
        if isinstance(color, str):
            color = ColorConverter.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*color)
        return colorsys.hls_to_rgb(h, min(1, l * scale), s = s)

    @staticmethod
    def add_colorbar(ax_image, aspect=30, fraction=0.5, color='black', fake=False):
        """ Append colorbar to the image on the right. """
        divider = axes_grid1.make_axes_locatable(ax_image.axes)
        width = axes_grid1.axes_size.AxesY(ax_image.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        if fake:
            cax.set_axis_off()
        else:
            colorbar = ax_image.axes.figure.colorbar(ax_image, cax=cax)
            colorbar.ax.yaxis.set_tick_params(color=color)

    @staticmethod
    def add_legend(ax, color, label, size, loc):
        """ Add patches to legend. All invalid colors are filtered. """
        handles = getattr(ax.get_legend(), 'legendHandles', [])
        colors = [color for color in to_list(color) if is_color_like(color)]
        labels = to_list(label, dtype='object' if any(isinstance(item, list) for item in label) else None)
        new_patches = [Patch(color=color, label=label) for color, label in zip(colors, labels) if label]
        handles += new_patches
        if handles:
            ax.legend(handles=handles, loc=loc, prop={'size': size})



class PlotlyPlotter:
    """ Plotting backend for plotly. """

    DEPRECATION_MESSAGE = "Plotly backend is deprecated."

    @staticmethod
    def convert_kwargs(mode, kwargs):
        """ Update kwargs-dict to match plotly-conventions: update keys of the dict and
        values in some cases.
        """
        # make conversion-dict for kwargs-keys
        keys_converter = {
            'label': 'title', 't': 'title',
            'xlabel': 'xaxis', 'ylabel': 'yaxis',
            'vmin': 'zmin', 'vmax': 'zmax',
        }

        # make new dict updating keys and values
        converted = {}
        for key, value in kwargs.items():
            if key in keys_converter:
                new_key = keys_converter[key]
                if key == 'xlabel':
                    converted[new_key] = {'title_text': value,
                                          'automargin': True,
                                          'titlefont': {'size': kwargs.get('fontsize', 30)}}
                if key == 'ylabel':
                    converted[new_key] = {'title_text': value,
                                          'titlefont': {'size': kwargs.get('fontsize', 30)},
                                          'automargin': True,
                                          'autorange': 'reversed'}
                else:
                    converted[new_key] = value
            else:
                converted[key] = value
        return converted

    @staticmethod
    def channelize_image(image, total_channels, color=None, greyscale=False, opacity=None):
        """ Channelize an image. Can be used to make an opaque rgb or grayscale image.
        """
        # case of a partially channelized image
        if image.ndim == 3:
            if image.shape[-1] == total_channels:
                return image

            background = np.zeros((*image.shape[:-1], total_channels))
            background[:, :, :image.shape[-1]] = image

            if opacity is not None:
                background[:, :, -1] = opacity
            return background

        # case of non-channelized image
        if isinstance(color, str):
            color = ColorConverter().to_rgb(color)
        background = np.zeros((*image.shape, total_channels))
        for i, value in enumerate(color):
            background[:, :, i] = image * value

        # in case of greyscale make all 3 channels equal to supplied image
        if greyscale:
            for i in range(3):
                background[:, :, i] = image

        # add opacity if needed
        if opacity is not None:
            background[:, :, -1] = opacity * (image != 0).astype(int)

        return background

    @staticmethod
    def save_and_show(fig, show=True, savepath=None, **kwargs):
        """ Save and show plot if needed.
        """
        save_kwargs = kwargs.get('save', dict())

        # save if necessary and render
        if savepath is not None:
            fig.write_image(savepath, **save_kwargs)
        if show:
            fig.show()
        else:
            fig.close()

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def single(cls, image, **kwargs):
        """ Plot single image/heatmap using plotly.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            zmin : float
                the lowest brightness-level to be rendered.
            zmax : float
                the highest brightness-level to be rendered.
            opacity : float
                transparency-level of the rendered image
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('single', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'reversescale': True,
                    'colorscale': 'viridis',
                    'opacity' : 1.0,
                    'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        ax_params = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_parameters(ax_params, ['reversescale', 'colorscale', 'opacity', 'showscale'])
        label_kwargs = filter_parameters(ax_params, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = ax_params['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = ax_params['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Heatmap(z=np.transpose(image, axes=ax_params['order_axes'])[slc], **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **ax_params)

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def overlap(cls, images, **kwargs):
        """ Plot several images on one canvas using plotly: render the first one in greyscale
        and the rest ones in opaque 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting. Can store up to four images.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            opacity : float
                opacity of 'rgb' channels.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('overlap', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'coloraxis_colorbar': {'title': 'amplitude'},
                    'colors': ('red', 'green', 'blue'),
                    'opacity' : 1.0,
                    'title': 'Seismic inline',
                    'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        ax_params = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_parameters(ax_params, ['zmin', 'zmax'])
        label_kwargs = filter_parameters(ax_params, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = ax_params['slice']

        # calculate canvas sizes
        width, height = images[0].shape[1], images[0].shape[0]
        coeff = ax_params['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # manually combine first image in greyscale and the rest ones colored differently
        combined = cls.channelize_image(255 * np.transpose(images[0], axes=ax_params['order_axes']),
                                    total_channels=4, greyscale=True)
        for i, img in enumerate(images[1:]):
            color = ax_params['colors'][i]
            combined += cls.channelize_image(255 * np.transpose(img, axes=ax_params['order_axes']),
                                         total_channels=4, color=color, opacity=ax_params['opacity'])
        plot_data = go.Image(z=combined[slc], **render_kwargs) # plot manually combined image

        # plot the figure
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **ax_params)

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def rgb(cls, image, **kwargs):
        """ Plot one image in 'rgb' using plotly.

        Parameters
        ----------
        image : np.ndarray
            3d-array containing channeled rgb-image.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of the rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('rgb', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'coloraxis_colorbar': {'title': 'depth'},
                    'max_size' : 600,
                    'order_axes': (1, 0, 2),
                    'slice': (slice(None, None), slice(None, None))}
        ax_params = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_parameters(ax_params, [])
        label_kwargs = filter_parameters(ax_params, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = ax_params['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = ax_params['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Image(z=np.transpose(image, axes=ax_params['order_axes'])[slc], **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **ax_params)

    @classmethod
    @deprecated(DEPRECATION_MESSAGE)
    def separate(cls, images, **kwargs):
        """ Plot several images on a row of canvases using plotly.
        TODO: add grid support.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('separate', kwargs)
        # defaults
        defaults = {'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        grid = (1, len(images))
        ax_params = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_parameters(ax_params, [])
        label_kwargs = filter_parameters(ax_params, ['title'])
        xaxis_kwargs = filter_parameters(ax_params, ['xaxis'])
        yaxis_kwargs = filter_parameters(ax_params, ['yaxis'])
        slc = ax_params['slice']

        # make sure that the images are greyscale and put them each on separate canvas
        fig = make_subplots(rows=grid[0], cols=grid[1])
        for i in range(grid[1]):
            img = cls.channelize_image(255 * np.transpose(images[i], axes=ax_params['order_axes']),
                                   total_channels=4, greyscale=True, opacity=1)
            fig.add_trace(go.Image(z=img[slc], **render_kwargs), row=1, col=i + 1)
            fig.update_xaxes(row=1, col=i + 1, **xaxis_kwargs['xaxis'])
            fig.update_yaxes(row=1, col=i + 1, **yaxis_kwargs['yaxis'])
        fig.update_layout(**label_kwargs)

        cls.save_and_show(fig, **ax_params)

def show_3d(x, y, z, simplices, title, zoom_slice, colors=None, show_axes=True, aspect_ratio=(1, 1, 1),
            axis_labels=None, width=1200, height=1200, margin=(0, 0, 20), savepath=None,
            images=None, resize_factor=2, colorscale='Greys', **kwargs):
    """ Interactive 3D plot for some elements of cube.

    Parameters
    ----------
    x, y, z : numpy.ndarrays
        Triangle vertices.
    simplices : numpy.ndarray
        (N, 3) array where each row represent triangle. Elements of row are indices of points
        that are vertices of triangle.
    title : str
        Title of plot.
    zoom_slice : tuple of slices
        Crop from cube to show.
    colors : list or None
        List of colors for each simplex.
    show_axes : bool
        Whether to show axes and their labels.
    aspect_ratio : tuple of floats.
        Aspect ratio for each axis.
    axis_labels : tuple
        Titel for each axis.
    width, height : number
        Size of the image.
    margin : tuple of ints
        Added margin for each axis, by default, (0, 0, 20).
    savepath : str
        Path to save interactive html to.
    images : list of tuples
        Each tuple is triplet of image, location and axis to load slide from seismic cube.
    resize_factor : float
        Resize factor for seismic slides. Is needed to spedify loading and ploting of seismic slices.
    colorscale : str
        Colormap for seismic slides.
    kwargs : dict
        Other arguments of plot creation.
    """
    #pylint: disable=too-many-arguments
    # Arguments of graph creation
    kwargs = {
        'title': title,
        'colormap': [plt.get_cmap('Depths')(x) for x in np.linspace(0, 1, 10)],
        'edges_color': 'rgb(70, 40, 50)',
        'show_colorbar': False,
        'width': width,
        'height': height,
        'aspectratio': {'x': aspect_ratio[0], 'y': aspect_ratio[1], 'z': aspect_ratio[2]},
        **kwargs
    }
    if colors is not None:
        fig = ff.create_trisurf(x=x, y=y, z=z, color_func=colors, simplices=simplices, **kwargs)
    else:
        fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, **kwargs)
    if images is not None:
        for image, loc, axis in images:
            shape = image.shape
            image = cv2.resize(image, tuple(np.array(shape) // resize_factor))[::-1]
            grid = np.meshgrid(
                np.linspace(0, shape[0], image.shape[0]),
                np.linspace(0, shape[1], image.shape[1])
            )
            if axis == 0:
                x, y, z = loc * np.ones_like(image), grid[0].T + zoom_slice[1].start, grid[1].T + zoom_slice[2].start
            elif axis == 1:
                y, x, z = loc * np.ones_like(image), grid[0].T + zoom_slice[0].start, grid[1].T + zoom_slice[2].start
            else:
                z, x, y = loc * np.ones_like(image), grid[0].T + zoom_slice[0].start, grid[1].T + zoom_slice[1].start
            fig.add_surface(x=x, y=y, z=z, surfacecolor=np.flipud(image),
                            showscale=False, colorscale='Greys')
    # Update scene with title, labels and axes
    fig.update_layout(
        {
            'scene': {
                'xaxis': {
                    'title': axis_labels[0] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[0].stop + margin[0], zoom_slice[0].start - margin[0]]
                },
                'yaxis': {
                    'title': axis_labels[1] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[1].start + margin[1], zoom_slice[1].stop - margin[1]]
                },
                'zaxis': {
                    'title': axis_labels[2] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[2].stop + margin[2], zoom_slice[2].start - margin[2]]
                },
                'camera_eye': {
                    "x": 1.25, "y": 1.5, "z": 1.5
                },
            }
        }
    )
    fig.show()

    if savepath:
        fig.write_html(savepath)
