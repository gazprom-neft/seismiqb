""" Utility functions. """
import numpy as np
import pandas as pd
import segyio
from tqdm import tqdm
from skimage.measure import label, regionprops
from numba import njit, types
from numba.typed import Dict
import matplotlib.pyplot as plt

from ..batchflow import Pipeline, D, L, B, V

FILL_VALUE = -999
FILL_VALUE_A = -999999


def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.

    Parameters
    ----------
    path : str
        Location of original .sgy cube.

    geometry : SeismicGeometry
        Infered information about original cube.

    path_save : str
        Place to save subcube.

    i_range : array-like
        Ilines to include in subcube.

    x_range : array-like
        Xlines to include in subcube.
    """

    i_low, i_high = i_range[0], i_range[-1]
    x_low, x_high = x_range[0], x_range[-1]

    with segyio.open(path, 'r', strict=False) as src:
        src.mmap()
        spec = segyio.spec()
        spec.sorting = int(src.sorting)
        spec.format = int(src.format)
        spec.samples = range(geometry.depth)
        spec.ilines = geometry.ilines[i_low:i_high]
        spec.xlines = geometry.xlines[x_low:x_high]

        with segyio.create(path_save, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]

            c = 0
            for il_ in tqdm(spec.ilines):
                for xl_ in spec.xlines:
                    tr_ = geometry.il_xl_trace[(il_, xl_)]
                    dst.header[c] = src.header[tr_]
                    dst.header[c][segyio.TraceField.FieldRecord] = il_
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_FILE] = il_

                    dst.header[c][segyio.TraceField.TraceNumber] = xl_ - geometry.xlines_offset
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_LINE] = xl_ - geometry.xlines_offset
                    dst.trace[c] = src.trace[tr_]
                    c += 1
            dst.bin = src.bin
            dst.bin = {segyio.BinField.Traces: c}

    # Check that repaired cube can be opened in 'strict' mode
    with segyio.open(path_save, 'r', strict=True) as _:
        pass


def convert_point_cloud(path, path_save, names=None, order=None, transform=None):
    """ Change set of columns in file with point cloud labels.
    Usually is used to remove redundant columns.

    Parameters
    ----------
    path : str
        Path to file to convert.

    path_save : str
        Path for the new file to be saved to.

    names : str or sequence of str
        Names of columns in the original file. Default is Petrel's export format, which is
        ('_', '_', 'iline', '_', '_', 'xline', 'cdp_x', 'cdp_y', 'height'), where `_` symbol stands for
        redundant keywords like `INLINE`.

    order : str or sequence of str
        Names and order of columns to keep. Default is ('iline', 'xline', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    data = df.loc[:, order]
    if transform:
        data = data.apply(transform)
    data.to_csv(path_save, sep=' ', index=False, header=False)


def read_point_cloud(paths, names=None, order=None, **kwargs):
    """ Read point cloud of labels from files using pandas.

    Parameters
    ----------
    paths : str or tuple or list
        array-like of paths to files containing point clouds (table of floats with several columns).
    names : sequence
        sequence of column names in files.
    order : array-like
        specifies the order of columns to keep in the resulting array.
    **kwargs
        additional keyword-arguments of pandas parser.

    Returns
    -------
    ndarray
        resulting point-cloud. First three columns contain (iline, xline, height) while the last one stores
        horizon number.
    """
    #pylint: disable=anomalous-backslash-in-string
    paths = [paths] if isinstance(paths, str) else paths

    # default params of pandas-parser
    names = names or ['iline', 'xline', 'height']
    order = order or ['iline', 'xline', 'height']

    # read point clouds
    point_clouds = []
    for ix, path in enumerate(paths):
        cloud = pd.read_csv(path, sep='\s+', names=names, usecols=set(order), **kwargs)

        temp = np.hstack([cloud.loc[:, order].values,
                          np.ones((cloud.shape[0], 1)) * ix])
        point_clouds.append(temp)
    point_cloud = np.concatenate(point_clouds)
    return np.rint(point_cloud).astype(np.int64)


@njit
def _filter_point_cloud(point_cloud, zero_matrix, ilines_offset, xlines_offset):
    """ Remove entries corresponding to zero traces.

    Parameters
    ----------
    point_cloud : ndarray
        Point cloud with labels.

    zero_matrix : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions of zero-traces.

    ilines_offset, xlines_offset : int
        Offsets of numeration.
    """
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(point_cloud), dtype=np.int32)

    for i in range(len(point_cloud)):
        il, xl = point_cloud[i, 0], point_cloud[i, 1]
        if zero_matrix[il-ilines_offset, xl-xlines_offset] == 1:
            mask[i] = 0
    return point_cloud[mask == 1, :]


def make_labels_dict(point_cloud):
    """ Make labels-dict using cloud of points.

    Parameters
    ----------
    point_cloud : array
        array `(n_points, 4)`, contains point cloud of labels in format `(x, y, z, horizon_number)`.

    Returns
    -------
    numba.Dict
        dict of labels `{(x, y): [z_1, z_2, ...]}`.
    """
    # round and cast
    point_cloud = np.rint(point_cloud).astype(np.int64)
    max_count = point_cloud[-1, -1] + 1

    # make typed Dict
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    labels = Dict.empty(key_type, value_type)

    @njit
    def fill_labels(labels, point_cloud, max_count):
        """ Fill in labels-dict.
        """
        for i in range(len(point_cloud)):
            il, xl = point_cloud[i, :2]
            if labels.get((il, xl)) is None:
                labels[(il, xl)] = np.full((max_count, ), FILL_VALUE, np.int64)

            idx = int(point_cloud[i, 3])
            labels[(il, xl)][idx] = point_cloud[i, 2]

    fill_labels(labels, point_cloud, max_count)
    return labels


@njit
def _filter_labels(labels, zero_matrix, ilines_offset, xlines_offset):
    """ Remove (inplace) keys from labels dictionary according to zero_matrix.

    Parameters
    ----------
    labels : dict
        Dictionary with keys in (iline, xline) format.

    zero_matrix : ndarray
        Matrix of (n_ilines, n_xlines) shape with 1 on positions of zero-traces.

    ilines_offset, xlines_offset : int
        Offsets of numeration.
    """
    n_zeros = int(np.sum(zero_matrix))
    if n_zeros > 0:
        c = 0
        to_remove = np.zeros((n_zeros, 2), dtype=np.int64)

        for il, xl in labels.keys():
            if zero_matrix[il - ilines_offset, xl - xlines_offset] == 1:
                to_remove[c, 0] = il
                to_remove[c, 1] = xl
                c = c + 1

        for i in range(c):
            key = (to_remove[i, 0], to_remove[i, 1])
            if key in labels: # for some reason that is necessary
                labels.pop(key)

@njit
def create_mask(ilines_, xlines_, hs_,
                il_xl_h, ilines_offset, xlines_offset, geom_depth,
                mode, width, n_horizons=-1):
    """ Jit-accelerated function for fast mask creation from point cloud data stored in numba.typed.Dict.
    This function is usually called inside SeismicCropBatch's method `load_masks`.
    """
    #pylint: disable=line-too-long, too-many-nested-blocks, too-many-branches
    mask = np.zeros((len(ilines_), len(xlines_), len(hs_)))
    selected_idx = False
    for i, iline_ in enumerate(ilines_):
        for j, xline_ in enumerate(xlines_):
            il_, xl_ = iline_ + ilines_offset, xline_ + xlines_offset
            if il_xl_h.get((il_, xl_)) is None:
                continue
            m_temp = np.zeros(geom_depth)
            if mode == 'horizon':
                heights = il_xl_h[(il_, xl_)]
                if not selected_idx:
                    filtered_idx = np.array([idx for idx, height_ in enumerate(heights)
                                             if height_ != FILL_VALUE])
                    filtered_idx = np.array([idx for idx in filtered_idx
                                             if heights[idx] > hs_[0] and heights[idx] < hs_[-1]])
                    if len(filtered_idx) == 0:
                        continue
                    if n_horizons != -1 and len(filtered_idx) >= n_horizons:
                        filtered_idx = np.random.choice(filtered_idx, replace=False, size=n_horizons)
                        selected_idx = True
                for idx in filtered_idx:
                    _height = heights[idx]
                    if width == 0:
                        m_temp[_height] = 1
                    else:
                        m_temp[max(0, _height - width):min(_height + width, geom_depth)] = 1
            elif mode == 'stratum':
                current_col = 1
                start = 0
                sorted_heights = sorted(il_xl_h[(il_, xl_)])
                for height_ in sorted_heights:
                    if height_ == FILL_VALUE:
                        height_ = start
                    if start > hs_[-1]:
                        break
                    m_temp[start:height_ + 1] = current_col
                    start = height_ + 1
                    current_col += 1
                    m_temp[sorted_heights[-1] + 1:min(hs_[-1] + 1, geom_depth)] = current_col
            else:
                raise ValueError('Mode should be either `horizon` or `stratum`')
            mask[i, j, :] = m_temp[hs_]
    return mask

def _get_horizons(mask, threshold, averaging, transforms, separate=False):
    """ Compute horizons from a mask.
    """
    mask_ = np.zeros_like(mask, np.int32)
    mask_[mask >= threshold] = 1

    # get regions
    labels = label(mask_)
    regions = regionprops(labels)

    # make horizons-structure
    horizons = dict() if not separate else []
    for n_horizon, region in enumerate(regions):
        if separate:
            horizons.append(dict())

        # compute horizon-height for each inline-xline
        coords = region.coords
        coords = pd.DataFrame(coords, columns=['iline', 'xline', 'height'])
        horizon_ = getattr(coords.groupby(['iline', 'xline']), averaging)()

        # separate the columns
        ilines = horizon_.index.get_level_values('iline').values
        xlines = horizon_.index.get_level_values('xline').values
        heights = horizon_.values

        # transform each column
        try:
            ilines_ = transforms[0](ilines)
        except Exception as e:
            print('ilines ', ilines, transforms[0](0))
            raise e
        try:
            xlines_ = transforms[1](xlines)
        except Exception as e:
            print('xlines_ ', xlines, transforms[0](0))
            raise e

        try:
            heights_ = np.ravel(transforms[2](heights))
        except Exception as e:
            print('heights_ ', heights, transforms[2](0))
            raise e

#         xlines_ = transforms[1](xlines)
#         heights_ = np.ravel(transforms[2](heights))

        if separate:
            for key, h in zip(zip(ilines_, xlines_), heights_):
                horizons[n_horizon][key] = [h]
        else:
            for key, h in zip(zip(ilines_, xlines_), heights_):
                if key in horizons:
                    horizons[key].append(h)
                else:
                    horizons[key] = [h]
    return horizons


def dump_horizon(horizon, geometry, path_save, idx=None, offset=1):
    """ Save horizon in a point cloud format.

    Parameters
    ----------
    horizon : dict
        Mapping from pairs (iline, xline) to height.

    geometry : SeismicGeometry
        Information about cube

    path_save : str
        Path for the horizon to be saved to.

    offset : int, float
        Shift horizont before applying inverse transform.
        Usually is used to take into account different numeration bases:
        Petrel uses 1-based numeration, whilst Python uses 0-based numberation.
    """
    ixh = []
    for (i, x), h in horizon.items():
        if idx is not None:
            h = h[idx]
        ixh.append([i, x, h])
    ixh = np.asarray(ixh)

    ixh[:, -1] = (ixh[:, -1] + offset) * geometry.sample_rate + geometry.delay

    df = pd.DataFrame(ixh, columns=['iline', 'xline', 'height'])
    df.sort_values(['iline', 'xline'], inplace=True)
    df.to_csv(path_save, sep=' ', columns=['iline', 'xline', 'height'],
              index=False, header=False)


def compare_horizons(dict_1, dict_2, printer=print, plot=False, sample_rate=1, offset=0):
    """ Compare two horizons in dictionary format.

    Parameters
    ----------
    dict_1, dict_2 : dict
        Mappings from (iline, xline) to heights. Value can be either array or one number.

    printer : callable
        Function to output results with, for example `print` or `log.info`.

    plot : bool
        Whether to plot histogram of errors.

    sample_rate : number
        Frequency of taking measures. Used to normalize 5ms window.

    offset : number
        Value to shift horizon up. Can be used to take into account different counting bases.
    """
    differences = []
    not_present_1, not_present_2 = 0, 0
    vals_1, vals_2 = [], []

    for key, val_1 in dict_1.items():
        try:
            val_1 = val_1[0]
        except IndexError:
            pass

        val_2 = dict_2.get(key)
        if val_2 is not None:
            diff_ = abs(val_2 - val_1 - offset)
            idx = np.argmin(diff_)
            diff = diff_[idx]
            differences.append(diff)

            vals_1.append(val_1)
            vals_2.append(val_2[idx])
        else:
            not_present_1 += 1

    for key, val_2 in dict_2.items():
        if dict_1.get(key) is None:
            not_present_2 += 1

    printer('First horizont length:                    {}'.format(len(dict_1)))
    printer('Second horizont length:                   {}'.format(len(dict_2)))
    printer('Mean value/std of error:                  {:8.7} / {:8.7}' \
            .format(np.mean(differences), np.std(differences)))
    printer('Number in 5 ms window:                    {}' \
            .format(np.sum(np.array(differences) <= 5/sample_rate)))
    printer('Rate in 5 ms window:                      {:8.7}' \
            .format(np.sum(np.array(differences) <= 5/sample_rate) / len(differences)))

    printer('Average height of FIRST horizont:         {:8.7}'.format(np.mean(vals_1)))
    printer('Average height of SECOND horizont:        {:8.7}'.format(np.mean(vals_2)))

    printer('In the FIRST, but not in the SECOND:      {}'.format(not_present_1))
    printer('In the SECOND, but not in the FIRST:      {}'.format(not_present_2))
    printer('\n\n')

    if plot:
        plt.title('Distribution of errors', fontdict={'fontsize': 15})
        _ = plt.hist(differences, bins=100)



@njit
def count_nonfill(array):
    """ Jit-accelerated function to count non-fill elements. """
    count = 0
    for i in array:
        if i != FILL_VALUE:
            count += 1
    return count


@njit
def aggregate(array_crops, array_grid, crop_shape, predict_shape, order):
    """ Jit-accelerated function to glue together crops according to grid.
    At positions, where different crops overlap, only the maximum value is saved.
    This function is usually called inside SeismicCropBatch's method `assemble_crops`.
    """
    #pylint: disable=assignment-from-no-return
    total = len(array_grid)
    background = np.zeros(predict_shape)

    for i in range(total):
        il, xl, h = array_grid[i, :]
        il_end = min(background.shape[0], il+crop_shape[0])
        xl_end = min(background.shape[1], xl+crop_shape[1])
        h_end = min(background.shape[2], h+crop_shape[2])

        crop = np.transpose(array_crops[i], order)
        crop = crop[:(il_end-il), :(xl_end-xl), :(h_end-h)]
        previous = background[il:il_end, xl:xl_end, h:h_end]
        background[il:il_end, xl:xl_end, h:h_end] = np.maximum(crop, previous)
    return background


@njit(parallel=True)
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.

    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        ticks_ = ticks[i]
        if p <= ticks_[0]:
            values[i] = ticks_[0]
        elif p >= ticks_[-1]:
            values[i] = ticks_[-1]
        else:
            ix = np.searchsorted(ticks_, p)

            if abs(ticks_[ix] - p) <= abs(ticks_[ix-1] - p):
                values[i] = ticks_[ix]
            else:
                values[i] = ticks_[ix-1]
    return values

@njit
def update_minmax(array, val_min, val_max, matrix, il, xl, ilines_offset, xlines_offset):
    """ Get both min and max values in just one pass through array.
    Simultaneously updates (inplace) matrix if the trace is filled with zeros.
    """
    maximum = array[0]
    minimum = array[0]
    for i in array[1:]:
        if i > maximum:
            maximum = i
        elif i < minimum:
            minimum = i

    if (minimum == 0) and (maximum == 0):
        matrix[il - ilines_offset, xl - xlines_offset] = 1

    if minimum < val_min:
        val_min = minimum
    if maximum > val_max:
        val_max = maximum

    return val_min, val_max, matrix

def convert_to_numba_dict(_labels):
    """ Convert a dict to Numba dict.

    Parameters
    ----------
    _labels : dict
        Designed for a dict with special format
        keys must be tuples of length 2 with int64 values;
        dict's values must be arrays.

    Returns
    -------
    A Numba dict.
    """
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    _type_labels = Dict.empty(key_type, value_type)
    for key, value in _labels.items():
        _type_labels[key] = np.asarray(np.rint(value), dtype=np.int64)
    return _type_labels

def update_horizon_dict(first, second):
    """ Left merge two dicts. """
    for k, v in second.items():
        if not k in first:
            first.update({k: v})
    return first

def make_grid_info(grid_array, cube_name, crop_shape):
    """ Create grid info based on the grid array with lower left coordinates of the crops. """
    grid_array = np.array(grid_array)
    offsets = np.array([min(grid_array[:, 0]),
                        min(grid_array[:, 1]),
                        min(grid_array[:, 2])])
    grid_array = grid_array[:, :].astype(int) - offsets

    # this is not ilines/xlines coords
    ilines_range = [np.min(grid_array[:, 0]), np.max(grid_array[:, 0]) + 1]
    xlines_range = [np.min(grid_array[:, 1]), np.max(grid_array[:, 1]) + crop_shape[1]]
    h_range = [np.min(grid_array[:, 2]), np.max(grid_array[:, 2]) + crop_shape[2]]
    predict_shape = (ilines_range[1] - ilines_range[0],
                     xlines_range[1] - xlines_range[0],
                     h_range[1] - h_range[0])

    grid_info = {'grid_array': grid_array[:, :],
                 'predict_shape': predict_shape,
                 'crop_shape': crop_shape,
                 'cube_name': cube_name,
                 'range': [ilines_range, xlines_range, h_range],
                 'offsets': offsets}
    return grid_info

def compute_next_points(points, prediction, crop_shape, strides_candidates, width):
    """ Compute next point for extension procedure.
    """
    compared_slices_ = []
    compared_slices_.append(np.sum(prediction[:width, crop_shape[1] - width:]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] - width:, crop_shape[1] - width:]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] // 2 - width // 2:crop_shape[2] // 2 + width // 2,
                                              crop_shape[1] - width:]))
    compared_slices_.append(np.sum(prediction[:width ** 2 // crop_shape[1], :]))
    compared_slices_.append(np.sum(prediction[crop_shape[2] - width ** 2 // crop_shape[1]:, :]))
    stride = strides_candidates[np.argmax(np.array(compared_slices_))]
    points = [sum(x) for x in zip(points, stride)]
    return points, compared_slices_


def labels_to_depth_map(labels, geom, labels_idx=0, offset=0):
    """ Converts labels-dictionary to matrix of depths.

    Parameters
    ----------
    labels : dict
        Labeled horizon.

    labels_idx : int
        Index of item inside `labels` values.

    offset : number
        Value to add to each entry in matrix.
    """
    @njit
    def _labels_to_depth_map(labels, i_offset, x_offset, i_len, x_len, labels_idx=0, offset=0):
        depth_map = np.full((i_len, x_len), FILL_VALUE_A)

        for il in range(i_len):
            for xl in range(x_len):
                key = (il+i_offset, xl+x_offset)
                value = labels.get(key)
                if value is not None:
                    h = value[labels_idx]
                    if h != FILL_VALUE:
                        h += int(np.rint(offset))
                        depth_map[il, xl] = h
        return depth_map

    return _labels_to_depth_map(labels, geom.ilines_offset, geom.xlines_offset,
                                geom.ilines_len, geom.xlines_len,
                                labels_idx, offset)


def depth_map_to_labels(depth_map, geom, labels=None, labels_idx=0):
    """ Converts matrix of depths back into dictionary.
    Can also be used to replace dictionary values with updated ones.

    Parameters
    ----------
    depth_map : array
        Matrix of depths.

    labels : dict, optional
        If None, then new numba dictionary is created.
        If dict, then values are written into that dict.

    labels_idx : int
        Index of value to replace in passed `labels`.
    """
    key_type = types.Tuple((types.int64, types.int64))
    value_type = types.int64[:]
    max_count = len(list(labels.values())) if labels else 1
    labels_idx = labels_idx if labels else 0
    labels = labels or Dict.empty(key_type, value_type)

    @njit
    def _depth_map_to_labels(depth_map, i_offset, x_offset, labels, labels_idx, max_count):
        i_len, x_len = depth_map.shape

        for il in range(i_len):
            for xl in range(x_len):
                key = (il+i_offset, xl+x_offset)
                value = depth_map[il, xl]
                if labels.get(key) is None:
                    labels[key] = np.full((max_count, ), FILL_VALUE, np.int64)
                labels[key][labels_idx] = value
        return labels

    return _depth_map_to_labels(depth_map, geom.ilines_offset, geom.xlines_offset, labels, labels_idx, max_count)


def get_cube_values(labels, geom, labels_idx=0, window=3, offset=0, scale=False):
    """ Get values from the cube along the horizon.

    Parameters
    ----------
    labels : dict
        Labeled horizon.

    labels_idx : int
        Index of item inside `labels` values.

    window : int
        Width of data to cut.

    offset : int
        Value to add to each entry in matrix.

    scale : bool, callable
        If bool, then values are scaled to [0, 1] range.
        If callable, then it is applied to iline-oriented slices of data from the cube.
    """
    low = window // 2
    high = max(window - low, 0)

    h5py_cube = geom.h5py_file['cube']
    i_offset, x_offset = geom.ilines_offset, geom.xlines_offset
    i_len, x_len = geom.ilines_len, geom.xlines_len
    scale_val = (geom.value_max - geom.value_min)

    background = np.zeros((i_len, x_len, window))
    depth_map = np.full((geom.ilines_len, geom.xlines_len), FILL_VALUE_A)

    @njit
    def _update(background, depth_map, slide, labels,
                xlines_len, il, ilines_offset, xlines_offset,
                low, high, labels_idx, offset):
        for xl in range(xlines_len):
            key = (il+ilines_offset, xl+xlines_offset)
            arr = labels.get(key)
            if arr is not None:
                h = arr[labels_idx]
                if h != FILL_VALUE:
                    h += int(np.rint(offset))
                    depth_map[il, xl] = h
                value = slide[xl, h-low:h+high]
                background[il, xl, :] = value
        return background, depth_map

    for il in range(i_len):
        slide = h5py_cube[il, :, :]

        if callable(scale):
            slide = scale(slide)
        elif scale is True:
            slide -= geom.value_min
            slide *= (1 / scale_val)
        # here we can insert even more transforms!
        background, depth_map = _update(background, depth_map, slide, labels,
                                        x_len, il, i_offset, x_offset,
                                        low, high, labels_idx, offset)
    background = np.squeeze(background)
    return background, depth_map



@njit
def compute_corrs(data):
    """ Compute average correlation between each column in data and nearest traces. """
    i_range, x_range = data.shape[:2]
    corrs = np.zeros((i_range, x_range))

    for i in range(i_range):
        for x in range(x_range):
            trace = data[i, x, :]
            if np.max(trace) == np.min(trace):
                continue

            s, c = 0.0, 0

            if i > 0:
                trace_1 = data[i-1, x, :]
                if not np.max(trace_1) == np.min(trace_1):
                    s += np.corrcoef(trace, trace_1)[0, 1]
                    c += 1
            if i < i_range-1:
                trace_2 = data[i+1, x, :]
                if not np.max(trace_2) == np.min(trace_2):
                    s += np.corrcoef(trace, trace_2)[0, 1]
                    c += 1
            if x > 0:
                trace_3 = data[i, x-1, :]
                if not np.max(trace_3) == np.min(trace_3):
                    s += np.corrcoef(trace, trace_3)[0, 1]
                    c += 1
            if x < x_range-1:
                trace_4 = data[i, x+1, :]
                if not np.max(trace_4) == np.min(trace_4):
                    s += np.corrcoef(trace, trace_4)[0, 1]
                    c += 1

            if c != 0:
                corrs[i, x] = s / c
    return corrs



def create_predict_ppl(model_pipeline, crops_gen_name, crop_shape, axes):
    """ Default predict pipeline for extension model. """
    pred_pipeline = (Pipeline()
                         .load_component(src=[D('geometries'), D('labels')],
                                         dst=['geometries', 'labels'])
                         .add_components('predicted_labels')
                         .crop(points=L(D(crops_gen_name)),
                               shape=crop_shape, passdown='predicted_labels')
                         .load_component(src=[D('prior_mask')], dst=['predicted_labels'])
                         .load_cubes(dst='images')
                         .create_masks(dst='cut_masks', width=1, n_horizons=1, src_labels='predicted_labels')
                         .apply_transform(np.transpose, axes=axes, src=['images', 'cut_masks'])
                         .rotate_axes(src=['images', 'cut_masks'])
                         .scale(mode='normalize', src='images')
                         .import_model('extension', model_pipeline)
                         .init_variable('result_preds', default=list())
                         .concat_components(src=('images', 'cut_masks'), dst='model_inputs')
                         .predict_model('extension', fetches='sigmoid',
                                          images=B('model_inputs'),
                                          save_to=V('result_preds', mode='e')))
    return pred_pipeline
