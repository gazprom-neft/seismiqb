""" Seismic Crop Batch."""
import string
import random
from copy import copy

import numpy as np
import cv2
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, lfilter, hilbert

from ..batchflow import FilesIndex, Batch, action, inbatch_parallel, SkipBatchException, apply_parallel

from .horizon import Horizon
from .plotters import plot_image



AFFIX = '___'
SIZE_POSTFIX = 12
SIZE_SALT = len(AFFIX) + SIZE_POSTFIX
CHARS = string.ascii_uppercase + string.digits


class SeismicCropBatch(Batch):
    """ Batch with ability to generate 3d-crops of various shapes."""
    components = None
    apply_defaults = {
        'target': 'for',
        'post': '_assemble'
    }

    def _init_component(self, *args, **kwargs):
        """ Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args
        dst = kwargs.get("dst")
        if dst is None:
            raise KeyError("dst argument must be specified")
        if isinstance(dst, str):
            dst = (dst,)
        for comp in dst:
            if not hasattr(self, comp):
                self.add_components(comp, np.array([np.nan] * len(self.index)))
        return self.indices


    @staticmethod
    def salt(path):
        """ Adds random postfix of predefined length to string.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        path : str
            supplied string with random postfix.

        Notes
        -----
        Action `crop` makes a new instance of SeismicCropBatch with different (enlarged) index.
        Items in that index should point to cube location to cut crops from.
        Since we can't store multiple copies of the same string in one index (due to internal usage of dictionary),
        we need to augment those strings with random postfix, which can be removed later.
        """
        return path + AFFIX + ''.join(random.choice(CHARS) for _ in range(SIZE_POSTFIX))

    @staticmethod
    def has_salt(path):
        """ Check whether path is salted. """
        return path[::-1].find(AFFIX) == SIZE_POSTFIX

    @staticmethod
    def unsalt(path):
        """ Removes postfix that was made by `salt` method.

        Parameters
        ----------
        path : str
            supplied string.

        Returns
        -------
        str
            string without postfix.
        """
        if AFFIX in path:
            return path[:-SIZE_SALT]
        return path


    def __getattr__(self, name):
        """ Retrieve data from either `self` or attached dataset. """
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        return super().__getattr__(name)

    def get(self, item=None, component=None):
        """ Custom access for batch attribures.
        If `component` looks like `label` or `geometry`, then we retrieve that dictionary from
        attached dataset and use unsalted version of `item` as key.
        Otherwise, we get position of `item` in the current batch and use it to index sequence-like `component`.
        """
        if sum([attribute in component for attribute in ['label', 'geom']]):
            if isinstance(item, str) and self.has_salt(item):
                item = self.unsalt(item)
            res = getattr(self, component)
            if isinstance(res, dict) and item in res:
                return res[item]
            return res

        if item is not None:
            data = getattr(self, component) if isinstance(component, str) else component
            if isinstance(data, (np.ndarray, list)) and len(data) == len(self):
                pos = np.where(self.indices == item)[0][0]
                return data[pos]

            return super().get(item, component)
        return getattr(self, component)

    @action
    def crop(self, points, shape=None, direction=(0, 0, 0), side_view=False,
             adaptive_slices=False, grid_src='quality_grid', eps=3,
             dst='locations', passdown=None, dst_points='points', dst_shapes='shapes'):
        """ Generate positions of crops. Creates new instance of `SeismicCropBatch`
        with crop positions in one of the components (`locations` by default).

        Parameters
        ----------
        points : array-like
            Upper rightmost points for every crop and name of cube to
            cut it from. Order is: name, iline, xline, height. For example,
            ['Cube.sgy', 13, 500, 200] stands for crop has [13, 500, 200]
            as its upper rightmost point and must be cut from 'Cube.sgy' file.
        shape : sequence, ndarray
            Desired shape of crops along (iline, xline, height) axis. If ndarray, then must have the same length,
            as `points`, and each row contains a shape for corresponding point.
        direction : sequence of numbers
            Direction of the cut crop relative to the point. Must be a vector on unit cube.
        side_view : bool or float
            Determines whether to generate crops of transposed shape (xline, iline, height).
            If False, then shape is never transposed.
            If True, then shape is transposed with 0.5 probability.
            If float, then shape is transposed with that probability.
        adaptive_slices: bool or str
            If True, then slices are created so that crops are cut only along the grid.
        grid_src : str
            Attribut of geometry to get the grid from.
        eps : int
            Initial length of slice, that is used to find the closest grid point.
        dst : str, optional
            Component of batch to put positions of crops in.
        passdown : str of list of str
            Components of batch to keep in the new one.
        dst_points, dst_shapes : str
            Components to put points and crop shapes in.

        Notes
        -----
        Based on the first column of `points`, new instance of SeismicCropBatch is created.
        In order to keep multiple references to the same cube, each index is augmented
        with prefix of fixed length (check `salt` method for details).

        Returns
        -------
        SeismicCropBatch
            Batch with positions of crops in specified component.
        """
        # pylint: disable=protected-access

        # Create all the points and shapes
        if isinstance(shape, dict):
            shape = {k: np.asarray(v) for k, v in shape.items()}
        else:
            shape = np.asarray(shape)

        if adaptive_slices:
            indices, points_, shapes = [], [], []
            for point in points:
                try:
                    shape_ = shape[points[0]] if isinstance(shape, dict) else shape
                    point_, shape_ = self._correct_point_to_grid(point, shape_, grid_src, eps)
                    indices.append(point[0])
                    points_.append(point_)
                    shapes.append(shape_)
                except RecursionError:
                    pass
            points = points_
        else:
            indices = points[:, 0]
            shapes = self._make_shapes(points, shape, side_view)

        locations = [self._make_location(point, shape, direction) for point, shape in zip(points, shapes)]

        # Create a new Batch instance, if needed
        if not hasattr(self, 'transformed'):
            new_index = [self.salt(ix) for ix in indices]
            new_dict = {ix: self.index.get_fullpath(self.unsalt(ix)) for ix in new_index}
            new_batch = type(self)(FilesIndex.from_index(index=new_index, paths=new_dict, dirs=False))
            new_batch.transformed = True

            passdown = passdown or []
            passdown = [passdown] if isinstance(passdown, str) else passdown

            for component in passdown:
                if hasattr(self, component):
                    new_batch.add_components(component, getattr(self, component))
        else:
            if len(points) != len(self):
                raise ValueError('Subsequent usage of `crop` must have the same number of points!')
            new_batch = self

        new_batch.add_components((dst_points, dst_shapes), (points, shapes))
        new_batch.add_components(dst, locations)
        return new_batch

    def _make_shapes(self, points, shape, side_view):
        """ Make an array of shapes to cut. """
        # If already array of desired shapes
        if isinstance(shape, np.ndarray) and shape.ndim == 2 and len(shape) == len(points):
            return shape

        if side_view:
            side_view = side_view if isinstance(side_view, float) else 0.5
        shapes = []
        for point in points:
            shape_ = shape[point[0]] if isinstance(shape, dict) else shape
            if not side_view:
                shapes.append(shape_)
            else:
                flag = np.random.random() > side_view
                if flag:
                    shapes.append(shape_)
                else:
                    shapes.append(shape_[[1, 0, 2]])
        shapes = np.array(shapes)
        return shapes

    def _make_location(self, point, shape, direction=(0, 0, 0)):
        """ Creates list of slices for desired location. """
        if isinstance(point[1], float) or isinstance(point[2], float) or isinstance(point[3], float):
            ix = point[0]
            cube_shape = np.array(self.get(ix, 'geometries').cube_shape)
            anchor_point = np.rint(point[1:].astype(float) * (cube_shape - np.array(shape))).astype(int)
        else:
            anchor_point = point[1:]

        location = []
        for i in range(3):
            start = int(max(anchor_point[i] - direction[i]*shape[i], 0))
            stop = start + shape[i]
            location.append(slice(start, stop))
        return location

    def _correct_point_to_grid(self, point, shape, grid_src='quality_grid', eps=3):
        """ Move the point to the closest location in the quality grid. """
        #pylint: disable=too-many-return-statements
        ix = point[0]
        geometry = self.get(ix, 'geometries')
        grid = getattr(geometry, grid_src) if isinstance(grid_src, str) else grid_src
        shape_t = shape[[1, 0, 2]]

        pnt = (point[1:] * geometry.cube_shape)
        pnt = np.rint(pnt.astype(float)).astype(int)

        # Point is already in grid
        if grid[pnt[0], pnt[1]] == 1:
            sum_i = np.nansum(grid[pnt[0], max(pnt[1]-eps, 0) : pnt[1]+eps])
            sum_x = np.nansum(grid[max(pnt[0]-eps, 0) : pnt[0]+eps, pnt[1]])
            if sum_i >= sum_x:
                return point, shape
            return point, shape_t

        # Horizontal search: xline changes, shape is x-oriented
        for pnt_ in range(max(pnt[1]-eps, 0), min(pnt[1]+eps, geometry.cube_shape[1])):
            if grid[pnt[0], pnt_] == 1:
                sum_i = np.nansum(grid[pnt[0], max(pnt_-eps, 0):pnt_+eps])
                sum_x = np.nansum(grid[max(pnt[0]-eps, 0):pnt[0]+eps, pnt_])
                point[1:3] = np.array((pnt[0], pnt_)) / geometry.cube_shape[:2]
                if sum_i >= sum_x:
                    return point, shape
                return point, shape_t

        # Vertical search: inline changes, shape is i-oriented
        for pnt_ in range(max(pnt[0]-eps, 0), min(pnt[0]+eps, geometry.cube_shape[0])):
            if grid[pnt_, pnt[1]] == 1:
                sum_i = np.nansum(grid[pnt_, max(pnt[1]-eps, 0) : pnt[1]+eps])
                sum_x = np.nansum(grid[max(pnt_-eps, 0) : pnt_+eps, pnt[1]])
                point[1:3] = np.array((pnt_, pnt[1])) / geometry.cube_shape[:2]
                if sum_i >= sum_x:
                    return point, shape
                return point, shape_t

        # Double the search radius
        return self._correct_point_to_grid(point, shape, grid_src, 2*eps)


    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def load_cubes(self, ix, dst, src_locations='locations', src_geometry='geometries', slicing='custom', **kwargs):
        """ Load data from cube in given positions.

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.
        dst : str
            Component of batch to put loaded crops in.
        slicing : str
            if 'native', crop will be looaded as a slice of geometry. If 'custom', use `load_crop` method to make crops.
            The 'native' option is prefered to 3D crops to speed up loading.
        """
        geometry = self.get(ix, src_geometry)
        location = self.get(ix, src_locations)
        if slicing == 'native':
            crop = geometry[tuple(location)]
        elif slicing == 'custom':
            crop = geometry.load_crop(location, **kwargs)
        else:
            raise ValueError(f"slicing must be 'native' or 'custom' but {slicing} were given.")
        return crop

    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def create_masks(self, ix, dst, src='locations', width=3, src_labels='labels', indices=-1):
        """ Create masks from labels-dictionary in given positions.

        Parameters
        ----------
        src : str
            Component of batch with positions of crops to load.
        dst : str
            Component of batch to put loaded masks in.
        width : int
            Width of horizons in the `horizon` mode.
        src_labels : str
            Component of batch with labels dict.
        indices : str, int or sequence of ints
            A choice scenario of used labels per crop.
            If -1 or 'all', all possible labels will be added.
            If 1 or 'single', one random label will be added.
            If array-like then elements are interpreted as indices of the desired labels
            and must be ints in range [0, len(horizons) - 1].
            Note if you want to pass an index of a single label it must be a list with one
            element.

        Returns
        -------
        SeismicCropBatch
            Batch with loaded masks in desired components.

        Notes
        -----
        Can be run only after labels-dict is loaded into labels-component.
        """
        location = self.get(ix, src)
        shape_ = self.get(ix, 'shapes')
        mask = np.zeros((shape_), dtype='float32')

        labels = self.get(ix, src_labels) if isinstance(src_labels, str) else src_labels
        labels = [labels] if not isinstance(labels, (tuple, list)) else labels
        if len(labels) == 0:
            return mask

        check_sum = False
        if indices in [-1, 'all']:
            indices = np.arange(0, len(labels))
        elif indices in [1, 'single']:
            indices = np.arange(0, len(labels))
            np.random.shuffle(indices)
            check_sum = True
        elif isinstance(indices, int):
            raise ValueError('Inidices should be either -1, 1 or a sequence of ints.')
        elif isinstance(indices, (tuple, list, np.ndarray)):
            pass
        labels = [labels[idx] for idx in indices]

        for label in labels:
            mask = label.add_to_mask(mask, locations=location, width=width)
            if check_sum and np.sum(mask) > 0.0:
                break
        return mask


    @action
    @inbatch_parallel(init='indices', post='_post_mask_rebatch', target='for',
                      src='masks', threshold=0.8, passdown=None, axis=-1)
    def mask_rebatch(self, ix, src='masks', threshold=0.8, passdown=None, axis=-1):
        """ Remove elements with masks area lesser than a threshold.

        Parameters
        ----------
        threshold : float
            Minimum percentage of covered area (spatial-wise) for a mask to be kept in the batch.
        passdown : sequence of str
            Components to filter.
        axis : int
            Axis to project horizon to before computing mask area.
        """
        _ = threshold, passdown
        mask = self.get(ix, src)

        reduced = np.max(mask, axis=axis) > 0.0
        return np.sum(reduced) / np.prod(reduced.shape)

    def _post_mask_rebatch(self, areas, *args, src=None, passdown=None, threshold=None, **kwargs):
        #pylint: disable=protected-access, access-member-before-definition, attribute-defined-outside-init
        _ = args, kwargs
        new_index = [self.indices[i] for i, area in enumerate(areas) if area > threshold]
        new_dict = {idx: self.index._paths[idx] for idx in new_index}
        if len(new_index):
            self.index = FilesIndex.from_index(index=new_index, paths=new_dict, dirs=False)
        else:
            raise SkipBatchException

        passdown = passdown or []
        passdown.extend([src, 'locations', 'shapes'])
        passdown = list(set(passdown))

        for compo in passdown:
            new_data = [getattr(self, compo)[i] for i, area in enumerate(areas) if area > threshold]
            setattr(self, compo, np.array(new_data))
        return self


    @action
    @inbatch_parallel(init='_init_component', post='_assemble', target='for')
    def filter_out(self, ix, src=None, dst=None, mode=None, expr=None, low=None, high=None,
                   length=None, p=1.0):
        """ Zero out mask for horizon extension task.

        Parameters
        ----------
        src : str
            Component of batch with mask
        dst : str
            Component of batch to put cut mask in.
        mode : str
            Either point, line, iline or xline.
            If point, then only one point per horizon will be labeled.
            If iline or xline then single iline or xline with labeled.
            If line then randomly either single iline or xline will be
            labeled.
        expr : callable, optional.
            Some vectorized function. Accepts points in cube, returns either float.
            If not None, low or high/length should also be supplied.
        p : float
            Probability of applying the transform. Default is 1.
        """
        if not (src and dst):
            raise ValueError('Src and dst must be provided')

        mask = self.get(ix, src)
        coords = np.where(mask > 0)

        if np.random.binomial(1, 1 - p) or len(coords[0]) == 0:
            return mask
        if mode is not None:
            new_mask = np.zeros_like(mask)
            point = np.random.randint(len(coords))
            if mode == 'point':
                new_mask[coords[0][point], coords[1][point], :] = mask[coords[0][point], coords[1][point], :]
            elif mode == 'iline' or (mode == 'line' and np.random.binomial(1, 0.5)) == 1:
                new_mask[coords[0][point], :, :] = mask[coords[0][point], :, :]
            elif mode in ['xline', 'line']:
                new_mask[:, coords[1][point], :] = mask[:, coords[1][point], :]
            else:
                raise ValueError('Mode should be either `point`, `iline`, `xline` or `line')
        if expr is not None:
            coords = np.where(mask > 0)
            new_mask = np.zeros_like(mask)

            coords = np.array(coords).astype(np.float).T
            cond = np.ones(shape=coords.shape[0]).astype(bool)
            coords /= np.reshape(mask.shape, newshape=(1, 3))
            if low is not None:
                cond &= np.greater_equal(expr(coords), low)
            if high is not None:
                cond &= np.less_equal(expr(coords), high)
            if length is not None:
                low = 0 if not low else low
                cond &= np.less_equal(expr(coords), low + length)
            coords *= np.reshape(mask.shape, newshape=(1, 3))
            coords = np.round(coords).astype(np.int32)[cond]
            new_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = mask[coords[:, 0],
                                                                      coords[:, 1],
                                                                      coords[:, 2]]
        else:
            new_mask = mask
        return new_mask


    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def scale(self, ix, mode, src=None, dst=None):
        """ Scale values in crop.

        Parameters
        ----------
        mode : str
            If `minmax`, then data is scaled to [0, 1] via minmax scaling.
            If `q` or `normalize`, then data is divided by the maximum of absolute values of the
            0.01 and 0.99 quantiles.
            If `q_clip`, then data is clipped to 0.01 and 0.99 quantiles and then divided by the
            maximum of absolute values of the two.
        """
        data = self.get(ix, src)
        geometry = self.get(ix, 'geometries')
        return geometry.scaler(data, mode)


    @action
    @inbatch_parallel(init='_init_component', post='_assemble', target='for')
    def concat_components(self, ix, src, dst, axis=-1):
        """ Concatenate a list of components and save results to `dst` component.

        Parameters
        ----------
        src : array-like
            List of components to concatenate of length more than one.
        dst : str
            Component of batch to put results in.
        axis : int
            The axis along which the arrays will be joined.
        """
        _ = dst
        if not isinstance(src, (list, tuple, np.ndarray)) or len(src) < 2:
            raise ValueError('Src must contain at least two components to concatenate')
        result = []
        for component in src:
            result.append(self.get(ix, component))
        return np.concatenate(result, axis=axis)

    @action
    @inbatch_parallel(init='indices', target='for', post='_masks_to_horizons_post')
    def masks_to_horizons(self, ix, src='masks', src_locations='locations', dst='predicted_labels', prefix='predict',
                          threshold=0.5, mode='mean', minsize=0, mean_threshold=2.0, adjacency=1,
                          order=(2, 0, 1), skip_merge=False):
        """ Convert predicted segmentation mask to a list of Horizon instances.

        Parameters
        ----------
        src_masks : str
            Component of batch that stores masks.
        src_locations : str
            Component of batch that stores locations of crops.
        dst : str/object
            Component of batch to store the resulting horizons.
        order : tuple of int
            Axes-param for `transpose`-operation, applied to a mask before fetching point clouds.
            Default value of (2, 0, 1) is applicable to standart pipeline with one `rotate_axes`
            applied to images-tensor.
        threshold, mode, minsize, mean_threshold, adjacency, prefix
            Passed directly to `:meth:Horizon.from_mask`.
        """
        _ = dst, mean_threshold, adjacency, skip_merge

        # Threshold the mask, transpose and rotate the mask if needed
        mask = self.get(ix, src)
        if np.array(order).reshape(-1, 3).shape[0] > 0:
            order = self.get(ix, np.array(order).reshape(-1, 3))
        mask = np.transpose(mask, axes=order)

        geometry = self.get(ix, 'geometries')
        shifts = [self.get(ix, src_locations)[k].start for k in range(3)]
        horizons = Horizon.from_mask(mask, geometry=geometry, shifts=shifts, threshold=threshold,
                                     mode=mode, minsize=minsize, prefix=prefix)
        return horizons


    def _masks_to_horizons_post(self, horizons_lists, *args, dst=None, skip_merge=False,
                                mean_threshold=2.0, adjacency=1, **kwargs):
        """ Flatten list of lists of horizons, attempting to merge what can be merged. """
        _, _ = args, kwargs
        if dst is None:
            raise ValueError("dst should be initialized with empty list.")

        if skip_merge:
            setattr(self, dst, [hor for hor_list in horizons_lists for hor in hor_list])
            return self

        for horizons in horizons_lists:
            for horizon_candidate in horizons:
                for horizon_target in dst:
                    merge_code, _ = Horizon.verify_merge(horizon_target, horizon_candidate,
                                                         mean_threshold=mean_threshold,
                                                         adjacency=adjacency)
                    if merge_code == 3:
                        merged = Horizon.overlap_merge(horizon_target, horizon_candidate, inplace=True)
                    elif merge_code == 2:
                        merged = Horizon.adjacent_merge(horizon_target, horizon_candidate, inplace=True,
                                                        adjacency=adjacency, mean_threshold=mean_threshold)
                    else:
                        merged = False
                    if merged:
                        break
                else:
                    # If a horizon can't be merged to any of the previous ones, we append it as it is
                    dst.append(horizon_candidate)
        return self


    @apply_parallel
    def adaptive_reshape(self, crop, shape):
        """ Changes axis of view to match desired shape.
        Must be used in combination with `side_view` argument of `crop` action.

        Parameters
        ----------
        shape : sequence
            Desired shape of resulting crops.
        """
        if (np.array(crop.shape) != np.array(shape)).any():
            return crop.transpose([1, 0, 2])
        return crop

    @apply_parallel
    def shift_masks(self, crop, n_segments=3, max_shift=4, max_len=10):
        """ Randomly shift parts of the crop up or down.

        Parameters
        ----------
        n_segments : int
            Number of segments to shift.
        max_shift : int
            Size of shift along vertical axis.
        max_len : int
            Size of shift along horizontal axis.
        """
        crop = np.copy(crop)
        for _ in range(n_segments):
            # Point of starting the distortion, its length and size
            begin = np.random.randint(0, crop.shape[1])
            length = np.random.randint(5, max_len)
            shift = np.random.randint(-max_shift, max_shift)

            # Apply shift
            segment = crop[:, begin:min(begin + length, crop.shape[1]), :]
            shifted_segment = np.zeros_like(segment)
            if shift > 0:
                shifted_segment[:, :, shift:] = segment[:, :, :-shift]
            elif shift < 0:
                shifted_segment[:, :, :shift] = segment[:, :, -shift:]
            crop[:, begin:min(begin + length, crop.shape[1]), :] = shifted_segment
        return crop

    @apply_parallel
    def bend_masks(self, crop, angle=10):
        """ Rotate part of the mask on a given angle.
        Must be used for crops in (xlines, heights, inlines) format.
        """
        shape = crop.shape

        if np.random.random() >= 0.5:
            point_x = np.random.randint(shape[0]//2, shape[0])
            point_h = np.argmax(crop[point_x, :, :])

            if np.sum(crop[point_x, point_h, :]) == 0.0:
                return np.copy(crop)

            matrix = cv2.getRotationMatrix2D((point_h, point_x), angle, 1)
            rotated = cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

            combined = np.zeros_like(crop)
            combined[:point_x, :, :] = crop[:point_x, :, :]
            combined[point_x:, :, :] = rotated[point_x:, :, :]
        else:
            point_x = np.random.randint(0, shape[0]//2)
            point_h = np.argmax(crop[point_x, :, :])

            if np.sum(crop[point_x, point_h, :]) == 0.0:
                return np.copy(crop)

            matrix = cv2.getRotationMatrix2D((point_h, point_x), angle, 1)
            rotated = cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

            combined = np.zeros_like(crop)
            combined[point_x:, :, :] = crop[point_x:, :, :]
            combined[:point_x, :, :] = rotated[:point_x, :, :]
        return combined

    @apply_parallel
    def linearize_masks(self, crop, n=3, shift=0, kind='random', width=None):
        """ Sample `n` points from the original mask and create a new mask by interpolating them.

        Parameters
        ----------
        n : int
            Number of points to sample.
        shift : int
            Maximum amplitude of random shift along the heights axis.
        kind : {'random', 'linear', 'slinear', 'quadratic', 'cubic', 'previous', 'next'}
            Type of interpolation to use. If 'random', then chosen randomly for each crop.
        width : int
            Width of interpolated lines.
        """
        # Parse arguments
        if kind == 'random':
            kind = np.random.choice(['linear', 'slinear', 'quadratic', 'cubic'])
        width = width or np.sum(crop, axis=2).mean()

        # Choose the anchor points
        axis = 1 - np.argmin(crop.shape)
        *nz, _ = np.nonzero(crop)
        min_, max_ = nz[axis][0], nz[axis][-1]
        idx = [min_, max_]

        step = (max_ - min_) // n
        for i in range(0, max_-step, step):
            idx.append(np.random.randint(i, i + step))

        # Put anchors into new mask
        mask_ = np.zeros_like(crop)
        slc = (idx if axis == 0 else slice(None),
               idx if axis == 1 else slice(None),
               slice(None))
        mask_[slc] = crop[slc]
        *nz, y = np.nonzero(mask_)

        # Shift heights randomly
        x = nz[axis]
        y += np.random.randint(-shift, shift + 1, size=y.shape)

        # Sort and keep only unique values, based on `x` to remove width of original mask
        sort_indices = np.argsort(x)
        x, y = x[sort_indices], y[sort_indices]
        _, unique_indices = np.unique(x, return_index=True)
        x, y = x[unique_indices], y[unique_indices]

        # Interpolate points; put into mask
        interpolator = interp1d(x, y, kind=kind)
        indices = np.arange(min_, max_, dtype=np.int32)
        heights = interpolator(indices).astype(np.int32)

        slc = (indices if axis == 0 else indices * 0,
               indices if axis == 1 else indices * 0,
               np.clip(heights, 0, 255))
        mask_[slc] = 1

        # Make horizon wider
        structure = np.ones((1, 3), dtype=np.uint8)
        return cv2.dilate(mask_, structure, iterations=width)

    @apply_parallel
    def transpose(self, crop, order):
        """ Change order of axis. """
        return np.transpose(crop, order)

    @apply_parallel
    def rotate_axes(self, crop):
        """ The last shall be first and the first last.

        Notes
        -----
        Actions `crop`, `load_cubes`, `create_mask` make data in [iline, xline, height]
        format. Since most of the models percieve ilines as channels, it might be convinient
        to change format to [xlines, height, ilines] via this action.
        """
        crop_ = np.swapaxes(crop, 0, 1)
        crop_ = np.swapaxes(crop_, 1, 2)
        return crop_

    @apply_parallel
    def add_axis(self, crop):
        """ Add new axis.

        Notes
        -----
        Used in combination with `dice` and `ce` losses to tell model that input is
        3D entity, but 2D convolutions are used.
        """
        return crop[..., np.newaxis]

    @apply_parallel
    def additive_noise(self, crop, scale):
        """ Add random value to each entry of crop. Added values are centered at 0.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution.
        """
        rng = np.random.default_rng()
        noise = scale * rng.standard_normal(dtype=np.float32, size=crop.shape)
        return crop + noise

    @apply_parallel
    def multiplicative_noise(self, crop, scale):
        """ Multiply each entry of crop by random value, centered at 1.

        Parameters
        ----------
        scale : float
            Standart deviation of normal distribution.
        """
        rng = np.random.default_rng()
        noise = 1 + scale * rng.standard_normal(dtype=np.float32, size=crop.shape)
        return crop * noise

    @apply_parallel
    def cutout_2d(self, crop, patch_shape, n):
        """ Change patches of data to zeros.

        Parameters
        ----------
        patch_shape : array-like
            Shape or patches along each axis.
        n : float
            Number of patches to cut.
        """
        rnd = np.random.RandomState(int(n*100)).uniform
        patch_shape = patch_shape.astype(int)

        copy_ = copy(crop)
        for _ in range(int(n)):
            x_ = int(rnd(max(crop.shape[0] - patch_shape[0], 1)))
            h_ = int(rnd(max(crop.shape[1] - patch_shape[1], 1)))
            copy_[x_:x_+patch_shape[0], h_:h_+patch_shape[1], :] = 0
        return copy_

    @apply_parallel
    def rotate(self, crop, angle):
        """ Rotate crop along the first two axes.

        Parameters
        ----------
        angle : float
            Angle of rotation.
        """
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), angle, 1)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

    @apply_parallel
    def flip(self, crop, axis=0, seed=0.1, threshold=0.5):
        """ Flip crop along the given axis.

        Parameters
        ----------
        axis : int
            Axis to flip along
        """
        rnd = np.random.RandomState(int(seed*100)).uniform
        if rnd() >= threshold:
            return cv2.flip(crop, axis).reshape(crop.shape)
        return crop

    @apply_parallel
    def scale_2d(self, crop, scale):
        """ Zoom in or zoom out along the first two axes of crop.

        Parameters
        ----------
        scale : float
            Zooming factor.
        """
        shape = crop.shape
        matrix = cv2.getRotationMatrix2D((shape[1]//2, shape[0]//2), 0, scale)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(shape)

    @apply_parallel
    def affine_transform(self, crop, alpha_affine=10):
        """ Perspective transform. Moves three points to other locations.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_affine : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_affine*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_affine >= min(shape)//16:
            alpha_affine = min(shape)//16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size]])

        pts2 = pts1 + rnd(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)


        matrix = cv2.getAffineTransform(pts1, pts2)
        return cv2.warpAffine(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    @apply_parallel
    def perspective_transform(self, crop, alpha_persp):
        """ Perspective transform. Moves four points to other four.
        Guaranteed not to flip image or scale it more than 2 times.

        Parameters
        ----------
        alpha_persp : float
            Maximum distance along each axis between points before and after transform.
        """
        rnd = np.random.RandomState(int(alpha_persp*100)).uniform
        shape = np.array(crop.shape)[:2]
        if alpha_persp >= min(shape) // 16:
            alpha_persp = min(shape) // 16

        center_ = shape // 2
        square_size = min(shape) // 3

        pts1 = np.float32([center_ + square_size,
                           center_ - square_size,
                           [center_[0] + square_size, center_[1] - square_size],
                           [center_[0] - square_size, center_[1] + square_size]])

        pts2 = pts1 + rnd(-alpha_persp, alpha_persp, size=pts1.shape).astype(np.float32)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(crop, matrix, (shape[1], shape[0])).reshape(crop.shape)

    @apply_parallel
    def elastic_transform(self, crop, alpha=40, sigma=4):
        """ Transform indexing grid of the first two axes.

        Parameters
        ----------
        alpha : float
            Maximum shift along each axis.
        sigma : float
            Smoothening factor.
        """
        rng = np.random.default_rng(seed=int(alpha*100))
        shape_size = crop.shape[:2]

        grid_scale = 4
        alpha //= grid_scale
        sigma //= grid_scale
        grid_shape = (shape_size[0]//grid_scale, shape_size[1]//grid_scale)

        blur_size = int(4 * sigma) | 1
        rand_x = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        rand_y = cv2.GaussianBlur(rng.random(size=grid_shape, dtype=np.float32) * 2 - 1,
                                  ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
        if grid_scale > 1:
            rand_x = cv2.resize(rand_x, shape_size[::-1])
            rand_y = cv2.resize(rand_y, shape_size[::-1])

        grid_x, grid_y = np.meshgrid(np.arange(shape_size[1]), np.arange(shape_size[0]))
        grid_x = (grid_x.astype(np.float32) + rand_x)
        grid_y = (grid_y.astype(np.float32) + rand_y)

        distorted_img = cv2.remap(crop, grid_x, grid_y,
                                  borderMode=cv2.BORDER_REFLECT_101,
                                  interpolation=cv2.INTER_LINEAR)
        return distorted_img.reshape(crop.shape)

    @apply_parallel
    def bandwidth_filter(self, crop, lowcut=None, highcut=None, fs=1, order=3):
        """ Keep only frequences between lowcut and highcut.

        Notes
        -----
        Use it before other augmentations, especially before ones that add lots of zeros.

        Parameters
        ----------
        lowcut : float
            Lower bound for frequences kept.
        highcut : float
            Upper bound for frequences kept.
        fs : float
            Sampling rate.
        order : int
            Filtering order.
        """
        nyq = 0.5 * fs
        if lowcut is None:
            b, a = butter(order, highcut / nyq, btype='high')
        elif highcut is None:
            b, a = butter(order, lowcut / nyq, btype='low')
        else:
            b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
        return lfilter(b, a, crop, axis=1)

    @apply_parallel
    def sign(self, crop):
        """ Element-wise indication of the sign of a number. """
        return np.sign(crop)

    @apply_parallel
    def analytic_transform(self, crop, axis=1, mode='phase'):
        """ Compute instantaneous phase or frequency via the Hilbert transform.

        Parameters
        ----------
        axis : int
            Axis of transformation. Intended to be used after `rotate_axes`, so default value
            is to make transform along depth dimension.
        mode : str
            If 'phase', compute instantaneous phase.
            If 'freq', compute instantaneous frequency.
        """
        analytic = hilbert(crop, axis=axis)
        phase = np.unwrap(np.angle(analytic))

        if mode == 'phase':
            return phase
        if 'freq' in mode:
            return np.diff(phase, axis=axis, prepend=0) / (2*np.pi)
        raise ValueError('Unknown `mode` parameter.')

    @apply_parallel
    def gaussian_filter(self, crop, axis=1, sigma=2, order=0):
        """ Apply a gaussian filter along specified axis. """
        return gaussian_filter1d(crop, sigma=sigma, axis=axis, order=order)


    def plot_components(self, *components, idx=0, slide=None, mode='overlap', order_axes=None, **kwargs):
        """ Plot components of batch.

        Parameters
        ----------
        idx : int or None
            If int, then index of desired image in list.
            If None, then no indexing is applied.
        components : str or sequence of str
            Components to get from batch and draw.
        plot_mode : bool
            If 'overlap', then images are drawn one over the other with transparency.
            If 'separate', then images are drawn on separate layouts.
        order_axes : sequence of int
            Determines desired order of the axis. The first two are plotted.
        """
        if idx is not None:
            imgs = [getattr(self, comp)[idx] for comp in components]
        else:
            imgs = [getattr(self, comp) for comp in components]

        if slide is not None:
            imgs = [img[slide] for img in imgs]

        # set some defaults
        kwargs = {
            'label': 'Batch components',
            'titles': components,
            'xlabel': 'xlines',
            'ylabel': 'depth',
            'cmap': ['gray'] + ['viridis']*len(components) if mode == 'separate' else 'gray',
            **kwargs
        }

        plot_image(imgs, mode=mode, order_axes=order_axes, **kwargs)
