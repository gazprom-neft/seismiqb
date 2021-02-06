""" A convenient holder for horizon detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
    - evaluating predictions
    - and more
"""
#pylint: disable=import-error, no-name-in-module, wrong-import-position, protected-access
import os
import gc
import logging
import random
from copy import copy
from glob import glob
import psutil

import numpy as np
import torch

from tqdm.auto import tqdm

from ...batchflow import Pipeline, FilesIndex
from ...batchflow import B, V, C, D, P, R
from ...batchflow.models.torch import EncoderDecoder

from ..cubeset import SeismicCubeset, Horizon
from ..metrics import HorizonMetrics
from ..plotters import plot_loss, plot_image



class BaseController:
    """ Provides interface for train, inference and quality assesment for the task of horizon detection.

    Parameters
    ----------
    batch_size : int
        Size of batches for train and inference.
    crop_shape : tuple of 3 ints
        Size of sampled crops for train and inference.
    model_config : dict
        Neural network architecture.
    model_path : str
        Path for pre-trained model.
    device : str or int
        Device specification.
    show_plots : bool
        Whether to show plots to the current output stream.
    save_dir : str
        Path to save images, logs, and other data.
    logger : None or callable
        If None, then logger is created inside `save_dir`.
        If callable, then it is used directly to log messages.
    bar : bool
        Whether to show progress bars for training and inference.
    """
    #pylint: disable=unused-argument, logging-fstring-interpolation, no-member, too-many-public-methods
    #pylint: disable=access-member-before-definition, attribute-defined-outside-init
    def __init__(self, batch_size=64, crop_shape=(1, 256, 256),
                 model_config=None, model_path=None, device=None,
                 show_plots=False, save_dir=None, logger=None, bar=True):
        for key, value in locals().items():
            if key != 'self':
                setattr(self, key, value)

        self.targets, self.predictions = None, None
        self.model_pipeline = None
        self.make_logger()

    # Utility functions
    def make_pbar(self, iterator, ncols=800, **kwargs):
        """ Wrap supplied iterator with progress bar. """
        if self.bar:
            return tqdm(iterator, total=len(iterator), ncols=ncols, **kwargs)
        return iterator

    def make_save_path(self, *postfix):
        """ Create nested path from provided strings; create corresponding directories.

        If `save_dir` attribute is None, then None is returned: that is used as signal to omit saving
        of, for example, metric map images, etc.
        """
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, *postfix[:-1])
            os.makedirs(path, exist_ok=True)
            return os.path.join(self.save_dir, *postfix)
        return None

    def make_logger(self):
        """ Create logger inside `save_dir`.

        Note that logging is important.
        """
        #pylint: disable=access-member-before-definition
        if self.logger is None and self.save_dir is not None:
            handler = logging.FileHandler(self.make_save_path('controller.log'), mode='w')
            handler.setFormatter(logging.Formatter('%(asctime)s      %(message)s'))

            logger = logging.getLogger('controller_logger')
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)
            self.logger = logger.info

    def log(self, msg):
        """ Log supplied message. """
        if self.logger is not None:
            process = psutil.Process(os.getpid())
            uss = process.memory_full_info().uss / (1024 ** 3)
            self.logger(f'{self.__class__.__name__} ::: {uss:2.4} ::: {msg}')

    # Dataset creation: geometries, labels, grids, samplers
    def make_dataset(self, cube_paths, horizon_paths=None):
        """ Create an instance of :class:`.SeismicCubeset` with cubes and horizons.

        Parameters
        ----------
        cube_paths : sequence or str
            Cube path(s) to load into dataset.
        horizon_paths : dict or str
            Horizons for each cube. Either a mapping from cube name to paths, or path only (if only one cube is used).

        Logs
        ----
        Inferred cubes and horizons for them.

        Returns
        -------
        Instance of dataset.
        """
        cube_paths = cube_paths if isinstance(cube_paths, (tuple, list)) else [cube_paths]

        dsi = FilesIndex(path=cube_paths, no_ext=True)
        dataset = SeismicCubeset(dsi)

        dataset.load_geometries()

        if horizon_paths:
            if isinstance(horizon_paths, str):
                horizon_paths = {dataset.indices[0]: glob(horizon_paths)}
            dataset.create_labels(horizon_paths)

        msg = '\n'
        for idx in dataset.indices:
            msg += f'{idx}\n'
            for hor in dataset.labels[idx]:
                msg += f'    {hor.name}'
        self.log(f'Created dataset ::: {msg}')
        return dataset

    def make_dataset_from_horizon(self, horizon):
        """ Create an instance of :class:`.SeismicCubeset` from a given horizon.

        Parameters
        ----------
        horizon : instance of :class:`.Horizon`
            Horizon for the inferred cube.
        """
        cube_path = horizon.geometry.path

        dsi = FilesIndex(path=[cube_path], no_ext=True)
        dataset = SeismicCubeset(dsi)
        dataset.geometries[dataset.indices[0]] = horizon.geometry
        dataset.labels[dataset.indices[0]] = [horizon]

        self.log(f'Created dataset from horizon {horizon.name}')
        return dataset


    def make_grid(self, dataset, frequencies, **kwargs):
        """ Create a grid, based on quality map, for each of the cubes in supplied `dataset`.
        Works inplace.

        Parameters
        ----------
        dataset : :class:`.SeismicCubeset`
            Dataset with cubes.
        frequencies : sequence of ints
            List of frequencies, corresponding to `easy` and `hard` places in the cube.
        kwargs : dict
            Other arguments, passed directly in quality grid creation function.

        Logs
        ----
        Grid coverage: ratio of the number of points inside the grid to the total number of non-bad traces in cube.

        Plots
        -----
        Map with quality grid.
        """
        grid_coverages = []
        for idx in dataset.indices:
            geometry = dataset.geometries[idx]
            geometry.make_quality_grid(frequencies, **kwargs)
            plot_image(
                geometry.quality_grid, title='quality grid',
                cmap='Reds', interpolation='bilinear', show=self.show_plots,
                savepath=self.make_save_path(f'quality_grid_{idx}.png')
            )

            grid_coverage = (np.nansum(geometry.quality_grid) /
                             (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
            self.log(f'Created grid on {idx}; coverage is: {grid_coverage}')
            grid_coverages.append(grid_coverage)
        return grid_coverages


    def make_sampler(self, dataset, bins=None, use_grid=False, grid_src='quality_grid', side_view=False, **kwargs):
        """ Create sampler. Works inplace.

        Plots
        -----
        Maps with examples of sampled slices of `crop_shape` size, both normalized and not.
        """
        if use_grid:
            grid = getattr(dataset.geometries[0], grid_src) if isinstance(grid_src, str) else grid_src
        else:
            grid = None

        dataset.create_sampler(quality_grid=grid, bins=bins)
        dataset.modify_sampler('train_sampler', finish=True, **kwargs)
        dataset.train_sampler(random.randint(0, 100000))
        self.log('Created sampler')

        # Cleanup
        dataset.sampler = None

        for i, idx in enumerate(dataset.indices):
            dataset.show_slices(
                src_sampler='train_sampler', normalize=False, shape=self.crop_shape,
                idx=i, adaptive_slices=use_grid, grid_src=grid_src, side_view=side_view,
                cmap='Reds', interpolation='bilinear', show=self.show_plots, figsize=(15, 15),
                savepath=self.make_save_path(f'slices_{idx}.png')
            )

            dataset.show_slices(
                src_sampler='train_sampler', normalize=True, shape=self.crop_shape,
                idx=i, adaptive_slices=use_grid, grid_src=grid_src, side_view=side_view,
                cmap='Reds', interpolation='bilinear', show=self.show_plots, figsize=(15, 15),
                savepath=self.make_save_path(f'slices_n_{idx}.png')
            )

    # Train model on a created dataset
    def train(self, dataset, model_config=None, device=None, n_iters=300, prefetch=1,
              use_grid=False, grid_src='quality_grid', side_view=False,
              width=3, batch_size_multiplier=1, rebatch_threshold=0.00, **kwargs):
        """ Train model for horizon detection.
        If `model_path` was supplied during instance initialization, model is loaded instead.

        In order to change architecture of the model, pass different `model_config` to the instance initialization.
        In order to change training procedure, re-define :meth:`.get_train_template`.

        Parameters
        ----------
        n_iters : int
            Number of iterations to train for.
        use_grid : bool
            Whether to sample crops only from `quality_grid`.
        side_view : bool or float
            If False, then has no effect.
            If float, then probability of crop being sampled along `x` axis instead of regular `i`-axis sampling.
            If True, then the same as 0.5.

        Logs
        ----
        Start of training; end of training; average loss at the last 50 iterations.

        Plots
        -----
        Graph of loss over iterations.
        """
        model_config = model_config or self.model_config
        device = device or self.device

        self.log('Train started')
        pipeline_config = {
            'model_config': {**model_config, 'device': device},
            'crop_shape': self.crop_shape,
            'adaptive_slices': use_grid, 'grid_src': grid_src,
            'side_view': side_view,
            'width': width,
            'rebatch_threshold': rebatch_threshold,
            **kwargs
        }

        bs = self.batch_size
        self.batch_size = int(self.batch_size * batch_size_multiplier)
        model_pipeline = (self.get_train_template(**kwargs) << pipeline_config) << dataset
        batch = model_pipeline.next_batch(D('size'))
        self.log(f'Used batch size is: {self.batch_size}; actual batch size is: {len(batch)}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')
        self.batch_size = bs

        model_pipeline.run(D('size'), n_iters=n_iters + np.random.randint(100),
                           bar={'bar': self.bar, 'monitors': 'loss_history'}, prefetch=prefetch)
        plot_loss(model_pipeline.v('loss_history'), show=self.show_plots,
                  savepath=self.make_save_path('model_loss.png'))

        self.model_pipeline = model_pipeline
        last_loss = np.mean(model_pipeline.v('loss_history')[-50:])
        self.log(f'Train finished; last loss is {last_loss}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[len(item._cached_load.cache()) for item in dataset.geometries.values()]}')

        # Cleanup
        torch.cuda.empty_cache()
        self.model_pipeline.reset('variables')
        batch.images, batch.masks = None, None
        for item in dataset.geometries.values():
            item.reset_cache()
        return last_loss

    def load_model(self, path=None):
        """ Load pre-trained model from disk. """
        path = path or self.model_path
        raise NotImplementedError('Yet to be implemented!')

    # Inference on a chosen set of data
    def inference(self, dataset, version=1, orientation='i', overlap_factor=2, heights_range=None,
                  batch_size_multiplier=1, **kwargs):
        """ Make inference with trained/loaded model on supplied dataset.
        Works by splitting the into `crop_shape` chunks, making predict for each of them,
        then aggregating into one horizon.

        Parameters
        ----------
        version : int
            How to do splitting:
            If 0, then cube is split into chunks of `crop_shape` size,
            model is used to create predictions for each of them,
            then chunks are aggregated into huge 3D array, from which the horizon surface is extracted.
            This approach is fast but very memory intensive: it is advised to use it only on small (<10GB) cubes.

            If 1, then cube is split into `big` chunks, each of them is split again into `crop_shape` pieces,
            model is used to create predictions for the latter,
            which are aggregated into 3D array of `big` chunks size, from which the horizon surfaces are extracted.
            At last, all of the horizons are merged into one.
            This approach is a tad slower, yet allows for finer memory control by controlling how big `big` chunks are.
            Additional parameters are:
            chunk_size : int
                Size of `big` chunks along smallest dimension.
            chunk_overlap : float
                Overlap percentage of successive chunks. Must be in 0, 1 range.

        orientation : {'i', 'x', 'ix'}
            Orientation of the inference:
            If 'i', then cube is split into inline-oriented slices.
            If 'x', then cube is split into crossline-oriented slices.
            If 'ix', then both of previous approaches applied, and results are merged.
        overlap_factor : number
            Overlapping ratio of successive crops. Can be seen as `how many crops would cross every through point`.
        heights_range : None or sequence of two ints
            If None, then heights are inffered: from minimum of heights of all horizons in dataset to the maximum.
            If sequence of two ints, heights to inference on.

        Logs
        ----
        Inference start along with its parameters; inference end along with the number of predicted horizons,
        total amount of predicted points and size of the biggest horizon.
        """
        self.log(f'Starting {orientation} inference_{version} with overlap of {overlap_factor}')
        self.targets = dataset.labels[0]
        method = getattr(self, f'inference_{version}')

        bs = self.batch_size
        self.batch_size = int(self.batch_size * batch_size_multiplier)

        if len(orientation) == 1:
            horizons = method(dataset, orientation=orientation, overlap_factor=overlap_factor,
                              heights_range=heights_range, **kwargs)
        else:
            horizons_i = method(dataset, orientation='i', overlap_factor=overlap_factor,
                                heights_range=heights_range, **kwargs)
            gc.collect()
            self.log('Done i-inference')

            horizons_x = method(dataset, orientation='x', overlap_factor=overlap_factor,
                                heights_range=heights_range, **kwargs)
            gc.collect()
            self.log('Done x-inference')

            horizons = Horizon.merge_list(horizons_i + horizons_x, minsize=1000)
            gc.collect()

        # Log some results
        if horizons:
            horizons.sort(key=len, reverse=True)
            self.log(f'Num of predicted horizons: {len(horizons)}')
            self.log(f'Total number of points in all of the horizons {sum(len(item) for item in horizons)}')
            self.log(f'Len max: {len(horizons[0])}')
        else:
            self.log('Zero horizons were predicted; possible problems..?')

        self.predictions = horizons
        self.batch_size = bs
        torch.cuda.empty_cache()

    def make_inference_ranges(self, dataset, heights_range):
        """ Ranges of inference. """
        geometry = dataset.geometries[0]
        spatial_ranges = [[0, item-1] for item in geometry.cube_shape[:2]]
        if heights_range is None:
            if self.targets:
                min_height = max(0,
                                 min(horizon.h_min for horizon in self.targets) - self.crop_shape[2]//2)
                max_height = min(geometry.depth,
                                 max(horizon.h_max for horizon in self.targets) + self.crop_shape[2]//2)
                heights_range = [min_height, max_height]
            else:
                heights_range = [0, geometry.depth-1]
        return spatial_ranges, heights_range

    def make_inference_config(self, orientation):
        """ Parameters depending on orientation. """
        config = {'model_pipeline': self.model_pipeline}
        if orientation == 'i':
            crop_shape_grid = self.crop_shape
            config['side_view'] = False
            config['order'] = (0, 1, 2)
        else:
            crop_shape_grid = np.array(self.crop_shape)[[1, 0, 2]]
            config['side_view'] = 1.0
            config['order'] = (1, 0, 2)
        return config, crop_shape_grid


    def inference_0(self, dataset, heights_range=None, orientation='i', overlap_factor=2,
                    filtering_matrix=None, filter_threshold=0, prefetch=1, **kwargs):
        """ Inference on chunks, assemble into massive 3D array, extract horizon surface. """
        _ = kwargs
        spatial_ranges, heights_range = self.make_inference_ranges(dataset, heights_range)
        config, crop_shape_grid = self.make_inference_config(orientation)

        # Actual inference
        dataset.make_grid(dataset.indices[0], crop_shape_grid,
                          *spatial_ranges, heights_range,
                          batch_size=self.batch_size,
                          overlap_factor=overlap_factor,
                          filtering_matrix=filtering_matrix,
                          filter_threshold=filter_threshold)

        inference_pipeline = (self.get_inference_template() << config) << dataset
        inference_pipeline.run(D('size'), n_iters=dataset.grid_iters, bar=self.bar,
                               prefetch=prefetch)

        # Assemble crops together in accordance to the created grid
        assembled_pred = dataset.assemble_crops(inference_pipeline.v('predicted_masks'),
                                                order=config.get('order'))

        # Log memory usage info and clean up
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')

        inference_pipeline.reset('variables')
        inference_pipeline = None
        for item in dataset.geometries.values():
            item.reset_cache()
        gc.collect()

        # Convert to Horizon instances
        return Horizon.from_mask(assembled_pred, dataset.grid_info, threshold=0.5, minsize=50)

    def inference_1(self, dataset, heights_range=None, orientation='i', overlap_factor=2, prefetch=1,
                    chunk_size=100, chunk_overlap=0.2, filtering_matrix=None, filter_threshold=0, **kwargs):
        """ Split area for inference into `big` chunks, inference on each of them, merge results. """
        _ = kwargs
        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = self.make_inference_ranges(dataset, heights_range)
        config, crop_shape_grid = self.make_inference_config(orientation)

        # Actual inference
        axis = np.argmin(crop_shape_grid[:2])
        iterator = range(spatial_ranges[axis][0], spatial_ranges[axis][1], int(chunk_size*(1 - chunk_overlap)))
        self.log(f'Starting chunk {orientation} inference with {len(iterator)} chunks ' +
                 f'over {spatial_ranges}, {heights_range}')

        horizons = []
        total_length, total_unfiltered_length = 0, 0
        for chunk in self.make_pbar(iterator, desc=f'Inference on {geometry.name}| {orientation}'):
            current_spatial_ranges = copy(spatial_ranges)
            current_spatial_ranges[axis] = [chunk, min(chunk + chunk_size, spatial_ranges[axis][-1])]

            dataset.make_grid(dataset.indices[0], crop_shape_grid,
                              *current_spatial_ranges, heights_range,
                              batch_size=self.batch_size,
                              overlap_factor=overlap_factor,
                              filtering_matrix=filtering_matrix,
                              filter_threshold=filter_threshold)
            total_length += dataset.grid_info['length']
            total_unfiltered_length += dataset.grid_info['unfiltered_length']

            inference_pipeline = (self.get_inference_template() << config) << dataset
            inference_pipeline.run(D('size'), n_iters=dataset.grid_iters, prefetch=prefetch)

            # Assemble crops together in accordance to the created grid
            assembled_pred = dataset.assemble_crops(inference_pipeline.v('predicted_masks'),
                                                    order=config.get('order'))

            # Extract Horizon instances
            chunk_horizons = Horizon.from_mask(assembled_pred, dataset.grid_info, threshold=0.5, minsize=50)
            horizons.extend(chunk_horizons)

            # Cleanup
            inference_pipeline.reset('variables')
            inference_pipeline = None
            gc.collect()

        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')
        self.log(f'Inferenced total of {total_length} out of {total_unfiltered_length} crops possible')
        for item in dataset.geometries.values():
            item.reset_cache()
        gc.collect()

        return Horizon.merge_list(horizons, mean_threshold=5.5, adjacency=3, minsize=500)


    def evaluate(self, n=5, add_prefix=False, dump=False, supports=50, name=''):
        """ Assess quality of predictions, created by :meth:`.inference`, against targets and seismic data.

        Parameters
        ----------
        n : int
            Number of the best horizons to evaluate.
        add_prefix : bool
            Whether to add add prefix to created images and other files.
        dump : bool
            Whether to store horizons on disk.
        supports : int
            Number of support traces for metric computation.

        Logs
        ----
        Basic stats like coverage, size, number of holes.
        If targets are provided, adds `window_rate` and mean difference.

        Plots
        -----
        Maps of computed metrics: correlation, local correlation.
        If targets are provided, also l1 differences.
        """
        #pylint: disable=cell-var-from-loop, invalid-name, protected-access
        results = []
        for i in range(n):
            info = {}
            horizon = self.predictions[i]
            horizon._horizon_metrics = None
            hm = HorizonMetrics((horizon, self.targets))
            prefix = [horizon.geometry.short_name, f'{i}_horizon'] if add_prefix else []

            # Basic demo: depth map and properties
            horizon.show(show=self.show_plots,
                         savepath=self.make_save_path(*prefix, name + 'horizon_img.png'))

            with open(self.make_save_path(*prefix, name + 'self_results.txt'), 'w') as result_txt:
                horizon.evaluate(compute_metric=False, printer=lambda msg: print(msg, file=result_txt))

            # Correlations
            corrs = hm.evaluate(
                'support_corrs',
                supports=supports,
                plot=True, show_plot=self.show_plots,
                savepath=self.make_save_path(*prefix, name + 'corrs.png')
            )

            phase = hm.evaluate(
                'instantaneous_phase',
                plot=True, show_plot=self.show_plots,
                savepath=self.make_save_path(*prefix, name + 'instantaneous_phase.png')
            )

            # Compare to targets
            if self.targets:
                _, oinfo = hm.evaluate('find_best_match', agg=None)
                info = {**info, **oinfo}

                with open(self.make_save_path(*prefix, name + 'results.txt'), 'w') as result_txt:
                    hm.evaluate(
                        'compare', agg=None, hist=False,
                        plot=True, show_plot=self.show_plots,
                        printer=lambda msg: print(msg, file=result_txt),
                        savepath=self.make_save_path(*prefix, name + 'l1.png')
                    )
                self.log(f'horizon {i}: wr {info["window_rate"]}, mean {info["mean"]}')

            # Save surface to disk
            if dump:
                dump_name = name + '_' if name else ''
                dump_name += f'{i}_' if n > 1 else ''
                dump_name += horizon.name or 'predicted'
                horizon.dump(path=self.make_save_path(*prefix, dump_name), add_height=False)

            info['corrs'] = np.nanmean(corrs)
            info['phase'] = np.nanmean(np.abs(phase))
            results.append((info))

            self.log(f'horizon {i}: len {len(horizon)}, cov {horizon.coverage:4.4}, '
                     f'corrs {info["corrs"]:4.4}, phase {info["phase"]:4.4}, depth {horizon.h_mean}')

        return results

    # Pipelines
    def load_pipeline(self, dynamic_factor=1, dynamic_low=None, dynamic_high=None, **kwargs):
        """ Define data loading pipeline.

        Following parameters are fetched from pipeline config: `adaptive_slices`, 'grid_src' and `rebatch_threshold`.
        """
        _ = kwargs
        self.log(f'Generating data with dynamic factor of {dynamic_factor}')
        return (
            Pipeline()
            .init_variable('shape', None)
            .call(generate_shape, shape=C('crop_shape'),
                  dynamic_factor=dynamic_factor, dynamic_low=dynamic_low, dynamic_high=dynamic_high,
                  save_to=V('shape'))
            .make_locations(points=D('train_sampler')(self.batch_size),
                            shape=V('shape'),
                            side_view=C('side_view', default=False),
                            adaptive_slices=C('adaptive_slices'),
                            grid_src=C('grid_src', default='quality_grid'))

            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.1))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'], shape=V('shape'))
            .normalize(mode='q', src='images')
        )

    def augmentation_pipeline(self, **kwargs):
        """ Define augmentation pipeline. """
        _ = kwargs
        return (
            Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -15, 15)),
                    src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', 0.85, 1.15)),
                      src=['images', 'masks'], p=0.3)
            .elastic_transform(alpha=P(R('uniform', 35, 45)), sigma=P(R('uniform', 4, 4.5)),
                               src=['images', 'masks'], p=0.2)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
        )

    def train_pipeline(self, **kwargs):
        """ Define model initialization and model training pipeline.

        Following parameters are fetched from pipeline config: `model_config`.
        """
        _ = kwargs
        return (
            Pipeline()
            .init_variable('loss_history', [])
            .init_model('dynamic', EncoderDecoder, 'model', C('model_config'))

            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         save_to=V('loss_history', mode='a'))
        )

    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        return (
            self.load_pipeline(**kwargs) +
            self.augmentation_pipeline(**kwargs) +
            self.train_pipeline(**kwargs)
        )


    def get_inference_template(self):
        """ Defines inference procedure.

        Following parameters are fetched from pipeline config: `model_pipeline`, `crop_shape`, `side_view` and `order`.
        """
        inference_template = (
            Pipeline()
            # Initialize everything
            .init_variable('predicted_masks', [])
            .import_model('model', C('model_pipeline'))

            # Load data
            .make_locations(points=D('grid_gen')(), shape=self.crop_shape,
                            side_view=C('side_view', default=False))
            .load_cubes(dst='images')
            .adaptive_reshape(src='images', shape=self.crop_shape)
            .normalize(mode='q', src='images')

            # Predict with model, then aggregate
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=V('predicted_masks', mode='e'))
        )
        return inference_template


def generate_shape(_, shape, dynamic_factor=1, dynamic_low=None, dynamic_high=None):
    """ Dynamically generate shape of a crop to get. """
    dynamic_low = dynamic_low or dynamic_factor
    dynamic_high = dynamic_high or dynamic_factor

    i, x, h = shape
    x_ = np.random.randint(x // dynamic_low, x * dynamic_high + 1)
    h_ = np.random.randint(h // dynamic_low, h * dynamic_high + 1)
    return (i, x_, h_)
