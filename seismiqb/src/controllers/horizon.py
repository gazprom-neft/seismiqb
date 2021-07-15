""" A convenient holder for horizon detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
    - evaluating predictions
    - and more
"""
#pylint: disable=import-error, no-name-in-module, wrong-import-position, protected-access
import gc

from time import perf_counter
from textwrap import indent
from pprint import pformat
from glob import glob

import numpy as np
import torch

from ...batchflow import Config, Pipeline, Monitor, Notifier
from ...batchflow import B, C, V, P, R
from ...batchflow.models.torch import EncoderDecoder

from ..cubeset import SeismicCubeset, Horizon
from ..samplers import SeismicSampler, RegularGrid
from ..metrics import HorizonMetrics
from ..plotters import plot_image
from ..utility_classes import Accumulator3D

from .base import BaseController


class HorizonController(BaseController):
    """ Controller for horizon detection tasks. """
    #pylint: disable=attribute-defined-outside-init
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # General parameters
        'savedir': None,
        'monitor': True,
        'logger': None,
        'bar': False,
        'plot': False,

        'train': {
            'model_class': EncoderDecoder,
            'model_config': None,

            'batch_size': None,
            'crop_shape': None,
            'width': 3,

            'rebatch_threshold': 0.8,
            'rescale_batch_size': True,

            'prefetch': 1,
            'n_iters': 100,
            'early_stopping': True,
        },
        'finetune': {
            'n_iters': 10,
        },
        'inference': {
            'orientation': 'ix',
            'batch_size': None,
            'crop_shape': None,
            'width': 3,

            # Grid making parameters
            'spatial_ranges': None,
            'heights_range': None,
            'overlap_factor': 2,
            'threshold': 0.,

            'prefetch': 0,
            'chunk_size': 100,
            'chunk_overlap': 0.1,

            'finetune': True,
        },

        # Common parameters for train and inference
        'common': {},

        # Make predictions smoother
        'postprocess': {},

        # Compute metrics
        'evaluate': {
            'supports': 100,
            'device': 'gpu',

            'add_prefix': False,
            'dump': False,
            'name': '',
            'rename': False,
        }
    })

    # Dataset creation: geometries, labels, grids, samplers
    def make_dataset(self, cube_paths=None, horizon_paths=None, horizon=None):
        """ Create an instance of :class:`.SeismicCubeset` with cubes and horizons.

        Parameters
        ----------
        cube_paths : sequence or str
            Cube path(s) to load into dataset.
        horizon_paths : dict or str
            Horizons for each cube. Either a mapping from cube name to paths, or path only (if only one cube is used).
        horizon : None or Horizon
            If supplied, then the dataset is initialized with a single horizon.

        Returns
        -------
        Instance of dataset.
        """
        if horizon is not None:
            dataset = SeismicCubeset.from_horizon(horizon)

            self.log(f'Created dataset from {horizon.name}')
        else:
            dataset = SeismicCubeset(cube_paths)
            dataset.load_geometries()

            if horizon_paths:
                if isinstance(horizon_paths, str):
                    horizon_paths = {dataset.indices[0]: glob(horizon_paths)}
                dataset.create_labels(horizon_paths, labels_class=Horizon)

            self.log(f'Created dataset\n{indent(str(dataset), " "*4)}')
        return dataset

    def make_carcass(self, dataset, frequencies=200, **kwargs):
        """ Cut a grid from the passed horizon. """
        carcass = dataset.labels[0][0].make_carcass(frequencies=frequencies, **kwargs)
        return carcass, SeismicCubeset.from_horizon(carcass)

    def make_grid(self, dataset, frequencies, **kwargs):
        """ Create a grid, based on quality map, for each of the cubes in supplied `dataset`. Works inplace.

        Parameters
        ----------
        dataset : :class:`.SeismicCubeset`
            Dataset with cubes.
        frequencies : sequence of ints
            List of frequencies, corresponding to `easy` and `hard` places in the cube.
        kwargs : dict
            Other arguments, passed directly in quality grid creation function.
        """
        for idx in dataset.indices:
            geometry = dataset.geometries[idx]
            geometry.make_quality_grid(frequencies, **kwargs)

            postfix = f'_{idx}' if len(dataset.indices) > 1 else ''
            plot_image(
                geometry.quality_grid, title=f'Quality grid for {idx}',
                show=self.plot, cmap='Reds', interpolation='bilinear',
                savepath=self.make_savepath(f'quality_grid{postfix}.png')
            )

            grid_coverage = (np.nansum(geometry.quality_grid) /
                             (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
            self.log(f'Created {frequencies} grid on {idx}; coverage is: {grid_coverage:4.4f}')

    def make_sampler(self, dataset, **kwargs):
        """ Create sampler for generating locations to train on. """
        crop_shape = self.config['train']['crop_shape']
        rebatch_threshold = self.config['train']['rebatch_threshold']
        sampler = SeismicSampler(labels=dataset.labels, crop_shape=crop_shape,
                                 threshold=rebatch_threshold, mode='horizon', **kwargs)
        return sampler

    # Train method is inherited from BaseController class

    # Inference
    def inference(self, dataset, model, config=None, **kwargs):
        """ Make inference on a supplied dataset with a provided model.

        Works by making inference on chunks, splitted into crops.
        Resulting predictions (horizons) are stitched together.
        """
        # Prepare parameters
        config = config or {}
        config = Config({**self.config['common'], **self.config['inference'], **config, **kwargs})
        orientation = config.pop('orientation')
        self.log(f'Starting {orientation} inference')

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config.config, depth=2), '末 inference_config.txt')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
            monitor.__enter__()

        horizons = []
        largest = []

        start_time = perf_counter()
        for letter in orientation:
            horizons_ = self._inference(dataset=dataset, model=model,
                                        orientation=letter, config=config)
            self.log(f'Done {letter}-inference')
            horizons.extend(horizons_)
            largest.append(horizons_[0])
        elapsed = perf_counter() - start_time

        # Compare two largest horizons from each orientation
        if len(orientation) == 2:
            with open(self.make_savepath('inference_ix', 'results.txt'), 'w') as result_txt:
                hm = HorizonMetrics(largest)
                hm.evaluate('compare', hist=False,
                            plot=True, show=self.plot,
                            printer=lambda msg: print(msg, file=result_txt),
                            savepath=self.make_savepath('inference_ix', 'l1.png'))

        # Merge all the predictions
        horizons = Horizon.merge_list(horizons, minsize=1000)
        self.log(f'Inference done in {elapsed:4.1f}')

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)

        # Log: lengths of predictions
        if horizons:
            horizons.sort(key=len, reverse=True)
            self.log(f'Num of predicted horizons: {len(horizons)}')
            self.log(f'Total number of points in all of the horizons {sum(len(item) for item in horizons)}')
            self.log(f'Len max: {len(horizons[0])}')
        else:
            self.log('Zero horizons were predicted; possible problems..?')

        self.inference_log = {
            'elapsed': elapsed,
        }
        return horizons

    def _inference(self, dataset, model, orientation, config):
        # Prepare parameters
        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = config.get(['spatial_ranges', 'heights_range'])
        chunk_size, chunk_overlap = config.get(['chunk_size', 'chunk_overlap'])

        # Make spatial and height ranges
        if spatial_ranges is None:
            spatial_ranges = [None, None]

        if heights_range is None:
            bases = dataset.labels[0]
            if bases:
                min_height = min(horizon.h_min for horizon in bases) - config.crop_shape[2]//2
                max_height = max(horizon.h_max for horizon in bases) + config.crop_shape[2]//2
                heights_range = [max(0, min_height), min(geometry.depth, max_height)]
            else:
                heights_range = [0, geometry.depth]

        # Create grid
        ranges = [*spatial_ranges, heights_range]
        grid = RegularGrid(geometry=geometry,
                           ranges=ranges,
                           orientation=orientation,
                           crop_shape=config.crop_shape,
                           threshold=config.threshold,
                           batch_size=config.batch_size,
                           overlap_factor=config.overlap_factor)

        # Make iterator of chunks over given axis
        chunk_iterator = grid.to_chunks(size=chunk_size, overlap=chunk_overlap)

        # Actual inference
        self.log(f'Starting {orientation}-inference with {len(chunk_iterator)} chunks')
        self.log(f'Inference over {spatial_ranges}, {heights_range}')
        notifier = Notifier(self.config.bar,
                            desc=f'{orientation}-inference', update_total=False,
                            file=self.make_savepath(f'末 inference_chunks_{orientation}.log'))
        chunk_iterator = notifier(chunk_iterator)

        horizons = []
        for i, grid_chunk in enumerate(chunk_iterator):
            horizons_ = self._inference_on_chunk(dataset=dataset, model=model, grid_chunk=grid_chunk,
                                                 config=config, iteration=i)
            horizons.extend(horizons_)

        self.log(f'Inferenced total of {grid.length} out of {grid.unfiltered_length} crops possible')

        # Cleanup
        for item in dataset.geometries.values():
            item.reset_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return Horizon.merge_list(horizons, mean_threshold=5.5, adjacency=3, minsize=500)

    def _inference_on_chunk(self, dataset, model, grid_chunk, config, iteration):
        # Prepare parameters
        prefetch = config.get('prefetch', 0)

        if config.get('finetune', True):
            sampler = self.make_sampler(dataset, ranges=grid_chunk.ranges,)
            if len(sampler) > 0:
                sampler.show_locations(show=self.plot,
                                       savepath=self.make_savepath('chunk', str(iteration), 'sampler_locations.png'))
                sampler.show_sampled(show=self.plot,
                                     savepath=self.make_savepath('chunk', str(iteration), 'sampler_generated.png'))

                self.finetune(dataset=dataset, sampler=sampler, model=model)
                self.log(f'Finetuned! {model.iteration}')

        # Create regular grid over desired ranges
        geometry = dataset.geometries[0]
        config['grid'] = grid_chunk

        # Create grid over chunk ranges
        accumulator = Accumulator3D.from_grid(grid=grid_chunk, aggregation='max', fill_value=0.0)
        config['accumulator'] = accumulator

        # Create pipeline TODO: make better `add_model`
        inference_pipeline = self.get_inference_template() << config << dataset
        inference_pipeline.models.add_model('model', model)

        # Make predictions over chunk
        inference_pipeline.run(n_iters=grid_chunk.n_iters, prefetch=prefetch)
        assembled_prediction = accumulator.aggregate()

        # Extract Horizon instances
        horizons = Horizon.from_mask(assembled_prediction,
                                     geometry=geometry, shifts=grid_chunk.origin,
                                     threshold=0.5, minsize=50)
        # Cleanup
        gc.collect()
        accumulator.clear()
        inference_pipeline.reset('variables')
        return horizons

    # Postprocess
    def postprocess(self, predictions, **kwargs):
        """ Modify predictions. """
        config = Config({**self.config['postprocess'], **kwargs})
        _ = config

        iterator = predictions if isinstance(predictions, (tuple, list)) else [predictions]
        for horizon in iterator:
            horizon.smooth_out(preserve_borders=False)
            horizon.filter()
        return predictions

    # Evaluate
    def evaluate(self, predictions, targets=None, dataset=None, config=None, **kwargs):
        """ Assess quality of predictions against targets and seismic data. """
        #pylint: disable=cell-var-from-loop
        # Prepare parameters
        config = config or {}
        config = Config({**self.config['evaluate'], **config, **kwargs})
        add_prefix, dump, name, rename = config.pop(['add_prefix', 'dump', 'name', 'rename'])
        supports, device = config.pop(['supports', 'device'])

        if targets is None and dataset is not None:
            targets = dataset.labels[0]

        predictions = predictions if isinstance(predictions, (tuple, list)) else [predictions]
        n = len(predictions)

        results = []
        for i, horizon in enumerate(predictions):
            info = {}
            prefix = [horizon.geometry.short_name, f'{i}_horizon'] if add_prefix else []

            # Basic demo: depth map and properties
            horizon.show(show=self.plot, savepath=self.make_savepath(*prefix, name + 'p_depth_map.png'))

            horizon.show_slide(horizon.geometry.lens[0]//2, axis=0, show=self.plot,
                               savepath=self.make_savepath(*prefix, name + 'p_slide_i.png'))
            horizon.show_slide(horizon.geometry.lens[1]//2, axis=1, show=self.plot,
                               savepath=self.make_savepath(*prefix, name + 'p_slide_x.png'))

            with open(self.make_savepath(*prefix, name + 'p_results_self.txt'), 'w') as result_txt:
                horizon.evaluate(compute_metric=False, printer=lambda msg: print(msg, file=result_txt))

            # Metric maps
            hm = HorizonMetrics((horizon, targets))
            corrs = hm.evaluate('support_corrs', supports=supports, device=device,
                                plot=True, show=self.plot,
                                savepath=self.make_savepath(*prefix, name + 'p_corrs.png'))

            phase = hm.evaluate('instantaneous_phase', device=device,
                                plot=True, show=self.plot,
                                savepath=self.make_savepath(*prefix, name + 'p_instantaneous_phase.png'))

            perturbed_mean, perturbed_max = hm.evaluate('perturbed', device=device,
                                                        plot=True, show=self.plot,
                                                        savepath=self.make_savepath(*prefix, name + 'p_perturbed.png'))

            # Compare to targets
            shift = 41 + len(self.__class__.__name__)
            if targets:
                other, _info = hm.evaluate('find_best_match', agg=None)
                info = {**info, **_info}

                with open(self.make_savepath(*prefix, name + 'p_results.txt'), 'w') as result_txt:
                    hm.evaluate('compare', hist=False,
                                plot=True, show=self.plot,
                                printer=lambda msg: print(msg, file=result_txt),
                                savepath=self.make_savepath(*prefix, name + 'l1.png'))

                msg = (f'\nPredicted horizon {i} compared to target:'
                       f'\n{horizon.name}'
                       f'\nwindow_rate={info["window_rate"]:4.3f}\navg error={info["mean"]:4.3f}')
                self.log(indent(msg, ' '*shift))

                if rename:
                    horizon.name = f'from_{other.name}'
                    self.log(f'Renamed to {horizon.name}')

            # Save surface to disk
            if dump:
                dump_name = name + '_' if name else ''
                dump_name += f'{i}_' if n > 1 else ''
                dump_name += horizon.name or 'predicted'
                horizon.dump_float(path=self.make_savepath(*prefix, dump_name), add_height=False)

            info['corrs'] = np.nanmean(corrs)
            info['phase'] = np.nanmean(np.abs(phase))
            info['perturbed_mean'] = np.nanmean(perturbed_mean)
            info['perturbed_max'] = np.nanmean(perturbed_max)
            results.append((info))

            msg = (f'\nPredicted horizon {i}:\n{horizon.name}\nlen={len(horizon)}'
                   f'\ncoverage={horizon.coverage:4.3f}\ncorrs={info["corrs"]:4.3f}'
                   f'\nphase={info["phase"]:4.3f}\navg depth={horizon.h_mean:4.3f}')
            self.log(indent(msg, ' '*shift))
            self.log(f'Done evaluation for {horizon.name}')
        return results

    # Pipelines
    def load_pipeline(self, **kwargs):
        """ Define data loading pipeline.

        Following parameters are fetched from pipeline config: `sampler`, `batch_size` `width` and `rebatch_threshold`.
        """
        _ = kwargs
        return (
            Pipeline()
            .make_locations(generator=C('sampler'), batch_size=C('batch_size'))

            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.1))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'])
            .normalize(src='images')
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

        Following parameters are fetched from pipeline config: `model_class` and `model_config`.
        """
        _ = kwargs
        return (
            Pipeline()
            .init_variable('loss_history', [])
            .init_model(mode='dynamic', model_class=C('model_class', default=EncoderDecoder),
                        name='model', config=C('model_config'), source=C('source_model', default=None))

            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         microbatch=C('microbatch', default=True),
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

        Following parameters are fetched from pipeline config: `crop_shape` and `side_view`.
        """
        inference_template = (
            Pipeline()

            # Load data
            .make_locations(generator=C('grid'))
            .load_cubes(dst='images')
            .adaptive_reshape(src='images')
            .normalize(src='images')

            # Predict with model, then aggregate
            .predict_model('model',
                           B('images'),
                           fetches='sigmoid',
                           save_to=B('predictions'))
            .update_accumulator(src='predictions', accumulator=C('accumulator'))
        )
        return inference_template
