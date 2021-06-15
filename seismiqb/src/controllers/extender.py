""" A holder for horizon extension steps inherited from `.class:Enhancer` with:
    - redifined get_mask_transform_ppl to thin out loaded mask
    - making an iterative inference to cover the holes in a given horizon.
"""
import gc
from copy import copy
from pprint import pformat
from time import perf_counter

import numpy as np
import torch

from ...batchflow import Pipeline, Config, Notifier, Monitor
from ...batchflow import D, B, F, C, V, P, R

from ..horizon import Horizon

from .enhancer import Enhancer



class Extender(Enhancer):
    """ Provides interface for train, inference and quality assesment for the task of horizon extension. """
    #pylint: disable=attribute-defined-outside-init
    DEFAULTS = Config({
        **Enhancer.DEFAULTS,
        'inference': {
            'batch_size': None,
            'crop_shape': None,
            'width': 3,

            'n_steps': 100,
            'stride': 32,
            'threshold': 10,

            'prefetch': 0,
        }
    })

    def inference(self, horizon, model, config=None, **kwargs):
        """ Fill the holes of a given horizon with the supplied model.

        Works by making predictions near the horizon boundaries and stitching them to the original one.
        """
        # Prepare parameters
        config = config or {}
        config = Config({**self.config['common'], **self.config['inference'], **config, **kwargs})
        n_steps, stride, batch_size, crop_shape = config.get(['n_steps', 'stride', 'batch_size', 'crop_shape'])
        threshold = config.get('threshold', 25)
        prefetch = config.get('prefetch', 0)

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config.config, depth=2), '末 inference_config.txt')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
            monitor.__enter__()

        # Make dataset and copy horizon
        horizon = copy(horizon)
        dataset = self.make_dataset(horizon=horizon)

        prev_len = initial_len = len(horizon)
        self.log(f'Inference started for {n_steps} steps with stride {stride}')
        notifier = Notifier(self.config.bar,
                            desc='Extender inference', update_total=False,
                            file=self.make_savepath('末 inference.log'))

        start_time = perf_counter()
        for _ in notifier(range(n_steps)):
            # Create grid of crops near horizon holes
            dataset.make_extension_grid(dataset.indices[0],
                                        crop_shape=crop_shape,
                                        stride=stride,
                                        labels_src=horizon,
                                        batch_size=batch_size)

            # Create pipeline TODO: make better `add_model`
            inference_pipeline = self.get_inference_template() << config << dataset
            inference_pipeline.models.add_model('model', model)

            try: # TODO: rethink control flow
                inference_pipeline.run(D.size, n_iters=dataset.grid_iters, prefetch=prefetch)
            except TypeError:
                # no additional predicts
                break

            # Merge surfaces on crops to the horizon itself
            for patch_horizon in inference_pipeline.v('predicted_horizons'):
                merge_code, _ = Horizon.verify_merge(horizon, patch_horizon,
                                                     mean_threshold=5.5, adjacency=5)
                if merge_code == 3:
                    _ = horizon.overlap_merge(patch_horizon, inplace=True)

            # Log length increase
            curr_len = len(horizon)
            if (curr_len - prev_len) < threshold:
                break
            self.log(f'Extended from {prev_len} to {curr_len}, + {curr_len - prev_len}')
            prev_len = curr_len

            # Cleanup
            gc.collect()
            inference_pipeline.reset('variables')

        notifier.close()
        elapsed = perf_counter() - start_time

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)

        self.log(f'Total points added: {curr_len - initial_len}')

        torch.cuda.empty_cache()
        horizon.name = f'extended_{horizon.name[8:]}' # get rid of `copy_of_` postfix

        self.inference_log = {
            'elapsed': elapsed,
            'added_points': curr_len - initial_len,
        }
        return horizon


    def distortion_pipeline(self):
        """ Defines transformations performed with `masks` component. """
        def functor(scale):
            return lambda m: np.sin(m[:, 0] * scale)

        return (
            Pipeline()
            .transpose(src='masks', order=(1, 2, 0))
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0., high=0.4)),
                        length=P(R('uniform', low=0.30, high=0.5)))
            .filter_out(src='prior_masks', dst='prior_masks',
                        expr=F(functor)(R('uniform', low=15, high=35)), low=0.0, p=0.7)
            .transpose(src=['masks', 'prior_masks'], order=(2, 0, 1))
        )


    def get_inference_template(self):
        """ Defines inference pipeline. """
        def concat_inputs(batch):
            batch.images = np.concatenate((batch.images, batch.prior_masks), axis=1)

        inference_template = (
            Pipeline()

            # Init everything
            .init_variable('predicted_horizons', default=[])

            # Load data
            .make_locations(points=D('grid_gen')(), shape=D('shapes_gen')())
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=C('width', default=3))
            .adaptive_reshape(src=['images', 'prior_masks'],
                              shape=C('crop_shape'))
            .normalize(src='images')
            .call(concat_inputs)

            # Use model for prediction
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=B('predicted_masks', mode='w'))
            .transpose(src='predicted_masks', order=(1, 2, 0))
            .masks_to_horizons(src_masks='predicted_masks', threshold=0.5, minsize=16,
                               order=D('orders_gen')(), dst='horizons', skip_merge=True)
            .update(V('predicted_horizons', mode='e'), B('horizons'))
        )
        return inference_template


    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ Run the entire procedure of horizon extension. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        horizon = dataset.labels[0][0]

        model = self.train(horizon=horizon, **kwargs)

        prediction = self.inference(horizon, model, **kwargs)
        prediction = self.postprocess(prediction)
        self.evaluate(prediction, dataset=dataset)
        return prediction
