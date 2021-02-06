""" A holder for horizon extension steps inherited from `.class:Enhancer` with:
    - redifined get_mask_transform_ppl to thin out loaded mask
    - making an iterative inference to cover the holes in a given horizon.
"""
from copy import copy

import numpy as np
import torch

from ...batchflow import Pipeline, B, V, C, D, P, R, L

from ..horizon import Horizon

from .enhancer import Enhancer
from .best_practices import MODEL_CONFIG_EXTENSION



class Extender(Enhancer):
    """
    Provides interface for train, inference and quality assesment for the task of horizon extension.
    """
    #pylint: disable=unused-argument, logging-fstring-interpolation, no-member, attribute-defined-outside-init

    def inference(self, horizon, n_steps=30, batch_size=128, stride=16):
        """ Extend, i.e. fill the holes of the given horizon with the
        Horizon Extension algorithm using loaded/trained model.
        For each step of the Extension algorithm crops close to the horizon boundaries
        are generated so they contain a part of the known horizon that
        will be used as a prior mask to the ExtensionModel model.
        Predicted horizons for every crop will be merged to the known horizon on each step.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be extended
        n_steps : int
            Number of steps of the Extension algorithm.
        batch_size : int
            Size of batches for train and inference.
        stride : int
            Distance between a horizon border and a corner of sampled crop that is fed to the model.

        Logs
        ----
        Start of inference, increased horizon length at each step of the procedure.
        """
        dataset = self.make_dataset_from_horizon(horizon)

        config = {
            'batch_size': batch_size,
            'model_pipeline': self.model_pipeline
        }
        horizon = copy(horizon)

        prev_len = len(horizon)
        self.log(f'Inference started for {n_steps} with stride {stride}.')
        for _ in self.make_pbar(range(n_steps), desc=f'Extender inference on {horizon.name}'):
            # Create grid of crops near horizon holes
            dataset.make_extension_grid(dataset.indices[0],
                                        crop_shape=self.crop_shape,
                                        stride=stride,
                                        labels_src=horizon,
                                        batch_size=batch_size)

            # Add current horizon to dataset labels in order to make create_masks work
            dataset.labels[dataset.indices[0]] = [horizon]

            inference_pipeline = (self.get_inference_template() << config) << dataset
            try:
                for _ in range(dataset.grid_iters):
                    inference_pipeline.next_batch(1, n_epochs=None)
            except TypeError:
                # no additional predicts
                break

            # Merge surfaces on crops to the horizon itself
            horizons = [*inference_pipeline.v('predicted_horizons')]
            for hor in horizons:
                merge_code, _ = Horizon.verify_merge(horizon, hor,
                                                     mean_threshold=5.5,
                                                     adjacency=5)
                if merge_code == 3:
                    _ = horizon.overlap_merge(hor, inplace=True)

            # Log length increase
            curr_len = len(horizon)
            if (curr_len - prev_len) < 25:
                break
            self.log(f'Extended from {prev_len} to {curr_len}, + {curr_len - prev_len}')
            prev_len = curr_len

            # Cleanup
            inference_pipeline.reset('variables')
            inference_pipeline = None

        torch.cuda.empty_cache()
        horizon.name = f'extended_{horizon.name[8:]}' # get rid of `copy_of_` postfix
        self.predictions = [horizon]


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
                        expr=L(functor)(R('uniform', low=15, high=35)), low=0.0, p=0.7)
            .transpose(src=['masks', 'prior_masks'], order=(2, 0, 1))
        )


    def get_inference_template(self):
        """ Defines inference pipeline. """
        inference_template = (
            Pipeline()
            # Init everything
            .init_variable('predicted_horizons', default=list())
            .import_model('base', C('model_pipeline'))
            # Load data
            .make_locations(points=L(D('grid_gen')), shape=L(D('shapes_gen')))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=3)
            .adaptive_reshape(src=['images', 'prior_masks'],
                              shape=self.crop_shape)
            .normalize(mode='q', src='images')
            # Use model for prediction
            .predict_model('base',
                           B('images'),
                           B('prior_masks'),
                           fetches='predictions',
                           save_to=B('predicted_masks', mode='w'))
            .transpose(src='predicted_masks', order=(1, 2, 0))
            .masks_to_horizons(src='predicted_masks', threshold=0.5, minsize=16,
                               order=L(D('orders_gen')), dst='horizons', skip_merge=True)
            .update(V('predicted_horizons', mode='e'), B('horizons'))
        )
        return inference_template



    @staticmethod
    def run(horizon, cube_path=None, save_dir='.', crop_shape=(1, 64, 64),
            model_config=None, n_iters=400, batch_size=128, device=None,
            n_steps=10, stride=16, return_instance=False):
        """ Run all steps of the Extension procedure including creating dataset for
        the given horizon, creating instance of the class and running train and inference
        methods.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be extended
        n_steps : int
            Number of steps of the Extension algorithm.
        model_config : dict
            Neural network architecture.
        n_iters : int
            Number of iterations to train model for.
        crop_shape : tuple of 3 ints
            Size of sampled crops for train and inference.
        batch_size : int
            Size of batches for train and inference.
        device : str or int
            Device specification.
        stride : int
            Distance between a horizon border and a corner of sampled crop that is fed to the model.
        save_dir : str
            Path to save images, logs, and other data.
        return_instance : bool
            Whether to return created `.class:Extender` instance.
        Returns
        -------
        Extended horizon.
        """
        model_config = MODEL_CONFIG_EXTENSION if model_config is None else model_config
        extender = Extender(save_dir=save_dir, model_config=model_config, device=device,
                            crop_shape=crop_shape, batch_size=batch_size)
        extender.train(horizon, n_iters=n_iters, use_grid=False)
        extender.inference(horizon, n_steps=n_steps,
                           batch_size=batch_size, stride=stride)
        if return_instance:
            return extender
        return extender.predictions[0]
