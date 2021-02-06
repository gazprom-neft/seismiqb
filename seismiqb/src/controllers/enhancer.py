""" A holder for horizon enhancement steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon with synthetic distortions.
    - making inference on a selected data.
"""
import numpy as np

from ...batchflow import Pipeline, B, V, C, D, P, R

from ..horizon import Horizon

from .base import BaseController
from .torch_models import ExtensionModel
from .best_practices import MODEL_CONFIG_ENHANCE


class Enhancer(BaseController):
    """
    Provides interface for train, inference and quality assesment for the task of horizon enhancement.
    """
    #pylint: disable=unused-argument, logging-fstring-interpolation, no-member

    def train(self, horizon, **kwargs):
        """ Train model for horizon extension.
        Creates dataset and sampler for a given horizon and calls `meth:Detector.train`.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be enhanced.
        kwargs : dict
            Other arguments for `.meth:Detector.train`.

        Note
        ----
        In order to change training procedure, re-define :meth:`.get_train_template`.
        """
        dataset = self.make_dataset_from_horizon(horizon)
        self.make_sampler(dataset, use_grid=False, bins=np.array([500, 500, 100]))
        super().train(dataset, **kwargs)


    def inference(self, horizon, filtering_matrix=None, **kwargs):
        """ Runs enhancement procedure for a given horizon with trained/loaded model.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be enhanced.
        kwargs : dict
            Other arguments for `.meth:Detector.inference`.
        """
        #pylint: disable=attribute-defined-outside-init
        dataset = self.make_dataset_from_horizon(horizon)
        if filtering_matrix is None:
            filtering_matrix = 1 - (horizon.full_matrix > 0)
        super().inference(dataset, filtering_matrix=filtering_matrix, **kwargs)
        self.predictions = [self.predictions[0]]
        self.predictions[0].name = f'enhanced_{horizon.name}'


    def load_pipeline(self):
        """ Defines data loading procedure.
        Following parameters are fetched from pipeline config: `width` and `rebatch_threshold`.
        """
        return (
            Pipeline()
            .make_locations(points=D('train_sampler')(self.batch_size),
                            shape=self.crop_shape, side_view=True)
            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.99))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'],
                              shape=self.crop_shape)
            .normalize(mode='q', src='images')
        )

    def distortion_pipeline(self):
        """ Defines transformations performed with `masks` component. """
        def binarize(batch):
            batch.prior_masks = (batch.prior_masks > 0).astype(np.float32)

        return (
            Pipeline()
            .shift_masks(src='masks', dst='prior_masks')
            .transpose(src='prior_masks', order=(1, 2, 0))
            .elastic_transform(alpha=P(R('uniform', 30, 50)), sigma=P(R('uniform', 6, 7)),
                               src='prior_masks', p=0.5)
            .bend_masks(src='prior_masks', angle=P(R('uniform', -15, 15)))
            .call(binarize)
            .transpose(src='prior_masks', order=(2, 0, 1))
        )

    def augmentation_pipeline(self):
        """ Defines augmentation pipeline. """
        return (
            Pipeline()
            .transpose(src=['images', 'masks', 'prior_masks'], order=(1, 2, 0))
            .rotate(angle=P(R('uniform', -30, 30)),
                    src=['images', 'masks', 'prior_masks'], p=0.3)
            .flip(src=['images', 'masks', 'prior_masks'], axis=1, p=0.3)
            .transpose(src=['images', 'masks', 'prior_masks'], order=(2, 0, 1))
        )

    def train_pipeline(self):
        """ Define model initialization and model training pipeline.
        Following parameters are fetched from pipeline config: `model_config`.
        """
        return (
            Pipeline()
            .init_variable('loss_history', default=[])
            .init_model('dynamic', ExtensionModel, 'base', C('model_config'))
            .train_model('base', fetches='loss', save_to=V('loss_history', mode='a'),
                         images=B('images'),
                         prior_masks=B('prior_masks'),
                         masks=B('masks'))
        )

    def get_train_template(self, distortion_pipeline=None, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        _ = kwargs
        return (
            self.load_pipeline() +
            (distortion_pipeline or self.distortion_pipeline()) +
            self.augmentation_pipeline() +
            self.train_pipeline()
        )

    def get_inference_template(self):
        """ Defines inference pipeline. """
        inference_template = (
            Pipeline()
            # Init everything
            .init_variable('predicted_masks', default=list())
            .import_model('base', C('model_pipeline'))
            # Load data
            .make_locations(points=D('grid_gen')(), shape=self.crop_shape,
                            side_view=C('side_view', default=False))
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
                           save_to=V('predicted_masks', mode='e'))
        )
        return inference_template



    @staticmethod
    def run(horizon, cube_path=None, save_dir='.', crop_shape=(1, 64, 64),
            model_config=None, n_iters=400, batch_size=128, device=None,
            orientation='ix', return_instance=False):
        """ Run all steps of the Enhancement procedure including creating dataset for
        the given horizon, creating instance of the class and running train and inference
        methods.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon` or str
            A horizon to be extended or a path to load horizon from.
        cube_path : str, optional
            Path to cube, used if horizon is str.
        save_dir : str
            Path to save images, logs, and other data.
        crop_shape : tuple of 3 ints
            Size of sampled crops for train and inference.
        model_config : dict
            Neural network architecture.
        n_iters : int
            Number of iterations to train model for.
        batch_size : int
            Size of batches for train and inference.
        device : str or int
            Device specification.
        orientation : {'i', 'x', 'ix'}
            Orientation of the inference:
            If 'i', then cube is split into inline-oriented slices.
            If 'x', then cube is split into crossline-oriented slices.
            If 'ix', then both of previous approaches applied, and results are merged.
        return_instance : bool
            Whether to return created `.class:Enhancer` instance.
        """
        model_config = MODEL_CONFIG_ENHANCE if model_config is None else model_config
        enhancer = Enhancer(save_dir=save_dir, model_config=model_config, device=device,
                            crop_shape=crop_shape, batch_size=batch_size)
        if isinstance(horizon, str) and isinstance(cube_path, str):
            dataset = enhancer.make_dataset(cube_path, horizon_paths=horizon)
        elif isinstance(horizon, Horizon):
            dataset = enhancer.make_dataset_from_horizon(horizon)
        else:
            raise TypeError('Pass either instance of Horizon or paths to both cube and horizon.')

        enhancer.make_sampler(dataset, use_grid=False, bins=np.array([500, 500, 100]))
        enhancer.train(horizon, n_iters=n_iters, use_grid=False)
        enhancer.inference(horizon, orientation=orientation)

        if return_instance:
            return enhancer
        return enhancer.predictions[0]
