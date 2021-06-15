""" A holder for horizon enhancement steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon with synthetic distortions.
    - making inference on a selected data.
"""
import numpy as np

from ...batchflow import Pipeline, B, V, C, D, P, R

from .horizon import HorizonController
from .torch_models import EncoderDecoder


class Enhancer(HorizonController):
    """
    Provides interface for train, inference and quality assesment for the task of horizon enhancement.
    """

    def train(self, horizon, **kwargs):
        """ Train model for horizon enhancement. """
        dataset = self.make_dataset(horizon=horizon)
        self.make_sampler(dataset, use_grid=False, side_view=True, bins=np.array([500, 500, 100]))
        return super().train(dataset, **kwargs)


    def inference(self, horizon, model, filtering_matrix=None, **kwargs):
        """ Runs enhancement procedure for a given horizon with provided model. """
        dataset = self.make_dataset(horizon=horizon)
        if filtering_matrix is None:
            filtering_matrix = 1 - (horizon.full_matrix > 0)
        prediction = super().inference(dataset=dataset, model=model,
                                       filtering_matrix=filtering_matrix, **kwargs)[0]
        prediction.name = f'enhanced_{horizon.name}'
        return prediction


    def load_pipeline(self):
        """ Defines data loading procedure.
        Following parameters are fetched from pipeline config: `width` and `rebatch_threshold`.
        """
        return (
            Pipeline()
            .make_locations(points=D('train_sampler')(C('batch_size')),
                            shape=C('crop_shape'), side_view=C('side_view', default=False))
            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.7))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'],
                              shape=C('crop_shape'))
            .normalize(src='images')
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
        def concat_inputs(batch):
            batch.images = np.concatenate((batch.images, batch.prior_masks), axis=1)

        return (
            Pipeline()
            .init_variable('loss_history', default=[])
            .init_model(mode='dynamic', model_class=C('model_class', default=EncoderDecoder),
                        name='model', config=C('model_config'))
            .call(concat_inputs)
            .train_model('model', fetches='loss', save_to=V('loss_history', mode='a'),
                         images=B('images'),
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
            .init_variable('predictions', [])

            # Load data
            .make_locations(points=D('grid_gen')(), shape=C('crop_shape'),
                            side_view=C('side_view', default=False))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=3)
            .adaptive_reshape(src=['images', 'prior_masks'],
                              shape=C('crop_shape'))
            .normalize(src='images')

            # Use model for prediction
            .predict_model('model',
                           B('images'),
                           B('prior_masks'),
                           fetches='predictions',
                           save_to=V('predictions', mode='e'))
        )
        return inference_template


    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ Run the entire procedure of horizon enhancement. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        horizon = dataset.labels[0][0]

        model = self.train(horizon=horizon, **kwargs)

        prediction = self.inference(horizon, model, **kwargs)
        prediction = self.postprocess(prediction)
        self.evaluate(prediction, dataset=dataset)
        return prediction
