""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch


from ...batchflow import Config, Pipeline, Notifier
from ...batchflow import B, C, D, P, R, V, F, I
from ...batchflow.models.torch import TorchModel, ResBlock, EncoderDecoder
from .base import BaseController
from ..cubeset import SeismicCubeset
from ..samplers import SeismicSampler, RegularGrid
from ..fault import Fault
from ..layers import InputLayer
from ..utils import adjust_shape_3d
from ..utility_classes import Accumulator3D
from ..plotters import plot_image
from ..metrics import FaultsMetrics

class FaultController(BaseController):
    """ Controller for faults detection tasks. """
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # Data
        'dataset': {
            'path': '/cubes/',
            'train_cubes': [],
            'transposed_cubes': [],
            'inference_cubes': dict(),
            'label_dir': '/INPUTS/FAULTS/NPY_WIDTH_{}/*',
            'width': 3,
            'ext': 'qblosc',
            'ratios': None,
            'threshold': 0
        },

        # Model parameters
        'train': {
            # Training parameters
            'batch_size': 1024,
            'crop_shape': [1, 128, 512],
            'n_iters': 2000,
            'callback/each': 100,
            'visualize_crops': True,

            # Augmentation parameters
            'angle': 25,
            'scale': (0.7, 1.5, 1),
            'augment': True,

            # Normalization parameters
            'itemwise': False,
            'normalization_layer': False,
            'normalization_window': (1, 1, 100),

            # Model parameters
            'phase': True,
            'continuous_phase': False,
            'filters': [64, 96, 128, 192, 256],
            'model': 'UNet',
            'loss': 'bce',
            'output': 'sigmoid',
            'slicing': 'native',
            'prefetch': 0,
            'rescale_batch_size': False,
        },

        'inference': {
            'batch_size': 32,
            'crop_shape': [1, 128, 512],
            'inference_batch_size': 32,
            'inference_chunk_shape': (100, None, None),
            'smooth_borders': False,
            'stride': 0.5,
            'orientation': 'ilines',
            'slicing': 'native',
            'output': 'sigmoid',
            'itemwise': False
        }
    })

    # Model (not controller) config
    BASE_MODEL_CONFIG = {
        'optimizer': {'name': 'Adam', 'lr': 0.01},
        "decay": {'name': 'exp', 'gamma': 0.9, 'frequency': 100, 'last_iter': 2000},
        'microbatch': C('microbatch', default=True),
        'initial_block': {
            'enable': C('phase'),
            'filters': C('filters')[0] // 2,
            'kernel_size': 5,
            'downsample': False,
            'attention': 'scse',
            'phases': C('phase'),
            'continuous': C('continuous_phase'),
            'window': C('normalization_window'),
            'normalization': C('normalization_layer')
        },
        'loss': C('loss')
    }

    UNET_CONFIG = {
        'initial_block/base_block': InputLayer,
        'body/encoder': {
            'num_stages': 4,
            'order': 'sbd',
            'blocks': {
                'base': ResBlock,
                'n_reps': 1,
                'filters': C('filters')[:-1],
                'attention': 'scse',
            },
        },
        'body/embedding': {
            'base': ResBlock,
            'n_reps': 1,
            'filters': C('filters')[-1],
            'attention': 'scse',
        },
        'body/decoder': {
            'num_stages': 4,
            'upsample': {
                'layout': 'tna',
                'kernel_size': 5,
            },
            'blocks': {
                'base': ResBlock,
                'filters': C('filters')[-2::-1],
                'attention': 'scse',
            },
        },
        'head': {
            'base_block': ResBlock,
            'filters': [16, 8],
            'attention': 'scse'
        },
        'output': torch.sigmoid,
        'common/activation': 'relu6',
        'loss': C('loss')
    }

    def make_train_dataset(self, **kwargs):
        """ Create train dataset as an instance of :class:`.SeismicCubeset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=True, **kwargs)

    def make_inference_dataset(self, labels=False, **kwargs):
        """ Create inference dataset as an instance of :class:`.SeismicCubeset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=False, labels=labels, **kwargs)

    def _make_dataset(self, train=True, labels=True, bar=False, **kwargs):
        """ Create an instance of :class:`.SeismicCubeset` with cubes and faults. Arguments are from 'dataset'
        subconfig and kwargs. """
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        cubes = config['train_cubes'] if train else config['inference_cubes']

        paths = [self.amplitudes_path(item) for item in cubes]
        dataset = SeismicCubeset(index=paths)

        if labels:
            transposed = config['transposed_cubes']
            direction = {f'amplitudes_{cube}': 0 if cube not in transposed else 1 for cube in cubes}
            dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True,
                         direction=direction, verify=True, bar=bar)
        else:
            dataset.load_geometries()

        return dataset

    def make_sampler(self, dataset, **kwargs):
        """ Create sampler for generating locations to train on. """
        config = {**self.config['dataset'], **kwargs}
        ratios = config['ratios']
        threshold = config['threshold']
        crop_shape = self.config['train']['crop_shape']

        if self.config['train/augment'] and self.config['train/adjust']:
            crop_shape = self.adjust_shape(crop_shape)

        if len(dataset) > 0:
            if ratios is None:
                ratios = {}
                for i in range(len(dataset)):
                    faults = dataset.labels[i]
                    fault_area = sum([len(np.unique(faults[j].points)) for j in range(len(faults))])
                    cube_area = np.prod(dataset.geometries[i].cube_shape)
                    ratios[dataset.indices[i]] = fault_area / cube_area

            weights = np.array([ratios[i] for i in dataset.indices])
            weights /= weights.sum()
            weights = np.clip(weights, 0.1, 0.3)
            weights /= weights.sum()

            weights = weights.tolist()
        else:
            weights = [1]

        self.log(f'Train dataset cubes weights: {weights}.')

        sampler = SeismicSampler(labels=dataset.labels, crop_shape=crop_shape, mode='fault',
                                 threshold=threshold, proportions=weights, **kwargs)

        return sampler

    def load_train(self):
        """ Create loading pipeline for train stages. """
        return (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', slicing=C('slicing'))
            .create_masks(dst='masks')
            .adaptive_reshape(src=['images', 'masks'])
            .normalize(mode='q', itemwise=C('itemwise'), src='images')
        )

    def load_test(self, create_masks=True):
        """ Create loading pipeline for inference stages. """
        ppl = (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', slicing=C('slicing'))
            .adaptive_reshape(src='images')
            .normalize(mode='q', itemwise=C('itemwise'), src='images')
        )
        if create_masks:
            ppl += (Pipeline()
                .create_masks(dst='masks')
                .adaptive_reshape(src='masks')
            )
        return ppl

    def augmentation_pipeline(self):
        """ Augmentations for training. """
        return (Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -C('angle'), C('angle'))), src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', C('scale')[0], C('scale')[1])), src=['images', 'masks'], p=0.3)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
            .central_crop(C('crop_shape'), src=['images', 'masks'])
            .cutout_2d(src=['images'], patch_shape=np.array((1, 40, 40)), n=3, p=0.2)
        )

    def train_pipeline(self):
        """ Create train pipeline. """
        model_class = F(self.get_model_class)(C('model'))
        model_config = F(self.get_model_config)(C('model'))
        ppl = (Pipeline()
            .init_variable('loss_history', [])
            .init_model(name='model', model_class=model_class, mode='dynamic', config=model_config)
            .adaptive_expand(src=['images', 'masks'])
            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         save_to=V('loss_history', mode='a'))
            .call(self.visualize_predictions, train_pipeline=B().pipeline,
                  savepath='prediction', each=self.config['train/callback/each'],
                  iteration=I())
        )
        if self.config['train/visualize_crops']:
            ppl += (Pipeline()
                .predict_model('model', images=B('images')[:1], fetches=C('output'), save_to=B('prediction'))
            )
        return ppl

    def make_notifier(self):
        """ Make notifier. """
        if self.config['train/visualize_crops']:
            return Notifier(None, graphs=['loss_history',
                {'source': B('images'), 'name': 'images', 'plot_function': self.custom_plotter},
                {'source': B('masks'), 'name': 'masks', 'plot_function': self.custom_plotter},
                {'source': B('prediction'), 'name': 'predictions', 'plot_function': self.custom_plotter},
            ])
        return super().make_notifier()

    def custom_plotter(self, ax=None, container=None, **kwargs):
        """ Plot examples during train stage. """
        data = container['data']
        data = data[0][0]
        if data.ndim == 3:
            data = data[0]
        ax.imshow(data.T)
        ax.set_title(container['name'], fontsize=18)

        ax.set_xlabel('axis one', fontsize=18)
        ax.set_ylabel('axis two', fontsize=18)

    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        return (
            self.load_train(**kwargs) +
            (self.augmentation_pipeline(**kwargs) if self.config['train/augment'] else Pipeline()) +
            self.train_pipeline(**kwargs)
        )

    def get_model_config(self, name):
        """ Get model config depending model architecture. """
        if name == 'UNet':
            return {**self.BASE_MODEL_CONFIG, **self.UNET_CONFIG}
        raise ValueError(f'Unknown model name: {name}')

    def get_model_class(self, name):
        """ Get model class depending model architecture. """
        if name == 'UNet':
            return EncoderDecoder
        return TorchModel

    def dump_model(self, model, path):
        """ Dump model. """
        model.save(os.path.join(self.config['savedir'], path))

    # Inference functional

    def metrics_pipeline_template(self, n_bins=1000):
        """ Metrics pipeline. """
        return (Pipeline()
            .init_variable('metric', [])
            .init_variable('semblance_hist', np.zeros(n_bins))
            .compute_attribute(src='images', dst='semblance', attribute='semblance', window=(1, 5, 20))
            .update(V('metric', mode='e'), F(FaultsMetrics().similarity_metric)(B('semblance'), B('predictions')))
            .update(V('semblance_hist'), V('semblance_hist') + F(np.histogram)(B('semblance').flatten(),
                    bins=n_bins, range=(0, 1))[0])
        )

    def get_inference_template(self, train_pipeline=None, model_path=None, create_masks=False):
        """ Define the whole inference procedure pipeline. """
        if train_pipeline is not None:
            test_pipeline = Pipeline().import_model('model', train_pipeline)
        else:
            test_pipeline = Pipeline().load_model('model', TorchModel, 'dynamic', path=model_path)

        test_pipeline += self.load_test(create_masks=create_masks)
        test_pipeline += (
            Pipeline()
            .adaptive_expand(src='images')
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=C('output'), save_to=B('predictions'))
            .adaptive_squeeze(src='predictions')
            .run_later(D('size'))
        )

        if self.config['inference/smooth_borders']:
            if isinstance(self.config['inference/smooth_borders'], bool):
                step = 0.1
            else:
                step = self.config['inference/smooth_borders']
            test_pipeline += Pipeline().update(B('predictions') , F(smooth_borders)(B('predictions'), step))

        if create_masks:
            test_pipeline += Pipeline().update(V('target', mode='e'), B('masks'))

        test_pipeline += self.metrics_pipeline_template()
        test_pipeline += (Pipeline()
            .update_accumulator(src='predictions', accumulator=C('accumulator'))
        )

        return test_pipeline

    def get_inference_ranges(self, cubes):
        """ Parse inference ranges. """
        cubes = cubes.copy()
        if isinstance(cubes, (list, tuple)):
            cubes = {cube: (0, None, None, None) for cube in cubes}
        for cube in cubes:
            cubes[cube] = [cubes[cube]] if isinstance(cubes[cube][0], int) else cubes[cube]
        return cubes

    def make_accumulator(self, geometry, slices, crop_shape, strides, orientation=0, path=None):
        """ Make grid and accumulator for inference. """
        batch_size = self.config['inference']['inference_batch_size']
        name = 'cube_i'

        grid = RegularGrid(geometry=geometry,
                           threshold=0,
                           orientation=orientation,
                           ranges=slices,
                           batch_size=batch_size,
                           crop_shape=crop_shape, strides=strides)

        accumulator = Accumulator3D.from_aggregation(aggregation='mean',
                                                     origin=grid.origin,
                                                     shape=grid.shape,
                                                     fill_value=0.0,
                                                     name=name,
                                                     path=path)

        return grid, accumulator

    def inference_on_slides(self, train_pipeline=None, model_path=None, create_mask=False, pbar=False, **kwargs):
        """ Make inference on slides. """
        config = {**self.config['dataset'], **self.config['inference'], **kwargs}
        crop_shape = config['crop_shape']
        cubes = config['inference_cubes']
        strides = config['stride']

        strides = strides if isinstance(strides, tuple) else [strides] * 3

        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log('Create test pipeline and dataset.')
        dataset = self.make_inference_dataset(create_mask)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_mask)
        inference_pipeline = inference_pipeline << config << dataset


        cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.get_inference_ranges(cubes).items()
        }

        outputs = {}
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            for item in cubes[cube_idx]:
                self.log(f'Create prediction for {cube_idx}: {item[1:]}. orientation={item[0]}.')
                orientation = item[0]
                slices = item[1:]
                if len(slices) != 3:
                    slices = (None, None, None)

                grid, accumulator = self.make_accumulator(geometry, slices, crop_shape, strides, orientation)
                ppl = inference_pipeline << {'sampler': grid, 'accumulator': accumulator}

                ppl.run(n_iters=grid.n_iters, bar=pbar)
                prediction = accumulator.aggregate()

                image = geometry[
                    slices[0][0]:slices[0][1],
                    slices[1][0]:slices[1][1],
                    slices[2][0]:slices[2][1]
                ]

                if orientation == 1:
                    image = image.transpose([1, 0, 2])
                    prediction = prediction.transpose([1, 0, 2])

                metrics = np.nanmean(ppl.v('metric')), np.argmax(ppl.v('semblance_hist')) / 1000
                outputs[cube_idx] += [[slices, image, prediction, *metrics]]
        return outputs

    def visualize_predictions(self, *args, overlap=True, threshold=0.05, each=100, iteration=0, **kwargs):
        """ Plot predictions for cubes and ranges specified in 'inference' section of config. """
        if iteration % each == 0:
            results = self.inference_on_slides(*args, **kwargs)
            for cube, cube_results in results.items():
                for item in cube_results:
                    slices, image, prediction, faults_metric, noise_metric = item[:5]

                    if self.config['savedir'] is not None:
                        savepath = os.path.join(self.config['savedir'], 'prediction')
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        savepath = os.path.join(
                            savepath,
                            f'{cube}_{slices}_{iteration}_{faults_metric:.03f}_{noise_metric:.03f}.png'
                        )
                        show = False
                    else:
                        savepath = None
                        show = True

                    prediction = prediction[0]
                    prediction[prediction < threshold] = np.nan
                    if overlap:
                        plot_image([image[0], prediction], separate=False, cmap='gray', figsize=(20, 20),
                                   savepath=savepath, show=show)
                    else:
                        plot_image(prediction, figsize=(20, 20), savepath=savepath, show=show)
            return faults_metric, noise_metric
        return None, None

    def inference_on_cube(self, train_pipeline=None, model_path=None, fmt='sgy', save_to=None, prefix=None,
                          tmp='hdf5', bar=True, **kwargs):
        """ Make inference on cube. """
        config = {**self.config['dataset'], **self.config['inference'], **kwargs}
        strides = config['stride'] if isinstance(config['stride'], tuple) else [config['stride']] * 3

        crop_shape = config['crop_shape']
        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log('Create test pipeline and dataset.')
        dataset = self.make_inference_dataset(create_mask=False)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_masks=False)
        inference_pipeline.set_config(config)

        cubes = config['inference_cubes']
        cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.get_inference_ranges(cubes).items()
        }

        if save_to:
            dirname = save_to
        else:
            dirname = os.path.join(
                    os.path.dirname(dataset.geometries[0].path),
                    'PREDICTIONS/FAULTS',
            )
            if not os.path.exists(os.path.dirname(dirname)):
                os.makedirs(os.path.dirname(dirname))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        prefix = prefix or ''

        outputs = dict()
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            for item in cubes[cube_idx]:
                self.log(f'Create prediction for {cube_idx}: {item[1:]}. axis={item[0]}.')
                axis = item[0]
                ranges = item[1:]
                if len(ranges) != 3:
                    ranges = (None, None, None)

                filenames = {ext: os.path.join(dirname, self.make_filename(prefix, 'x' if axis==1 else 'i', ext))
                             for ext in ['hdf5', 'sgy', 'blosc', 'qblosc', 'npy', 'meta']}

                if fmt in ['hdf5', 'blosc', 'qblosc']:
                    path = filenames[fmt]
                elif fmt in ['sgy', 'segy']:
                    path = filenames[tmp]
                elif fmt == 'npy':
                    path = None

                grid, accumulator = self.make_accumulator(geometry, ranges, crop_shape, strides, axis, path)
                ppl = inference_pipeline << dataset << {'sampler': grid, 'accumulator': accumulator}

                ppl.run(n_iters=grid.n_iters, bar=bar)
                prediction = accumulator.aggregate()

                if fmt == 'npy':
                    outputs.append(prediction)
                if fmt == 'sgy':
                    copyfile(dataset.geometries[0].path_meta, filenames['meta'])
                    dataset.geometries[0].make_sgy(
                        path_hdf5=filenames[tmp],
                        path_spec=dataset.geometries[0].segy_path.decode('utf-8'),
                        path_segy=filenames['sgy'],
                        remove_hdf5=True, zip_result=True, pbar=True
                    )

    # Path utils

    def amplitudes_path(self, cube):
        """ Get full path for cube. """
        ext = self.config['dataset/ext']
        filename = self.config['dataset/path'] + f'CUBE_{cube}/amplitudes_{cube}.{ext}'
        if os.path.exists(filename):
            return filename
        raise ValueError(f"File doesn't exist: {filename}")

    def cube_name_from_alias(self, path):
        """ Get cube name from alias. """
        return os.path.splitext(self.amplitudes_path(path).split('/')[-1])[0]

    def cube_name_from_path(self, path):
        """ Get cube name from path. """
        return os.path.splitext(path.split('/')[-1])[0]

    def make_filename(self, prefix, orientation, ext):
        """ Make filename for infered cube. """
        return (prefix + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_{}.{}').format(orientation, ext)

    # Train utils

    def adjust_shape(self, crop_shape):
        """ Adjust shape before augmentations. """
        config = self.config['train']
        crop_shape = np.array(crop_shape)
        load_shape = adjust_shape_3d(crop_shape[[1, 2, 0]], config['angle'], scale=config['scale'])
        return (load_shape[2], load_shape[0], load_shape[1])

# Train utils

def smooth_borders(crops, step):
    """ Smooth image borders. """
    mask = border_smoothing_mask(crops.shape[-3:], step)
    mask = np.expand_dims(mask, axis=0)
    if len(crops.shape) == 5:
        mask = np.expand_dims(mask, axis=0)
    crops = crops * mask
    return crops

def border_smoothing_mask(shape, step):
    """ Make mask to smooth borders. """
    mask = np.ones(shape)
    axes = [(1, 2), (0, 2), (0, 1)]
    if isinstance(step, (int, float)):
        step = [step] * 3
    for length, s, axis in zip(shape, step, axes):
        if isinstance(s, float):
            s = int(length * s)
        if length >= 2 * s:
            _mask = np.ones(length, dtype='float32')
            _mask[:s] = np.linspace(0, 1, s+1)[1:]
            _mask[:-s-1:-1] = np.linspace(0, 1, s+1)[1:]
            _mask = np.expand_dims(_mask, axis)
            mask = mask * _mask
    return mask
