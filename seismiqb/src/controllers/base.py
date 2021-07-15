""" A convenient class to hold:
    - dataset creation
    - model train procedure
    - inference on dataset
    - evaluating predictions
    - and more
"""
#pylint: disable=import-error, no-name-in-module, wrong-import-position, protected-access
import os
import gc
import logging

from time import perf_counter
from ast import literal_eval
from pprint import pformat

import psutil

import numpy as np
import torch

from ...batchflow import Config, Monitor
from ...batchflow.models.torch import EncoderDecoder

from ..plotters import plot_loss



class BaseController:
    """ A common interface for train, inference, postprocessing and quality assessment.
    Supposed to be used in an environment with set `CUDA_VISIBLE_DEVICES` variable.

    At initialization, a nested configuration dict should be provided.

    Common parameters are defined on root level of the config:
        savedir : str
            Directory to store outputs: logs, graphs, predictions.
        monitor : bool
            Whether to track resources during execution.
        logger : callable
            Function to log messages.
        bar : bool
            Whether to show progress bars during execution.
        plot : bool
            Whether to display graphs during execution.

    Each of the methods retrieves parameters from the configuration by its name:
        - `train`
        - `inference`
        - `postprocess`
        - `evaluate`
    Each of the methods also has the `config` argument to override parameters from that configuration.
    Keyword arguments are used with the highest priority.
    """
    #pylint: disable=attribute-defined-outside-init
    DEFAULTS = Config({
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

            'rebatch_threshold': 0.8,
            'rescale_batch_size': True,

            'prefetch': 1,
            'n_iters': 100,
            'early_stopping': True,
        },

        'inference': {},

        # Common keys for both train and inference
        'common': {},

        # Make predictions better
        'postprocess': {},

        # Compute metrics
        'evaluate': {}
    })

    def __init__(self, config=None, **kwargs):
        self.config = Config(self.DEFAULTS)
        self.config += config or {}
        self.config += kwargs

        self.monitor = self.config.monitor
        self.plot = self.config.plot

        devices = os.getenv('CUDA_VISIBLE_DEVICES')
        if devices:
            gpu_list = literal_eval(devices)
            self.gpu_list = list(gpu_list) if isinstance(gpu_list, tuple) else [gpu_list]
        else:
            self.gpu_list = []

        self.make_filelogger()
        self.log(f'Initialized {self.__class__.__name__}')

    # Utility functions
    def make_savepath(self, *postfix):
        """ Create nested path from provided strings.
        Uses `savedir` config option.

        If `savedir` config option is None, then None is returned: that is used as signal to omit saving
        of, for example, metric map images, etc.
        """
        savedir = self.config['savedir']
        if savedir is not None:
            path = os.path.join(savedir, *postfix[:-1])
            os.makedirs(path, exist_ok=True)
            return os.path.join(savedir, *postfix)
        return None

    # Logging
    def make_filelogger(self):
        """ Create logger inside `savedir`.

        Note that logging is important.
        """
        log_path = self.make_savepath('controller.log')
        if log_path:
            handler = logging.FileHandler(log_path, mode='w')
            handler.setFormatter(logging.Formatter('%(asctime)s      %(message)s'))

            logger = logging.getLogger(str(id(self)))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            self.filelogger = logger.info
        else:
            self.filelogger = None

    def log(self, msg):
        """ Log supplied message into both filelogger and supplied one. """
        process = psutil.Process(os.getpid())
        uss = process.memory_full_info().uss / (1024 ** 3)
        msg = f'{self.__class__.__name__} ::: {uss:2.4f} ::: {msg}'

        logger = self.config.logger
        if logger:
            logger = logger if isinstance(logger, (tuple, list)) else [logger]
            for logger_ in logger:
                logger_(msg)
        if self.filelogger:
            self.filelogger(msg)

    def log_to_file(self, msg, path):
        """ Log message to a separate file. """
        log_path = self.make_savepath(path)
        if log_path:
            with open(log_path, 'w') as file:
                print(msg, file=file)

    # Dataset creation
    def make_dataset(self, **kwargs):
        """ Create dataset to train/inference on. Must be implemented in inherited classes. """
        _ = kwargs

    def make_notifier(self):
        """ Create notifier. """
        return {
            'bar': self.config.bar,
            'monitors': 'loss_history',
            'file': self.make_savepath('末 model_loss.log'),
        }

    # Train
    def train(self, dataset, sampler, config=None, **kwargs):
        """ Train model on a provided dataset.
        Uses the `get_train_template` method to create pipeline of model training.

        Returns
        -------
        Model instance
        """
        # Prepare parameters
        config = config or {}
        pipeline_config = Config({**self.config['common'], **self.config['train'], **config, **kwargs})
        n_iters, prefetch, rescale = pipeline_config.pop(['n_iters', 'prefetch', 'rescale_batch_size'])

        notifier = self.make_notifier() if self.config['bar'] else None
        self.log(f'Train started on device={self.gpu_list}')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
            monitor.__enter__()

        # Make pipeline
        pipeline_config['sampler'] = sampler
        train_pipeline = self.get_train_template(**kwargs) << pipeline_config << dataset

        # Log: pipeline_config to a file
        self.log_to_file(pformat(pipeline_config.config, depth=2), '末 train_config.txt')

        # Test batch to initialize model and log stats
        batch = train_pipeline.next_batch()
        model = train_pipeline.m('model')

        self.log(f'Target batch size: {pipeline_config["batch_size"]}')
        self.log(f'Actual batch size: {len(batch)}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')

        # Log: full and short model repr
        self.log_to_file(repr(model.model), '末 model_repr.txt')
        self.log_to_file(model._short_repr(), '末 model_shortrepr.txt')

        # Rescale batch size, if needed
        if rescale:
            scale = pipeline_config['batch_size'] / len(batch)
            pipeline_config['batch_size'] = int(pipeline_config['batch_size'] * scale)
            self.log(f'Rescaling batch size to: {pipeline_config["batch_size"]}')

        train_pipeline.set_config(pipeline_config)

        # Run training procedure
        start_time = perf_counter()
        self.log(f'Train run: n_iters={n_iters}, prefetch={prefetch}')
        train_pipeline.run(n_iters=n_iters, prefetch=prefetch, notifier=notifier)
        elapsed = perf_counter() - start_time

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 train_resource.png'), show=self.plot)

        # Log: loss over iteration
        plot_loss(model.loss_list, show=self.plot,
                  savepath=self.make_savepath('末 model_loss.png'))
        final_loss = np.mean(model.loss_list[-25:])

        # Log: model train information
        self.log_to_file(model._information(config=True, devices=True, model=False, misc=True), '末 model_info.txt')

        # Log: stats
        self.log(f'Trained for {model.iteration} iterations in {elapsed:4.1f}s')
        self.log(f'Average of 25 last loss values: {final_loss:4.3f}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()
        train_pipeline.reset('variables')
        for item in dataset.geometries.values():
            item.reset_cache()
        self.log('')

        self.train_log = {
            'start_time': start_time,
            'elapsed': elapsed,
            'final_loss': final_loss,
        }
        return model

    def finetune(self, dataset, sampler, model, config=None, **kwargs):
        """ Train given model for a couple more iterations on a specific sampler.
        Used to fine-tune the model on specific range during inference stage.
        """
        # Prepare parameters
        config = config or {}
        pipeline_config = Config({**self.config['common'], **self.config['train'],
                                  **self.config['finetune'], **config, **kwargs})
        n_iters, prefetch = pipeline_config.pop(['n_iters', 'prefetch'])

        pipeline_config['sampler'] = sampler
        pipeline_config['source_model'] = model
        train_pipeline = self.get_train_template(**kwargs) << pipeline_config << dataset
        train_pipeline.run(n_iters=n_iters, prefetch=prefetch)

        torch.cuda.empty_cache()


    # Inference
    def inference(self, dataset, model, **kwargs):
        """ Inference: use trained/loaded model for making predictions on the supplied dataset.
        Must be implemented in inherited classes.
        """
        _ = dataset, model, kwargs

    # Postprocess
    def postprocess(self, predictions, **kwargs):
        """ Optional postprocessing: algorithmic adjustments to predictions.
        Must be implemented in inherited classes.
        """
        _ = predictions, kwargs

    # Evaluate
    def evaluate(self, predictions, targets=None, dataset=None, **kwargs):
        """ Assess quality of model generated outputs. Must be implemented in inherited classes. """
        _ = predictions, targets, dataset, kwargs


    # Pipelines: used inside train/inference methods
    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        _ = kwargs
