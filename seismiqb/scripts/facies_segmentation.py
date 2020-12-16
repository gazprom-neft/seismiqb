# Alluvial fans segmentation on seismic data

import os
import sys
import json
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

sys.path.insert(0, '..')

from seismiqb.batchflow import Pipeline
from seismiqb.batchflow import B, V, D, P, R
from seismiqb.batchflow.models.torch import EncoderDecoder

from seismiqb import SeismicCubeset
from seismiqb import MODEL_CONFIG as DEFAULT_MODEL_CONFIG

# Global parameters
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0"
CROP_SHAPE = (256, 256, 21)       # shape of sampled crops
ITERS = 1000                      # number of train iterations
BATCH_SIZE = 64                   # number of crops inside one batch


def detect_fans():
    # Load predefined train-test split
    DATA_SPLIT = 'split.json'
    with open(f'{DATA_SPLIT}', 'r') as handle:
        train_test_split = json.load(handle)

    DATA_DIR = train_test_split['DATA_DIR']
    SMOOTH_OUT_CUBES = [f'amplitudes_{cube_name}' for cube_name in ['33_SAMBURG', '39_YY']]


    ##########################################
    ######## Model training procedure ########
    ##########################################

    # Load and preprocess train data
    TRAIN_HDF5_PATHS = [f'{DATA_DIR}/CUBE_{CUBE_NAME}/amplitudes_{CUBE_NAME}.hdf5'
                        for CUBE_NAME in train_test_split['TRAIN']]
    train_dataset = SeismicCubeset(TRAIN_HDF5_PATHS)

    train_dataset.load_corresponding_labels(train_test_split['TRAIN'],
                                            [f'INPUTS/FACIES/{subdir}' for subdir in ('FANS_HORIZONS', 'FANS')],
                                            ['horizons', 'fans'],
                                            main_labels='horizons')
    train_dataset.apply_to_attrs('smooth_out', SMOOTH_OUT_CUBES, ['horizons', 'fans'])

    # Define model config
    OVERRIDE_CONFIG = {
        'optimizer': {'name': 'Adam', 'lr': 0.01,},
        "decay": {'name': 'exp', 'gamma': 0.1, 'frequency': 150},
        'microbatch': 2,
        'common/activation': 'relu6',
    }

    MODEL_CONFIG = {**DEFAULT_MODEL_CONFIG, **OVERRIDE_CONFIG}

    CUTOUT_SHAPE = P(R('uniform',
                       (CROP_SHAPE[0]//4, CROP_SHAPE[1]//4, CROP_SHAPE[2]),
                       (CROP_SHAPE[0]//6, CROP_SHAPE[1]//6, CROP_SHAPE[2]),
                       size=3))

    # Define train pipeline
    train_pipeline = (
        Pipeline()
        # Load data and all of the horizon-derived attributes
        .make_locations(points=D.sampler.sample(BATCH_SIZE), shape=CROP_SHAPE)
        .load_attribute(src_attribute='cube_values', dst='amplitudes',
                        transform='min-max')
        .load_attribute(src_attribute='heights', dst='heights',
                        transform='min-max')
        .load_attribute(src_attribute='instant_phases', dst='instant_phases',
                        transform='min-max')
        .load_attribute(src_attribute='instant_amplitudes', dst='instant_amplitudes',
                        transform='min-max')
        .concat_components(src=['amplitudes', 'heights',
                                'instant_phases', 'instant_amplitudes'],
                           dst='images')

        # Create segmentation mask
        .load_attribute(src_labels='fans', src_attribute='masks', dst='masks',
                        transform={'fill_value': 0})

        # Apply augmentations
        .cutout_2d(p=.2, patch_shape=CUTOUT_SHAPE, n=P(R('uniform', 1, 4)),
                   src=['images', 'masks'], dst=['images', 'masks'])
        .rotate(p=.2, angle=P(R('uniform', -45, +45)),
                src=['images', 'masks'], dst=['images', 'masks'])
        .transpose(src=['images', 'masks'], order=(2, 0, 1))

        # Init model
        .init_model('dynamic', EncoderDecoder, 'model', MODEL_CONFIG)

        # Train model
        .train_model('model', images=B.images, masks=B.masks)
    ) << train_dataset

    # Run training process
    train_pipeline.run(D.size, n_iters=ITERS, bar='t')


    ##########################################
    ####### Model evaluation procedure #######
    ##########################################

    # Load and preprocess train data
    TEST_HDF5_PATHS = [f'{DATA_DIR}/CUBE_{CUBE_NAME}/amplitudes_{CUBE_NAME}.hdf5'
                       for CUBE_NAME in train_test_split['TEST']]
    test_dataset = SeismicCubeset(TEST_HDF5_PATHS)

    test_dataset.load_corresponding_labels(train_test_split['TEST'],
                                           [f'INPUTS/FACIES/{subdir}' for subdir in ('FANS_HORIZONS', 'FANS')],
                                           ['horizons', 'fans'],
                                           main_labels='horizons')

    test_dataset.apply_to_attrs('smooth_out', SMOOTH_OUT_CUBES, ['horizons', 'fans'])

    # Define evaluation pipeline
    test_pipeline = (
        Pipeline()
        # Load data and all of the horizon-derived attributes
        .make_locations(points=D.grid_gen(), shape=CROP_SHAPE)
        .load_attribute(src_attribute='cube_values', dst='amplitudes',
                        transform='min-max')
        .load_attribute(src_attribute='heights', dst='heights',
                        transform='min-max')
        .load_attribute(src_attribute='instant_phases', dst='instant_phases',
                        transform='min-max')
        .load_attribute(src_attribute='instant_amplitudes', dst='instant_amplitudes',
                        transform='min-max')
        .concat_components(src=['amplitudes', 'heights',
                                'instant_phases', 'instant_amplitudes'],
                           dst='images')

        # Apply augmentations
        .transpose(src=['images'], order=(2, 0, 1))

        # Init pipeline variable to store predictions
        .init_variable('predictions', [])

        # Import trained model
        .import_model('model', train_pipeline)

        # Run inference
        .predict_model('model', images=B('images'),
                       fetches='predictions',
                       save_to=V('predictions', mode='e'))
    )

    # Run evaluation process
    test_dataset.make_labels_prediction(test_pipeline, CROP_SHAPE, 5, bar='t')

    # Calculate metrics and dump predictions
    for idx in test_dataset.indices:
        for true_label, pred_label in zip(test_dataset[idx, 'fans'], test_dataset[idx, 'predictions']):
            true = true_label.load_attribute('masks')
            pred = pred_label.load_attribute('masks')

            tp = np.sum(true * pred)
            fp = np.sum((1 - true) * pred)
            fn = np.sum(true * (1 - pred))
            dice = 2 * tp / (2 * tp + fp + fn)

            title = f'`{true_label.name}` on `{idx}`'

            print("Sørensen–Dice coefficient for true and predicted labels of {} is {:.2f}".format(title, dice))

            pred_label.dump(f"{pred_label.name}")

if __name__ == "__main__":
    detect_fans()
