""" Methods to save data as seismic cubes in different formats. """
import os
import shutil
from tqdm.auto import tqdm

import numpy as np
import h5pickle as h5py
import segyio

class ExportMixin:
    """ Container for methods to save data as seismic cubes in different formats. """
    def make_sgy(self, path_hdf5=None, path_spec=None, postfix='',
                 remove_hdf5=False, zip_result=True, path_segy=None, pbar=False):
        """ Convert POST-STACK HDF5 cube to SEG-Y format with supplied spec.

        Parameters
        ----------
        path_hdf5 : str
            Path to load hdf5 file from.
        path_spec : str
            Path to load segy file from with geometry spec.
        path_segy : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        """
        path_segy = path_segy or (os.path.splitext(path_hdf5)[0] + postfix + '.sgy')
        if not path_spec:
            if hasattr(self, 'segy_path'):
                path_spec = self.segy_path.decode('ascii')
            else:
                path_spec = os.path.splitext(self.path)[0] + '.sgy'

        # By default, if path_hdf5 is not provided, `temp.hdf5` next to self.path will be used
        if path_hdf5 is None:
            path_hdf5 = os.path.join(os.path.dirname(self.path), 'temp.hdf5')

        with h5py.File(path_hdf5, 'r') as src:
            cube_hdf5 = src['cube_i']

            from .base import SeismicGeometry #pylint: disable=import-outside-toplevel
            geometry = SeismicGeometry(path_spec)
            segy = geometry.segyfile

            spec = segyio.spec()
            spec.sorting = None if segy.sorting is None else int(segy.sorting)
            spec.format = None if segy.format is None else int(segy.format)
            spec.samples = range(self.depth)

            idx = np.stack(geometry.dataframe.index)
            ilines, xlines = self.load_meta_item('ilines'), self.load_meta_item('xlines')

            i_enc = {num: k for k, num in enumerate(ilines)}
            x_enc = {num: k for k, num in enumerate(xlines)}

            spec.ilines = ilines
            spec.xlines = xlines

            with segyio.create(path_segy, spec) as dst_file:
                # Copy all textual headers, including possible extended
                for i in range(1 + segy.ext_headers):
                    dst_file.text[i] = segy.text[i]
                dst_file.bin = segy.bin

                for c, (i, x) in enumerate(tqdm(idx, disable=(not pbar))):
                    locs = tuple([i_enc[i], x_enc[x], slice(None)])
                    dst_file.header[c] = segy.header[c]
                    dst_file.trace[c] = cube_hdf5[locs]
                dst_file.bin = segy.bin
                dst_file.bin[segyio.BinField.Traces] = len(idx)

        if remove_hdf5:
            os.remove(path_hdf5)

        if zip_result:
            dir_name = os.path.dirname(os.path.abspath(path_segy))
            file_name = os.path.basename(path_segy)
            shutil.make_archive(os.path.splitext(path_segy)[0], 'zip', dir_name, file_name)

def make_segy_from_array(array, path_segy, zip_segy=True, remove_segy=None, **kwargs):
    """ Make a segy-cube from an array. Zip it if needed. Segy-headers are filled by defaults/arguments from kwargs.

    Parameters
    ----------
    array : np.ndarray
        Data for the segy-cube.
    path_segy : str
        Path to store new cube.
    zip_segy : bool
        whether to zip the resulting cube or not.
    remove_segy : bool
        whether to remove the cube or not. If supplied (not None), the supplied value is used.
        Otherwise, True if option `zip` is True (so that not to create both the archive and the segy-cube)
        False, whenever `zip` is set to False.
    kwargs : dict
        sorting : int
            2 stands for ilines-sorting while 1 stands for xlines-sorting.
            The default is 2.
        format : int
            floating-point mode. 5 stands for IEEE-floating point, which is the standard -
            it is set as the default.
        sample_rate : int
            sampling frequency of the seismic in microseconds. Most commonly is equal to 2000
            microseconds for on-land seismic.
        delay : int
            delay time of the seismic in microseconds. The default is 0.
    """
    if remove_segy is None:
        remove_segy = zip_segy

    # make and fill up segy-spec using kwargs and array-info
    spec = segyio.spec()
    spec.sorting = kwargs.get('sorting', 2)
    spec.format = kwargs.get('format', 5)
    spec.samples = range(array.shape[2])
    spec.ilines = np.arange(array.shape[0])
    spec.xlines = np.arange(array.shape[1])

    # parse headers' kwargs
    sample_rate = int(kwargs.get('sample_rate', 2000))
    delay = int(kwargs.get('delay', 0))

    with segyio.create(path_segy, spec) as dst_file:
        # Make all textual headers, including possible extended
        num_ext_headers = 1
        for i in range(num_ext_headers):
            dst_file.text[i] = segyio.tools.create_text_header({1: '...'}) # add header-fetching from kwargs

        # Loop over the array and put all the data into new segy-cube
        for i in tqdm(range(array.shape[0])):
            for x in range(array.shape[1]):
                # create header in here
                header = dst_file.header[i * array.shape[1] + x]

                # change inline and xline in trace-header
                header[segyio.TraceField.INLINE_3D] = i
                header[segyio.TraceField.CROSSLINE_3D] = x

                # change depth-related fields in trace-header
                header[segyio.TraceField.TRACE_SAMPLE_COUNT] = array.shape[2]
                header[segyio.TraceField.TRACE_SAMPLE_INTERVAL] = sample_rate
                header[segyio.TraceField.DelayRecordingTime] = delay

                # copy the trace from the array
                trace = array[i, x]
                dst_file.trace[i * array.shape[1] + x] = trace

        dst_file.bin = {segyio.BinField.Traces: array.shape[0] * array.shape[1],
                        segyio.BinField.Samples: array.shape[2],
                        segyio.BinField.Interval: sample_rate}

    if zip_segy:
        dir_name = os.path.dirname(os.path.abspath(path_segy))
        file_name = os.path.basename(path_segy)
        shutil.make_archive(os.path.splitext(path_segy)[0], 'zip', dir_name, file_name)
    if remove_segy:
        os.remove(path_segy)
