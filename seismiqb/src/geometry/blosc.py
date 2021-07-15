""" Blosc sliced geometry. """
#pylint: disable=unpacking-non-sequence
import os
from zipfile import ZipFile, BadZipFile

import dill
import blosc
import numpy as np

from .converted import SeismicGeometryConverted



class BloscFile:
    """ Blosc file with slices.
    To make it a single file, `ZipFile` is used.
    The inner structure is as follows:
    file.blosc/
        dataset_name_0/
            _meta
            0
            1
            2
            ...
        dataset_name_1/
            _meta
            0
            1
            ...

    The drawback of `ZipFile` is being unable to remove individual sub-files without recreating the archive or
    very dirty hacks. Therefore, this file container is not supposed to be appended to after creation.

    Semantics and namings are the same, as in `h5py` to provide identical API:
    this way, we can make storage-agnostic code.
    """
    def __init__(self, path, mode='r', clevel=6, cname='lz4hc', shuffle=0):
        self.path = path
        self.mode = mode
        self.open_handler()

        self.clevel = clevel
        self.cname = cname
        self.shuffle = shuffle

        if mode == 'w':
            self.key_to_dataset = {}

        elif mode == 'r':
            available_keys = {name.split('/')[0] for name in self.zipfile.namelist()}

            self.key_to_dataset = {}
            for key in available_keys:
                self.key_to_dataset[key] = BloscDataset(key, parent=self)

    def open_handler(self):
        """ Open the ZipFile handler. Can be re-used when the same file is accessed from multiple processes. """
        #pylint: disable=consider-using-with
        self.zipfile = ZipFile(self.path, mode=self.mode)

    def repack(self, aggregation=None):
        """ Aggregate files with duplicate names. """
        namelist = set(self.namelist())
        infolist = self.zipfile.infolist()

        path_out = self.path + '_temporal'
        with BloscFile(path_out, mode='w') as out:
            for key, dataset in self.key_to_dataset.items():
                out.create_dataset(key, shape=dataset.shape, dtype=dataset.dtype)

            for name in namelist:
                infos = [info for info in infolist if info.filename == name]

                # Get all versions of duplicates
                slides = []
                for info in infos:
                    try:
                        with self.zipfile.open(info, mode='r') as f:
                            slide = self.load(f)
                            slides.append(slide)
                    except BadZipFile:
                        pass

                # Aggregate
                if aggregation is None:
                    slide = slides[0]
                elif aggregation in ['max', 'maximum']:
                    slide = np.maximum.reduce(slides)
                elif aggregation in ['mean', 'average']:
                    slide = np.mean(slides, axis=0)

                # Write to new file
                with out.zipfile.open(name, mode='w') as file:
                    out.dump(slide, file)

        os.remove(self.path)
        os.rename(out.path, self.path)
        return BloscFile(self.path, mode='r')

    # Utilities
    def namelist(self):
        """ Contents of the file. """
        namelist = self.zipfile.namelist()
        return [name for name in namelist if '_meta' not in name]

    def __len__(self):
        return len(self.zipfile.infolist())


    def __contains__(self, key):
        """ Check if projections is available. """
        return key in self.key_to_dataset

    def __repr__(self):
        return f'BloscFile for {self.path}'


    # Inner dataset creating/indexing
    def create_dataset(self, key, shape, dtype):
        """ Create additional subdirectory in the root archive of a given name.
        Shape and dtype are stored in the meta.
        """
        with self.zipfile.open(f'{key}/_meta', mode='w') as file:
            dill.dump((shape, dtype), file)

        dataset = BloscDataset(key, parent=self)

        self.key_to_dataset[key] = dataset
        return dataset

    def __getitem__(self, key):
        """ Get existing `BloscDataset` instance by key. """
        dataset = self.key_to_dataset.get(key)
        if dataset:
            return dataset
        raise KeyError(f'Dataset {key} does not exist!')


    # Instance manager
    def close(self):
        """ Close the underlying `ZipFile`.
        Unlike most other file formats in Python, actually needed: without that, file becomes corrupted.
        """
        self.zipfile.close()

    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.close()

    def __del__(self):
        """ An extra safety measure for actually closing the file. """
        self.close()


    # Blosc utilities
    def dump(self, array, file):
        """ Store compressed `NumPy` array to an opened file handler. Makes the data C-contigious, if it is not.
        Also stores shape and dtype of an array.
        """
        if not array.data.c_contiguous:
            array = np.ascontiguousarray(array)

        compressed = blosc.compress_ptr(array.__array_interface__['data'][0],
                                        array.size, array.dtype.itemsize,
                                        clevel=self.clevel, cname=self.cname,
                                        shuffle=self.shuffle)

        dill.dump((array.shape, array.dtype), file)
        file.write(compressed)

    def load(self, file):
        """ Read the shape and dtype from a file, create a buffer, decompress data into it. """
        shape, dtype = dill.load(file)
        compressed = file.read()

        array = np.empty(shape=shape, dtype=dtype)
        blosc.decompress_ptr(compressed, array.__array_interface__['data'][0])
        return array

    def load_to_buffer(self, file, buffer):
        """ Decompress data from a file to a provided buffer. """
        _ = dill.load(file)
        compressed = file.read()

        blosc.decompress_ptr(compressed, buffer.__array_interface__['data'][0])
        return buffer



class BloscDataset:
    """ A dataset inside `BloscFile`. Essentially, a subdirectory.
    Contains a reference to the original `BloscFile`.
    """
    RETRIES = 5
    def __init__(self, key, parent):
        self.key = key
        self.parent = parent

        with self.zipfile.open(f'{key}/_meta', mode='r') as file:
            self.shape, self.dtype = dill.load(file)

    def namelist(self):
        """ Contents of the dataset. """
        namelist = self.parent.namelist()
        return [item for item in namelist if item.startswith(self.key)]

    # Utility
    def __getattr__(self, name):
        """ Re-direct everything to the parent. """
        return getattr(self.parent, name)

    def __repr__(self):
        return f'<BLOSC dataset "{self.key}": shape {tuple(self.shape)}, type {self.dtype}>'

    # Item management
    def __setitem__(self, key, slide):
        """ Save slide to a sub-directory. Number of slide is used as the filename. """
        key = key if isinstance(key, (int, slice)) else key[0]
        if isinstance(key, slice):
            for i, pos in enumerate(np.arange(self.shape[0])[key]):
                self[int(pos)] = slide[i]
        else: # int
            with self.zipfile.open(f'{self.key}/{key}', mode='w') as file:
                self.dump(slide, file)


    def __getitem__(self, key):
        """ Load the file, named as the number of a slide. """
        key = key if isinstance(key, (int, slice)) else key[0]
        if isinstance(key, slice):
            array = []
            for i in np.arange(self.shape[0])[key]:
                array.append(self[int(i)])
            return np.stack(array, axis=0)

        for _ in range(self.RETRIES):
            # In a multi-processing setting, the ZipFile can be (somehow) closed from other process
            # We can mitigate that by re-opening the handler, if needed.
            try:
                with self.zipfile.open(f'{self.key}/{key}', mode='r') as file:
                    slide = self.load(file)
            except ValueError:
                self.parent.open_handler()
        return slide



class SeismicGeometryBLOSC(SeismicGeometryConverted):
    """ Infer `BloscFile` with multiple projections of the same data inside. """
    #pylint: disable=attribute-defined-outside-init
    def process(self, **kwargs):
        """ Detect available projections in the cube and store handlers to them in attributes. """
        self.file = BloscFile(self.path, mode='r')

        # Check available projections
        self.available_axis = [axis for axis, name in self.AXIS_TO_NAME.items()
                               if name in self.file]
        self.available_names = [self.AXIS_TO_NAME[axis] for axis in self.available_axis]

        # Save cube handlers to instance
        self.axis_to_cube = {}
        for axis in self.available_axis:
            name = self.AXIS_TO_NAME[axis]
            cube = self.file[name]

            self.axis_to_cube[axis] = cube
            setattr(self, name, cube)

        # Parse attributes from meta / set defaults
        self.add_attributes(**kwargs)
