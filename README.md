[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.5-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-1.14-orange.svg)](https://tensorflow.org)

# SeismiQB

`seismiQB` is a framework for deep learning research on 3d-cubes of seismic data. It allows to

* `sample` and `load` crops of `SEG-Y` cubes for training neural networks
* convert `SEG-Y` cubes to `HDF5`-format for even faster `load`
* `create_masks` of different types from horizon labels for segmenting horizons, facies and other seismic bodies
* build augmentation pipelines using custom augmentations for seismic data as well as `rotate`, `noise` and `elastic_transform`
* segment horizons and interlayers using [`UNet`](https://arxiv.org/abs/1505.04597) and [`Tiramisu`](https://arxiv.org/abs/1611.09326)
* extend horizons from a couple of seismic `ilines` in spirit of classic autocorrelation tools but with deep learning
* convert predicted masks into horizons for convenient validation by geophysicists


## Installation

With [pipenv](https://docs.pipenv.org/):

    pipenv install git+https://github.com/gazprom-neft/seismiqb.git#egg=seismiqb

With [pip](https://pip.pypa.io/en/stable/):

    pip3 install git+https://github.com/gazprom-neft/seismiqb.git

After that just import `seismiqb`:
```python
import seismiqb
```

To get the developer version, run
```
git clone --recursive https://github.com/gazprom-neft/seismiqb.git
```

## Turorials

### [Cube processing](tutorials/01_Geometry.ipynb)
Working with SEG-Y cubes with various indexing headers (e.g. pre-stack and post-stack).

### [Labeled structures](tutorials/02_Horizon.ipynb)
Our dedicated `Horizon` class is capable of loading data from multiple geological formats, computing a wealth of statistics of it, and a lot more. We also provide interfaces for other types of geological bodies like faults, facies and labels in pre-stack cubes.

### [Cubeset](tutorials/03_Cubeset.ipynb)
A wrapper aroung `geometries` and `labels`, that can generate data from random labeled locations from multiple cubes and apply both geological and computer vision augmentations.

### [Metrics](tutorials/04_Metrics.ipynb)
In order to evaluate our results (particularly predicted horizons), we developed a few seismic attributes to assess quality of seismic cubes, sparse carcasses and labeled surfaces.


## Ready-to-use ML models

### [Carcass interpolation](models/Carcass_interpolation/01_Demo_E.ipynb)
This model spreads a very sparse hand-labeled carcass of a horizon to the whole cube spatial area by solving a task of binary segmentation.

### [Horizon extension](models/Horizon_extension/Demo_E.ipynb)
Enlarge picked (possibly by other models) horizons to cover more area.

### [Interlayer segmentation](models/Interlayer_segmentation/Segmenting_interlayers.ipynb)
Applying the multi-class segmentation model to the task of horizon detection. Note that the model was developed with older `seismiQB` versions and does not work anymore.

### [Inter-cube generalization](models/Intercube_generalization/01_Model.ipynb)
Application of a model, trained on a set of cubes, to a completely unseen data.


## Citing seismiQB

Please cite `seismicqb` in your publications if it helps your research.

    Khudorozhkov R., Koryagin A., Tsimfer S., Mylzenova D. SeismiQB library for seismic interpretation with deep learning. 2019.

```
@misc{seismiQB_2019,
  author       = {R. Khudorozhkov and A. Koryagin and S. Tsimfer and D. Mylzenova},
  title        = {SeismiQB library for seismic interpretation with deep learning},
  year         = 2019
}
```
