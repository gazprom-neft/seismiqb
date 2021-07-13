[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7-green.svg)](https://pytorch.org)

# SeismiQB

`seismiQB` is a framework for research and deployment of deep learning models on post-stack seismic data. It allows to

* convert `SEG-Y` to a compressed and quantized data formats, that take 5x less disk space and load slices up to 40 times faster
* work with a number of labels: horizons, faults, facies (both 2d and 3d)
* load crops of data and create segmentation masks to train neural networks, following defined distribution of location generation
* augment seismic images and masks with both usual CV methods like `flip`, `rotate` and `elastic_transform`, as well as to compute geological attributes: `phases`, `frequencies`, `instantaneous_amplitudes`
* define complex neural networks with simple and intuitive configurations: just a few lines of code are enough to define models ranging from vanilla `UNet` to most sophisticated versions of modern `EfficientNets`
* apply ready-to-use models and pipelines to detect horizons, faults, alluvial fans and fluvial channels
* export predicted entities to a convenient formats like CHARISMA and FAULT_STICKS for an easy validation by geophysicists


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

## Tutorials

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

Please cite `seismiqb` in your publications if it helps your research.

    Khudorozhkov R., Koryagin A., Tsimfer S., Mylzenova D. SeismiQB library for seismic interpretation with deep learning. 2019.

```
@misc{seismiQB_2019,
  author       = {R. Khudorozhkov and A. Koryagin and S. Tsimfer and D. Mylzenova},
  title        = {SeismiQB library for seismic interpretation with deep learning},
  year         = 2019
}
```
