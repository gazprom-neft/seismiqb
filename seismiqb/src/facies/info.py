""" Seismic facies linkage manager. """
import os
import json

from collections import defaultdict
from pandas import DataFrame

from ..utils import get_environ_flag, to_list



class FaciesInfo():
    """ Class to manage geometry-labels linkage.

    Initialized from path to json file or/and keyword arguments.
    Key-value structure in json file must match parameters setup described below.
    Some items are optional — their defaults are defined in class variable `DEFAULT_INFO`.

    Parameters
    ----------
    root : str, optional
        Path to cubes files storage folder.
    cubes : str or list of str
        Names of cubes files to load.
    labels_dirs : str
        Path to folders containing corresponding labels files.
        Must be relative to loaded cube file location.
    subsets : nested dict
        keys : str
            Subset names.
        values : dict
            keys : str
                Cubes files names included in subset.
            values : list of str
                Labels files names from `labels_dirs` included in subset.
    json_path : str
        Path to json file containing keyword arguments in a format defined above.
        Arguments from json file have higher priority, than those explicitly provided.

    Notes
    -----
    Cubes names in parameters above are implied to be in 'XX_YYY' format, i.e. without 'CUBE_' prefix.

    Example of json file
    --------------------
    {
        "root": "/data/cubes",
        "cubes": ["01_AAA.blosc", "02_BBB.blosc"],

        "labels_dir": ["INPUTS/FACIES/FANS_HORIZONS", "INPUTS/FACES/FANS"],

        "subsets":
        {
            "train":
            {
                "01_AAA.blosc": ["horizon_1.char"],
                "02_BBB.blosc": ["horizon_2.char"],
            },
            "infer":
            {
                "01_AAA.blosc": ["horizon_2.char"],
                "02_BBB.blosc": ["horizon_1.char"],
            }
        }
    }
    """
    def __init__(self, root='/data/seismic_data/seismic_interpretation', cubes=None,
                 labels_dirs="INPUTS/HORIZONS/RAW", dst_labels=None, subsets=None, json_path=None):
        self.anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        json_kwargs = {}
        if json_path is not None:
            with open(json_path, 'r') as f:
                json_kwargs = json.load(f)

        self.root = json_kwargs.get('root', root)

        cubes = json_kwargs.get('cubes', cubes)
        if cubes is None:
            raise ValueError("Cubes list must be specified in either kwargs or provided json.")
        self.cubes = to_list(cubes)

        labels_dirs = json_kwargs.get('labels_dirs', labels_dirs)
        self.labels_dirs = to_list(labels_dirs)

        self.dst_labels = json_kwargs.get('dst_labels', dst_labels)

        subsets = json_kwargs.get('subsets', subsets)
        self.subsets = self.make_subsets_storage(subsets)

    def make_subsets_storage(self, subsets):
        """ Wrap subsets linkage info with flexible nested structure.
        Besides cubes-labels linkage given in `self.info['subsets']`,
        create subset containing all possible labels for every cube name under 'all' key.
        """
        result = defaultdict(lambda: defaultdict(list))

        # Detect labels common for corresponding labels directories and add them to `all` subset
        for cube in self.cubes:
            cube_dir = self.cube_dir_from_name(cube)
            cube_labels = []
            for labels_dir in self.labels_dirs:
                labels_set = set()
                for file in os.listdir(f"{self.root}/{cube_dir}/{labels_dir}"):
                    if file.startswith('.') or file.endswith('.dvc'):
                        continue
                    labels_set.add(file)
                cube_labels.append(labels_set)
            result['all'][cube] = list(set.intersection(*cube_labels))

        # Add subsets linkages provided in `subsets` to subsets storage
        if subsets is not None:
            for subset, linkage in subsets.items():
                result[subset] = defaultdict(list, linkage)

        return result

    def _update_on_event(self, event):
        """ Method to pass to ipywidgets `observe` call. """
        # pylint: disable=protected-access
        event_name = type(event).__name__

        if event_name == 'Bunch':
            # if event is a checkbox click
            subset = event['owner']._subset
            cube = event['owner']._cube
            label = event['owner']._label
            interact = 'append' if event['new'] else 'remove'
            getattr(self.subsets[subset][cube], interact)(label)
        elif event_name == 'Button':
            # if event is a button click
            cubes = getattr(event, '_cubes', None)
            if cubes is None:
                # if cube button
                boxes = getattr(event, '_labels', None)
            else:
                # if subset button
                boxes = sum([cube._labels for cube in cubes], [])
            first_box_value = boxes[0].value
            for box in boxes:
                box.value = not first_box_value

    @staticmethod
    def cube_dir_from_name(name):
        """ Make cube directory name from cube name. """
        return f"CUBE_{name.split('.')[0]}"

    def displayed_cube_name(self, name):
        """ Remove extension from cube name and optionally remove field name. """
        displayed_name = name.split('.')[0]
        if self.anonymize:
            displayed_name = displayed_name[:displayed_name.rfind('_')]
        return displayed_name

    @staticmethod
    def displayed_label_name(name):
        """ Remove extension from labels name. """
        return name.split('.')[0]

    def show_subsets(self):
        """ Display predefined subsets as a dataframe. """
        data = {}
        for cube, labels in self.subsets['all'].items():
            for label in labels:
                idx = (self.displayed_cube_name(cube), self.displayed_label_name(label))
                data[idx] = [label in values[cube] for values in self.subsets.values()]
        df = DataFrame(data=data, index=self.subsets.keys()).T
        df = df.replace([False, True], ['ㅤㅤ❌ㅤㅤ', 'ㅤㅤ✅ㅤㅤ'])
        style = [dict(selector="th", props=[('text-align', 'center')]),
                 dict(selector="caption", props=[('font-size', '15px'), ('font-weight', 'bold')])]
        return df.style.set_caption("SUBSETS").set_table_styles(style)

    def interactive_split(self, subsets=('train', 'infer'), main_subset='all'):
        """ Render interactive menu to include/exclude labels for every name in `subsets`. """
        # pylint: disable=import-outside-toplevel, protected-access
        from ipywidgets import Checkbox, VBox, HBox, Button, Layout
        from IPython.display import display

        subsets = to_list(subsets)

        box_layout = Layout(display='flex', flex_flow='column', align_items='center', width='23%')

        vboxes = []
        for subset in subsets:
            subset_controls = []
            subset_button = Button(description=subset, button_style='info')
            subset_button._cubes = []
            subset_button.on_click(self._update_on_event)
            subset_controls.append(subset_button)
            for cube in self.cubes:

                cube_button = Button(description=self.displayed_cube_name(cube))
                cube_button._labels = []
                cube_button._controls = subset_controls
                cube_button._subset = subset
                cube_button.on_click(self._update_on_event)
                subset_button._cubes.append(cube_button)
                subset_controls.append(cube_button)

                for label in self.subsets[main_subset][cube]:
                    default_label_value = label in self.subsets[subset].get(cube, [])
                    label_box = Checkbox(description=self.displayed_label_name(label),
                                         value=default_label_value)
                    label_box._subset = subset
                    label_box._cube = cube
                    label_box._label = label
                    label_box.observe(self._update_on_event, names='value')
                    cube_button._labels.append(label_box)
                    subset_controls.append(label_box)
            vboxes.append(VBox(subset_controls, layout=box_layout))

        hbox = HBox(vboxes)
        display(hbox)

    def make_linkage(self, subset='all', dst_labels=None):
        """ Return cubes paths and cubes-labels linkage to load for requested subset. """
        linkage = {
            cube: sorted(labels)
            for cube, labels in sorted(self.subsets[subset].items())
            if labels
            }

        if not sum(linkage.values(), []):
            msg = f"No labels were selected for subset `{subset}`. "\
                  "Either choose non-empty subset or add some labels to requested one. "\
                  "Labels can be added in either loaded json or via `FaciesInfo.interactive_split`."
            raise ValueError(msg)

        cubes_paths = [
            f"{self.root}/{self.cube_dir_from_name(cube)}/{cube}"
            for cube in linkage.keys()
        ]

        dst_labels = dst_labels or self.dst_labels
        if dst_labels is None:
            raise ValueError("`dst_labels` must be provided either in `__init__` or in `make_linkage`.")
        dst_labels = to_list(dst_labels)

        labels_linkage = {}
        for labels_dir, dst in zip(self.labels_dirs, dst_labels):
            src_labels_linkage = {}
            for cube, labels in linkage.items():
                labels_path = f"{self.root}/{self.cube_dir_from_name(cube)}/{labels_dir}"
                labels_files_paths = [f"{labels_path}/{label}" for label in labels]
                geometry_index = cube.split('.')[0] # geometry indices are cubes filenames without extension
                src_labels_linkage[geometry_index] = labels_files_paths
            labels_linkage[dst] = src_labels_linkage

        return cubes_paths, labels_linkage

    def dump(self, path):
        """ Save info. """
        info = {'root': self.root, 'cubes': self.cubes, 'labels_dirs': self.labels_dirs, 'subsets': self.subsets}
        with open(path, 'w') as f:
            f.write(json.dumps(info, indent=4, sort_keys=False))
