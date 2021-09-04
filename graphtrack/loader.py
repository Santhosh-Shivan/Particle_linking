from graphtrack.deeptrack.features import Merge
from . import deeptrack as dt
import numpy as np
import os
import glob
import itertools
from typing import List
import scipy
import pandas as pd

from . import graphs

_path_to_data = os.path.join(
    ".", "datasets", "{dataset}", "{sequence}", "t{number}.tif"
)
_path_to_mask = os.path.join(
    ".",
    "datasets",
    "{dataset}",
    "{sequence}_{quality}",
    "SEG",
    "man_seg{number}.tif",
)
_path_to_graph = os.path.join(
    ".", "datasets", "{dataset}", "{sequence}_GT", "TRA", "man_track.txt"
)


def Augmentation(
    image: List[dt.Feature],
    augmentation_list=None,
    default_value=lambda x: x,
    **kwargs
):
    """Applies a list of augmentations to a list of
       deeptrack features
    Parameters
    ----------
    image: list, deeptrack features
        Cell images and their corresponding masks
    augmentation_list: dict, optional
        dictionary of augmentation functions and
         their parameters
    default_value: function, optional
        function to return if no augmentation is applied
    kwargs:
        additional arguments to pass to the
        augmentation functions

    Returns
    -------
        Augmented features
    """
    augmented_image = image
    for augmentation in augmentation_list:
        print("With ", augmentation)

        args = augmentation_list[augmentation].copy()
        for key, val in args.items():
            if isinstance(val, str):
                args[key] = eval(val)

        augmented_image >>= getattr(dt, augmentation, default_value)(
            **args, **kwargs
        )

    return augmented_image


def GetLoaders(
    dataset,
    fraction_gold=1,
    validation_sequence=1,
    min_sequence_length=10,
    max_sequence_length=30,
    augmentation=None,
    **kwargs
):
    """
    Creates a data loader for a given dataset
    Parameters
    ----------
    dataset: str
        name of the dataset
    fraction_gold: float, optional
        fraction of the ground truth to use for the mask
    validation_sequence: int, optional
        sequence to use for validation
    min_sequence_length: int, optional
        minimum sequence length to use
    max_sequence_length: int, optional
        maximum sequence length to use
    augmentation: dict, optional
        dictionary of augmentation functions and
        their parameters
    kwargs:
        additional arguments to pass to the
        augmentation functions
    Returns
    -------
        dt.DataLoader
    """
    training_set, training_graph = GetAvailableImages(
        dataset, sequence=3 - validation_sequence
    )
    validation_set, validation_graph = GetAvailableImages(
        dataset, sequence=validation_sequence
    )

    def get_sequence(validation, start_frame, frame_steps):
        if validation:
            return validation_set[start_frame : start_frame + frame_steps]
        else:
            return training_set[start_frame : start_frame + frame_steps]

    def get_graph(validation):
        if validation:
            return validation_graph
        else:
            return training_graph

    root = dt.DummyFeature(
        sequence=get_sequence,
        graph=get_graph,
        frame_steps=lambda: np.random.randint(
            min_sequence_length, max_sequence_length
        ),
        start_frame=lambda frame_steps, num_of_images: np.random.randint(
            0, num_of_images - frame_steps
        ),
        dataset=dataset,
        num_of_images=lambda validation: len(validation_set)
        if validation
        else len(training_set),
        validation=False,
    )

    image_loader = root >> dt.LoadImage(
        path=lambda sequence: [
            _path_to_data.format(
                dataset=dataset, sequence=_sequence[0], number=_sequence[1]
            )
            for _sequence in sequence
        ],
        **root.properties,
    )

    ST_mask_loader = root >> dt.LoadImage(
        path=lambda sequence: [
            _path_to_mask.format(
                dataset=dataset,
                sequence=_sequence[0],
                quality="ST",
                number=_sequence[1],
            )
            for _sequence in sequence
        ],
        **root.properties,
    )

    GT_mask_loader = root >> dt.LoadImage(
        path=lambda sequence: [
            _path_to_mask.format(
                dataset=dataset,
                sequence=_sequence[0],
                quality="GT",
                number=_sequence[0],
            )
            for _sequence in sequence
        ],
        **root.properties,
    )

    mask_loader = (ST_mask_loader & GT_mask_loader) >> dt.Merge(
        MergeMasks, fraction_gold=fraction_gold
    )

    Combined = image_loader & mask_loader

    if augmentation:
        augmented_dataset = Augmentation(
            Combined, augmentation_list=augmentation
        )
    else:
        augmented_dataset = Combined

    return dt.ConditionalSetFeature(
        on_false=augmented_dataset
        >> dt.Merge(
            lambda: lambda sequence: graphs.GraphExtractor(sequence, **kwargs),
        ),
        on_true=augmented_dataset
        >> dt.Merge(
            lambda: lambda sequence: [
                sequence,
                graphs.GraphExtractor(sequence, **kwargs),
            ],
        ),
        condition="return_sequence",
        return_sequence=False,
    )


def GetAvailableImages(dataset, sequence=1):
    """
    Returns a list of available images for a given dataset
    Parameters
    ----------
    dataset: str
        name of the dataset
    sequence: int, optional
        sequence to use for training/validation
    """

    mask_path = _path_to_mask.format(
        dataset=dataset, sequence="0" + str(sequence), quality="ST", number="*"
    )
    mask_paths = glob.glob(mask_path)

    graph_data = pd.read_csv(
        _path_to_graph.format(dataset=dataset, sequence="0" + str(sequence)),
        sep=" ",
        usecols=[0, 3],
        names=["child", "parent"],
    )
    graph_data = graph_data[graph_data["parent"] > 0]

    return (
        [
            (
                os.path.normpath(p).split(os.sep)[-3][:2],
                os.path.normpath(p).split(os.sep)[-1][7:-4],
            )
            for p in mask_paths
        ],
        graph_data,
    )


def GetMask(ST, GT, fraction_gold=1):
    """
    Returns a mask for a given ST and GT image
    """
    if not isinstance(GT, np.ndarray) or GT.ndim < 2 or fraction_gold == 0:
        return ST

    GT_labels = set(GT[GT > 0].flatten())
    for label in GT_labels:
        if np.random.rand() > fraction_gold:
            continue

        mask = GT == label

        ST_overlap = ST[mask]

        mode = scipy.stats.mode(ST_overlap, axis=None).mode

        if mode > 0:
            ST[ST == mode] = 0
            ST[mask] = mode
        else:
            ST[mask] = np.max(ST) + 1

    return ST


def MergeMasks(fraction_gold=1):
    """
    Returns a stack of masks for given ST and GT images
    """

    def inner(images):

        ST = np.array(images[0])
        GT = np.array(images[1])

        masks = []
        for idx in range(np.shape(ST)[-1]):
            masks.append(GetMask(ST[..., idx], GT[..., idx], fraction_gold))

        return np.stack(masks, axis=-1)

    return inner


class ContinuousGraphGenerator(dt.generators.ContinuousGenerator):
    """
    Generator that asynchronously generates graph representations.

    The generator aims to speed up the training of networks by striking a
    balance between the generalization gained by generating new images and
    the speed gained from reusing images. The generator will continuously
    create new trainingdata during training, until `max_data_size` is reached,
    at which point the oldest data point is replaced.

    Parameters
    ----------
    feature : dt.Feature
        The feature to resolve the graphs from.
    label_function : Callable
        Function that returns the label corresponding to a feature output.
    batch_function : Callable
        Function that returns the training data corresponding a feature output.
    min_data_size : int
        Minimum size of the training data before training starts
    max_data_set : int
        Maximum size of the training data before old data is replaced.
    batch_size : int or Callable[int, int] -> int
        Number of images per batch. A function is expected to accept the current epoch
        and the size of the training data as input.
    shuffle_batch : bool
        If True, the batches are shuffled before outputting.
    feature_kwargs : dict or list of dicts
        Set of options to pass to the feature when resolving
    ndim : int
        Number of dimensions of each batch (including the batch dimension).
    output_type : str
        Type of output. Either "nodes", "edges", or "graph". If 'key' is not a
        supported output type, then the output will be the concatenation of the
        node and edge labels.
    """

    def __init__(self, feature, *args, output_type="graph", **kwargs):
        self.output_type = output_type

        dt.utils.safe_call(
            super().__init__, positional_args=[feature, *args], **kwargs
        )

    def __getitem__(self, idx):
        batch, labels = super().__getitem__(idx)

        # Extracts minimum number of nodes in the batch
        cropNodesTo = np.min(
            list(map(lambda _batch: np.shape(_batch[0])[0], batch))
        )

        inputs, outputs, nofedges = [[], [], [], []], [[], []], []

        for i in range(len(batch)):
            # Clip node features to the minimum number of nodes
            # in the batch
            nodef = batch[i][0][:cropNodesTo, :]
            inputs[0].append(nodef)

            # Extracts index of the last node in the adjacency matrix
            last_node_idx = int(
                np.where(batch[i][2][:, 1] <= cropNodesTo - 1)[0][-1] + 1
            )

            # Clips edge features and adjacency matrix to the index
            # of the last node
            edgef = batch[i][1][:last_node_idx, :]
            adjmx = batch[i][2][:last_node_idx, :]
            wghts = batch[i][3][:last_node_idx, :]

            inputs[1].append(edgef)
            inputs[2].append(adjmx)
            inputs[3].append(wghts)

            # Appends the number of edges in the batch
            nofedges.append(np.shape(edgef)[0])

            # Clips node and edge solutions
            nodesol = labels[i][0][:cropNodesTo, :]
            edgesol = labels[i][1][:last_node_idx, :]

            outputs[0].append(nodesol)
            outputs[1].append(edgesol)

        maxnOfedges = np.max(nofedges)

        # Edge augmentation
        inputs[1], weights, idxs = graphs.SelfDuplicateEdgeAugmentation(
            inputs[1], inputs[3], maxnofedges=maxnOfedges
        )
        inputs[2], *_ = graphs.SelfDuplicateEdgeAugmentation(
            inputs[2], inputs[3], maxnofedges=maxnOfedges, idxs=idxs
        )

        outputs[1], *_ = graphs.SelfDuplicateEdgeAugmentation(
            outputs[1], inputs[3], maxnofedges=maxnOfedges, idxs=idxs
        )
        inputs[3] = weights

        # Converts to numpy arrays
        inputs = tuple(map(np.array, inputs))
        outputs = tuple(map(np.array, outputs))

        output_dict = {
            "nodes": outputs[0],
            "edges": outputs[1],
            "graph": outputs,
        }
        try:
            outputs = output_dict[self.output_type]
        except KeyError:
            outputs = output_dict["graph"]

        return inputs, outputs


conf = {}


def GraphGenerator(min_data_size=1000, max_data_size=2000, **kwargs):
    """
    Returns a generator that generates graphs asynchronously.
    Parameters
    ----------
    min_data_size : int
        Minimum size of the training data before training starts
    max_data_size : int
        Maximum size of the training data before old data is replaced.
    kwargs : dict
        Keyword arguments to pass to the features.
    """
    feature = GetLoaders(**kwargs)

    conf["feature"] = feature

    # Removes augmentation from kwargs
    kwargs.pop("augmentation", None)

    args = {
        # "feature": feature,
        "batch_function": lambda graph: graph[0],
        "label_function": lambda graph: graph[1],
        "min_data_size": min_data_size,
        "max_data_size": max_data_size,
        **kwargs,
    }

    return ContinuousGraphGenerator(feature, **args)


def GetValidationSet(size=None, **kwargs):
    """
    Returns a validation set from a given loader
    Parameters
    ----------
    loader: dt.Feature
        data loader
    size: int, optional
        size of the validation set
    """
    # Get the data loader
    loader = conf["feature"]

    # Lists of graphs and solutions
    sequences = []
    graphs = [[], [], [], []]
    solutions = [[], []]

    for _ in range(size):
        # Update the loader
        loader.update()

        # Resolve the features
        sequence, (graph, solution) = loader.update().resolve(
            validation=True, return_sequence=True
        )

        # Append sequence
        sequences.append(sequence)

        # Append the graphs and solutions
        for i, (g, s) in enumerate(itertools.zip_longest(graph, solution)):
            graphs[i].append(np.array(g))

            if not (s is None):
                solutions[i].append(np.array(s))

    return sequences, graphs, solutions
