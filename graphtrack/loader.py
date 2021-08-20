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
        **root.properties
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
        **root.properties
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
        **root.properties
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
        on_true=augmented_dataset
        >> dt.Merge(
            lambda: lambda sequence: graphs.GraphExtractor(sequence, **kwargs),
        ),
        on_false=augmented_dataset,
        condition="return_graphs",
        return_graphs=True,
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


def GetValidationSet(loader: dt.Feature = None, size=None, **kwargs):
    """
    Returns a validation set from a given loader
    Parameters
    ----------
    loader: dt.Feature
        data loader
    size: int, optional
        size of the validation set
    """
    # Lists of graphs and solutions
    graphs = [[], []]
    solutions = [[], []]

    for _ in range(size):
        # Update the loader
        loader.update()

        # Resolve the features
        graph, solution = loader.resolve(validation=True)

        # Append the graphs and solutions
        for i, (g, s) in enumerate(zip(graph, solution)):
            graphs[i].append(np.array(g))
            solutions[i].append(np.array(s))

    return graphs, solutions
