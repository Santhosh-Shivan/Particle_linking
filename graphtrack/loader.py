from re import search
from numpy.core.numeric import full
from scipy.ndimage.measurements import label
from tensorflow.python.keras.backend import dropout
from . import deeptrack as dt
import numpy as np
import os
import glob
import itertools
from typing import List
import scipy
import pandas as pd

import tqdm
from . import graphs

import more_itertools as mit
from functools import partial
from operator import is_not

import graphtrack as gt

_path_to_data = os.path.join(
    "..", "datasets", "{dataset}", "{sequence}", "t{number}.tif"
)
_path_to_st_mask = os.path.join(
    "..",
    "datasets",
    "{dataset}",
    "{sequence}_ST",
    "SEG",
    "man_seg{number}.tif",
)
_path_to_gt_mask = os.path.join(
    "..",
    "datasets",
    "{dataset}",
    "{sequence}_GT",
    "SEG",
    "man_seg{number}.tif",
)
_path_to_graph = os.path.join(
    "..", "datasets", "{dataset}", "{sequence}_GT", "TRA", "man_track.txt"
)


def Augmentation(
    image: List[dt.Feature],
    augmentation_list=None,
    default_value=lambda x: x,
    **kwargs,
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


def LoadGraph(
    dataset,
    sequence=1,
    return_frames=False,
    **kwargs,
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
    dset, graph = GetAvailableImages(dataset, sequence=sequence)

    def load(sequence, formatter, desc="data"):
        output = None
        idx = 0
        for s in tqdm.tqdm(sequence, desc=f"Loading {desc}"):
            image_path = formatter.format(
                dataset=dataset, sequence=s[0], number=s[1]
            )
            res = dt.LoadImage(image_path)()._value
            res = np.squeeze(res)
            if output is None:
                output = np.zeros((*res.shape, len(sequence)))

            output[..., idx] = res
            idx += 1

        return output

    training_images = load(dset, _path_to_data, "images")
    training_masks = load(dset, _path_to_st_mask, "masks")

    graph = graphs.GraphExtractor(
        (training_images, training_masks),
        graph=graph,
        **kwargs,
    )
    if return_frames:
        return graph, (training_images, training_masks)
    else:
        return graph


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

    mask_path = _path_to_st_mask.format(
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

        inputs, outputs, nofedges = [[], [], [], []], [[], [], []], []

        batch_size = 0
        for i in range(len(batch)):

            # Clip node features to the minimum number of nodes
            # in the batch
            nodef = batch[i][0][:cropNodesTo, :]

            last_node_idx = 0
            # Extracts index of the last node in the adjacency matrix
            try:
                last_node_idx = int(
                    np.where(batch[i][2][:, 1] <= cropNodesTo - 1)[0][-1] + 1
                )
            except IndexError:
                continue

            # Clips edge features and adjacency matrix to the index
            # of the last node
            edgef = batch[i][1][:last_node_idx]
            adjmx = batch[i][2][:last_node_idx]
            wghts = batch[i][3][:last_node_idx]

            # Clips node and edge solutions
            nodesol = labels[i][0][:cropNodesTo]
            edgesol = labels[i][1][:last_node_idx]

            inputs[1].append(edgef)
            inputs[2].append(adjmx)
            inputs[3].append(wghts)
            nofedges.append(np.shape(edgef)[0])
            inputs[0].append(nodef)
            outputs[0].append(nodesol)
            outputs[1].append(edgesol)

            if len(labels[i]) > 2:
                outputs[2].append(labels[i][2].astype(np.float))

            batch_size += 1

        if batch_size == 0:
            return self.__getitem__((i + 1) % len(self))

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
            "graph": [outputs[0], outputs[1]],
            "global": outputs[2],
        }
        try:
            outputs = output_dict[self.output_type]
        except KeyError:
            outputs = output_dict["graph"]

        return inputs, outputs


def GetSubGraph(num_nodes, node_start, return_edge_idxs=False, **kwargs):
    def inner(data):
        graph, labels, *_ = data

        edge_connects_removed_node = np.any(
            (graph[2][:, -2:] < node_start)
            | (
                graph[2][:, -2:] > node_start + num_nodes
            ),  # >= removes relevant edges
            axis=-1,
        )

        node_features = graph[0][node_start : node_start + num_nodes]
        edge_features = graph[1][~edge_connects_removed_node]

        if np.shape(graph[2])[-1] > 2:
            # This happens when validating
            edge_connections = graph[2][~edge_connects_removed_node]
            edge_connections[:, 2:] = edge_connections[:, 2:] - node_start
        else:
            # During training
            edge_connections = (
                graph[2][~edge_connects_removed_node] - node_start
            )

        weights = graph[3][~edge_connects_removed_node]

        node_labels = labels[0][node_start : node_start + num_nodes]
        edge_labels = labels[1][~edge_connects_removed_node]

        if return_edge_idxs:
            return (
                (node_features, edge_features, edge_connections, weights),
                (
                    node_labels,
                    edge_labels,
                ),
                edge_connects_removed_node,
            )
        else:
            return (node_features, edge_features, edge_connections, weights), (
                node_labels,
                edge_labels,
            )

    return inner


def GetSubSet(randset, substract_node_idx=True, **kwargs):
    def inner(data):
        graph, labels, sets = data

        nodeidxs = np.where(sets[0] == randset)[0]
        edgeidxs = np.where(sets[1] == randset)[0]

        node_features = graph[0][nodeidxs]
        edge_features = graph[1][edgeidxs]

        if substract_node_idx:
            edge_connections = graph[2][edgeidxs] - nodeidxs[0]
        else:
            edge_connections = graph[2][edgeidxs]

        weights = graph[3][edgeidxs]

        node_labels = labels[0][nodeidxs]
        edge_labels = labels[1][edgeidxs]
        global_labels = labels[2][randset]

        return (node_features, edge_features, edge_connections, weights), (
            node_labels,
            edge_labels,
            global_labels,
        )

    return inner


def GetSubGraphsFromFrames(num_frames, frame_start, **kwargs):
    def inner(data):
        graph, _ = data
        node_start, node_end = (
            np.where(graph[0][:, 0] == frame_start)[0][0],
            np.where(graph[0][:, 0] == frame_start + num_frames)[0][-1],
        )
        return GetSubGraph(node_end - node_start, node_start, **kwargs)(data)

    return inner


# def find_missing(lst):
#     return [x for x in range(lst[0], lst[-1] + 1) if x not in lst]


def SlicedPrediction(frames_num, frame_step, model, mergedfs=True):
    def inner(data):
        graph, labels = data

        maxframe = int(graph[0][:, 0].max())

        frames = range(0, maxframe + frames_num)
        windows = mit.windowed(frames, n=frames_num, step=frame_step)
        windows = map(
            lambda x: list(filter(partial(is_not, None), x)), windows
        )

        ids = np.arange(0, len(graph[2]))

        preddfs = []
        for window in windows:
            window = [elem for elem in window if elem <= maxframe]
            # print(window)

            if not len(window) > 0:
                continue

            sub_data, sub_labels, edge_idxs = GetSubGraphsFromFrames(
                num_frames=window[-1] - window[0],
                frame_start=window[0],
                return_edge_idxs=True,
            )((graph, labels))

            preddf = ComputeEdgePrediction(
                sub_data, sub_labels, model, to_frame=True
            )
            # print(find_missing(ids[~edge_idxs]))
            preddf["id"] = ids[~edge_idxs]
            preddfs.append(preddf)

        if mergedfs:
            preddfs = (
                pd.concat(preddfs)
                .groupby("id")
                .agg(
                    {
                        "frame_x": "first",
                        "frame_y": "first",
                        "gt": np.prod,
                        "prediction": np.prod,
                    }
                )
                .reset_index()
                .drop(columns=["id"])
            )
        preddfs[["node_x", "node_y"]] = graph[2][:, -2:]
        return preddfs

    return inner


def ComputeEdgePrediction(data, labels, model, to_frame=False):
    """
    Computes the prediction of the model on the given data.
    """
    model_input = list(
        map(
            lambda x: np.expand_dims(x, 0),
            [
                data[0][:, 1:],
                data[1],
                data[2][:, 2:],
                data[3],
            ],
        )
    )
    pred = model.predict(model_input)
    pred = (pred > 0.5)[0, ...]

    true = np.expand_dims(labels[1], axis=-1)

    edges = np.append(data[2], true, axis=-1)
    # append predicted labels
    edges = np.append(edges, pred, axis=-1)

    if to_frame:
        edges_df = pd.DataFrame(
            edges,
            columns=[
                "frame_x",
                "frame_y",
                "node_x",
                "node_y",
                "gt",
                "prediction",
            ],
        )
        return edges_df
    else:
        return edges


def NoisyNode():
    def inner(data):
        graph, labels = data

        features = graph[0][:, 2:]
        features += np.random.randn(*features.shape) * np.random.rand() * 0.1

        node_features = np.array(graph[0])
        node_features[:, 2:] = features

        return (node_features, *graph[1:]), labels

    return inner


def AugmentCentroids(rotate, translate, flip_x, flip_y):
    def inner(data):
        graph, labels = data

        centroids = graph[0][:, :2]

        centroids = centroids - 0.5
        centroids_x = (
            centroids[:, 0] * np.cos(rotate)
            + centroids[:, 1] * np.sin(rotate)
            + translate[0]
        )
        centroids_y = (
            centroids[:, 1] * np.cos(rotate)
            - centroids[:, 0] * np.sin(rotate)
            + translate[1]
        )
        if flip_x:
            centroids_x *= -1
        if flip_y:
            centroids_y *= -1

        node_features = np.array(graph[0])
        node_features[:, 0] = centroids_x + 0.5
        node_features[:, 1] = centroids_y + 0.5

        return (node_features, *graph[1:]), labels

    return inner


def NodeDropout(dropout_rate=0.02):
    def inner(data):
        graph, labels = data

        # Get indexes of randomly dropped nodes
        idxs = np.array(list(range(len(graph[0]))))
        dropped_idxs = idxs[np.random.rand(len(graph[0])) < dropout_rate]

        node_f, edge_f, edge_adj, weights = graph
        node_l, edge_l, global_l = labels

        for dropped_node in dropped_idxs:

            # Find all edges connecting to the dropped node
            edge_connects_removed_node = np.any(
                edge_f == dropped_node, axis=-1
            )

            # Remove bad edges
            edge_f = edge_f[~edge_connects_removed_node]
            edge_adj = edge_adj[~edge_connects_removed_node]
            edge_l = edge_l[~edge_connects_removed_node]
            weights = weights[~edge_connects_removed_node]

        return (node_f, edge_f, edge_adj, weights), (node_l, edge_l, global_l)

    return inner


def GetFeature(full_graph, **kwargs):
    return (
        dt.Value(full_graph)
        >> dt.Lambda(
            GetSubSet,
            randset=lambda: np.random.randint(np.max(full_graph[-1][0])),
            **kwargs,
        )
        >> dt.Lambda(
            AugmentCentroids,
            rotate=lambda: np.random.rand() * 2 * np.pi,
            translate=lambda: np.random.randn(2) * 0.05,
            flip_x=lambda: np.random.randint(2),
            flip_y=lambda: np.random.randint(2),
        )
        >> dt.Lambda(NoisyNode)
        >> dt.Lambda(NodeDropout, dropout_rate=0.03)
    )


def GetGlobalFeature(full_graph, **kwargs):
    return (
        dt.Value(full_graph)
        >> dt.Lambda(
            GetSubSet,
            randset=lambda: np.random.randint(np.max(full_graph[-1][0])),
            **kwargs,
        )
        >> dt.Lambda(
            AugmentCentroids,
            rotate=lambda: np.random.rand() * 2 * np.pi,
            translate=lambda: np.random.randn(2) * 0.05,
            flip_x=lambda: np.random.randint(2),
            flip_y=lambda: np.random.randint(2),
        )
    )


def GraphGenerator(
    loader: str = "LoadGraph",
    min_data_size=1000,
    max_data_size=2000,
    feature_function=GetFeature,
    **kwargs,
):
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
    full_graph = getattr(gt, loader, "LoadGraph")(**kwargs)

    feature = feature_function(full_graph, **kwargs)

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
