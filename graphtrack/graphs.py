import skimage
import numpy as np
import pandas as pd
import itertools
import time
from . import deeptrack as dt

import tqdm

# from .embeddings import tonehot

import more_itertools as mit
from operator import is_not
from functools import partial

import graphtrack as gt
from .extractors import from_masks

_default_properties = {
    "area": 20000,
    "mean_intensity": 127.5,
    "perimeter": 800,
    "eccentricity": 1,
    "solidity": 1,
}


def NodeExtractor(
    sequence: dt.Feature = None,
    graph=None,
    properties: dict = _default_properties,
    extractor_function=from_masks,
    **kwargs
):
    """
    Extracts nodes from a sequence of frames
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames. Cell images and
        their corresponding masks.
    properties: dict
        A dictionary containing the properties
        of the nodes to be extracted from the
        cell images and masks. A normalization
        factor is also defined for each property.
    crop_size: int
        Size of the cropped cell images.
    resize_shape: tuple, optional
        Shape of the resized cropped cell images.
    Returns
    -------
    nodes: pd.DataFrame
        A dataframe containing the extracted
        properties of the cell images.
    Parenthood: numpy.ndarray
        A numpy array containing the parent-child
        relationships for the nodes.
    crops: numpy.ndarray
        A numpy array containing the cropped cell
        images.
    """

    # Extract cell images and masks
    images, masks = sequence

    nodes, properties_names = extractor_function(
        images, masks, properties, **kwargs
    )

    # Concatenate the dataframes in a single
    #  dataframe for the whole sequence
    nodes = pd.concat(nodes)

    # Add parent column to the dataframe
    nodes["parent"] = 0

    # Returns a solution for each node. If a node has a parent,
    # the solution is 1, otherwise it is 0 representing
    # cells that did not divide.
    def GetSolution(x):
        return (1.0 if x.parent != 0 else 0.0,)

    # Add parent ids to each element in the dataframe
    # if the id is different than zero
    if not (graph is None):
        parenthood = graph.values
        for child, parent in parenthood:
            nodes.loc[
                (nodes["label"] == child) & (nodes["frame"] > 0), "parent"
            ] = parent

        return (
            AppendSolution(
                nodes.reset_index(drop=True), GetSolution, append_weight=False
            ),
            parenthood,
            properties_names,
        )
    else:
        parenthood = np.array([-1, 1])[np.newaxis, :]
        return (
            AppendSolution(
                nodes.reset_index(drop=True), GetSolution, append_weight=False
            ),
            parenthood,
            properties_names,
        )


def GetEdge(
    df: pd.DataFrame,
    start: int,
    end: int,
    radius: int,
    scales: list,
    parenthood: pd.DataFrame,
    rare_event_weight: float = 1,
    **kwargs
):
    """
    Extracts the edges from a windowed sequence of frames
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted node properties.
    start: int
        Start frame of the edge.
    end: int
        End frame of the edge.
    radius: int
        Search radius for the edge (pixel units).
    scales: list
        Scales to be used for the edge extraction.
    parenthood: list
        A list of parent-child relationships
        between nodes.
    Returns
    -------
    edges: pd.DataFrame
        A dataframe containing the extracted
        properties of the edges.
    """
    # Add a column for the indexes of the nodes
    df.loc[:, "index"] = df.index

    # Filter the dataframe to only include the the
    # frames, centroids, labels, and indexes
    df = df.loc[(df["frame"] >= start) & (df["frame"] <= end)].filter(
        regex="(frame|centroid|label|index)"
    )

    # Merge columns contaning the centroids into a single column of
    # numpy arrays, i.e., centroid = [centroid_x, centroid_y,...]
    df.loc[:, "centroid"] = df.filter(like="centroid").apply(np.array, axis=1)

    # Add key column to the dataframe
    df["key"] = 1

    # Group the dataframe by frame
    df = df.groupby(["frame"])
    dfs = [_df for _, _df in df]

    edges = []
    for scale, (dfi, dfj) in zip(scales, itertools.product(dfs[0:1], dfs[1:])):

        # Merge the dataframes for frames i and j
        combdf = pd.merge(dfi, dfj, on="key").drop("key", axis=1)

        # Compute distances between centroids
        combdf.loc[:, "diff"] = combdf.centroid_x - combdf.centroid_y
        combdf.loc[:, "feature-dist"] = combdf["diff"].apply(
            lambda diff: np.linalg.norm(diff, ord=2)
        )

        # Filter out edges with a feature-distance less than scale * radius
        combdf = combdf[combdf["feature-dist"] < scale * radius].filter(
            regex=("frame|label|index|feature")
        )
        edges.append(combdf)
    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgedf = pd.concat(edges)

    # Merge columns contaning the labels into a single column
    # of numpy arrays, i.e., label = [label_x, label_y]
    edgedf.loc[:, "label"] = edgedf.filter(like="label").apply(
        np.array, axis=1
    )

    # Returns a solution for each edge. If label is the parenthood
    # array or if label_x == label_y, the solution is 1, otherwise
    # it is 0 representing edges are not connected.
    def GetSolution(x):
        if np.any(np.all(x["label"][::-1] == parenthood, axis=1)):
            solution = 1.0
            weight = rare_event_weight
        elif x["label_x"] == x["label_y"]:
            solution = 1.0
            weight = 1.0
        else:
            solution = 0.0
            weight = 1.0

        return solution, weight

    # Initialize solution and weight columns
    edgedf[["solution", "weight"]] = [0.0, 1.0]

    return AppendSolution(edgedf, GetSolution)


def EdgeExtractor(nodesdf, nofframes=3, **kwargs):
    """
    Extracts edges from a sequence of frames
    Parameters
    ----------
    nodesdf: pd.DataFrame
        A dataframe containing the extracted node properties.
    noframes: int
        Number of frames to be used for
        the edge extraction.
    """
    # Create a copy of the dataframe to avoid overwriting
    df = nodesdf.copy()

    edgedfs = []
    sets = np.unique(df["set"])
    for setid in tqdm.tqdm(sets):
        df_set = df.loc[df["set"] == setid].copy()

        # Create subsets from the frame list, with
        # "nofframes" elements each
        maxframe = range(0, df_set["frame"].max() + 1 + nofframes)
        windows = mit.windowed(maxframe, n=nofframes, step=1)
        windows = map(
            lambda x: list(filter(partial(is_not, None), x)), windows
        )
        windows = list(windows)[:-2]

        # Get scales for the search radius
        scales = GetScale(nofframes)

        for window in windows:
            # remove excess frames
            window = [elem for elem in window if elem <= df_set["frame"].max()]

            # Compute the edges for each frames window
            edgedf = GetEdge(
                df_set,
                start=window[0],
                end=window[-1],
                scales=scales,
                **kwargs
            )
            edgedf["set"] = setid
            edgedfs.append(edgedf)

    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgesdfs = pd.concat(edgedfs)

    return edgesdfs


def GetScale(nofframes):
    # TODO: implement this function
    return [1] * nofframes


def AppendSolution(df, func, append_weight=True, **kwargs):
    """
    Appends a solution to the dataframe
    Parameters
    ----------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    func: function
        A function that takes a dataframe
        and returns a solution.
    Returns
    -------
    df: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes with
        a solution.
    """
    # Get solution
    df.loc[:, "solution"] = df.apply(lambda x: func(x)[0], axis=1, **kwargs)

    if append_weight:
        # Get weight
        df.loc[:, "weight"] = df.apply(lambda x: func(x)[1], axis=1, **kwargs)

    return df


def DataframeSplitter(df, props: tuple, to_array=True, **kwargs):
    """
    Splits a dataframe into features and labels
    Parameters
    ----------
    dt: pd.DataFrame
        A dataframe containing the extracted
        properties for the edges/nodes.
    atts: list
        A list of attributes to be used as features.
    to_array: bool
        If True, the features are converted to numpy arrays.
    Returns
    -------
    X: np.ndarray or pd.DataFrame
        Features.
    """
    # Extract features from the dataframe
    if len(props) == 1:
        features = df.filter(like=props[0])
    else:
        regex = ""
        for prop in props[1:]:
            regex += prop + "|"
        regex = regex[:-1]
        features = df.filter(regex=regex)

    # Extract labels from the dataframe
    label = df["solution"]

    if "index_x" in df:
        outputs = [features, df.filter(like="index"), label]
    else:
        outputs = [features, label]

    # Convert features to numpy arrays if needed
    if to_array:
        outputs = list(
            map(
                lambda x: np.stack(x.apply(np.array, axis=1).values),
                outputs[:-1],
            )
        ) + [
            np.stack(outputs[-1].values),
        ]

    return outputs


def GraphExtractor(
    sequence: dt.Feature = None,
    validation=False,
    nodesdf=None,
    global_property=None,
    **kwargs
):
    """
    Extracts the graph from a sequence of frames
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames.
    """

    if nodesdf is None:
        print("Building node features...")
        # Extract nodes from the sequence
        nodesdf, parenthood, properties = NodeExtractor(sequence, **kwargs)
    else:
        print("Loading node features...")
        nodesdf, properties = nodesdf
        # Dummy parenthood array
        parenthood = np.array([-1, -1])[np.newaxis, :]

    # Extract edges and edge features from nodes
    print("Creating graph edges...")
    edgesdf = EdgeExtractor(nodesdf, parenthood=parenthood, **kwargs)

    # Split the nodes dataframe into features and labels
    nodefeatures, nfsolution = DataframeSplitter(
        nodesdf, props=properties, **kwargs
    )

    # Split the edges dataframe into features, sparse adjacency
    # matrix, and labels
    edgefeatures, sparseadjmtx, efsolution = DataframeSplitter(
        edgesdf, props=("feature",), **kwargs
    )

    if np.any(validation):
        # Add frames to the adjacency matrix
        frames = edgesdf.filter(like="frame").to_numpy()
        sparseadjmtx = np.concatenate((frames, sparseadjmtx), axis=-1)

        # Add frames to the node features matrix
        frames = nodesdf.filter(like="frame").to_numpy()
        nodefeatures = np.concatenate((frames, nodefeatures), axis=-1)

        # Weights edges equally
        edgeweights = np.ones(sparseadjmtx.shape[0])
    else:
        # Computes weight matrix from the adjacency matrix
        edgeweights = edgesdf["weight"].to_numpy()

    # Add indexes to the weight matrix
    edgeweights = np.stack(
        (np.arange(0, edgeweights.shape[0]), edgeweights), axis=1
    )

    # Extract set ids
    nodesets = nodesdf["set"].to_numpy()
    edgesets = edgesdf["set"].to_numpy()

    if global_property is None:
        global_property = np.zeros(np.unique(nodesdf["set"]).shape[0])

    return (
        (nodefeatures, edgefeatures, sparseadjmtx, edgeweights),
        (nfsolution, efsolution, global_property),
        (nodesets, edgesets),
    )


def SelfDuplicateEdgeAugmentation(edges, w, maxnofedges=None, idxs=None):
    """
    Augments edges by randomly adding edges to the graph. The new edges
    are copies of the original edges, and their influence to the solution
    is pondered using uniformely distributed weights.
    Parameters
    ----------
    edges : list of numpy arrays
        List of edges to augment
    maxnofedges : int, optional
        Maximum number of edges to add to the graph. If None, the maximum
        number of edges is set to the number of edges in the graph.
    idxs : list of numpy arrays, optional
        List of indices of the edges to be augmented. If None, the edges
        are selected randomly.
    """
    weights = []
    use_idxs = True if idxs else False
    idxs = idxs if use_idxs else []

    def inner(items):
        itr, (edge, w) = items
        edge = np.array(edge)
        w = np.array(w, dtype=np.float64)

        # Computes the number of additional edges to add
        nofedges = np.shape(edge)[0]
        offset = maxnofedges - nofedges

        # Randomly selects edges to duplicate
        if use_idxs:
            idx = idxs[itr]
        else:
            # privileges the edges denoting rare events
            probability = np.ones(nofedges)
            probability[w[:, 1] > 1] = 2
            probability = probability / np.sum(probability)

            # TODO: Design more efficient to replicate edges
            if offset > len(edge):
                idx = np.random.choice(
                    nofedges, offset, replace=True, p=list(probability)
                )
            else:
                idx = np.random.choice(
                    nofedges, offset, replace=False, p=list(probability)
                )
            idxs.append(idx)

        # Balances repeated edges
        w[idx, 1] /= 2

        # Augment the weights
        w = np.concatenate((w, w[idx]), axis=0)
        weights.append(w)

        # Duplicate the edges
        edge = np.concatenate((edge, edge[idx]), axis=0)
        return edge

    return list(map(inner, enumerate(zip(edges, w)))), weights, idxs
