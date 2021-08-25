import skimage
import numpy as np
import pandas as pd
import itertools
from . import deeptrack as dt
from .embeddings import tonehot

import more_itertools as mit
from operator import is_not
from functools import partial

_default_properties = (
    "area",
    "mean_intensity",
    "perimeter",
    "eccentricity",
    "solidity",
)


def NodeExtractor(
    sequence: dt.Feature = None,
    properties: tuple = _default_properties,
    crop_size: int = 100,
    resize_shape: tuple = (96, 96),
    **kwargs
):
    """
    Extracts nodes from a sequence of frames
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames. Cell images and
        their corresponding masks.
    properties: tuple
        Properties to be extracted from the
        cell images and masks.
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

    # Get the graph information (used to create ground truth)
    graph = images.properties[0]["graph"]

    # Convert images and masks to numpy arrays
    images, masks = np.array(images), np.array(masks)

    # Properties to be extracted from the cell images and masks.
    # By default label and centroid are extracted
    properties = ("label", "centroid") + properties

    nodes, crops = [], []

    # Roll the axis -1 backwards before the for loop
    iterator = map(lambda x: np.rollaxis(x, -1), (images, masks))

    for frame_idx, (image, mask) in enumerate(zip(*iterator)):
        # Compute image properties and return them as a pandas-compatible table
        props = skimage.measure.regionprops_table(
            mask.astype(int), intensity_image=image, properties=properties
        )

        # Create dataframe with the properties
        df = pd.DataFrame(props)

        # Extract centroids from the dataframe to crop the cell images
        centroids = df.filter(like="centroid").values.astype(np.int)

        # Pad cell images with zeros before cropping
        image = np.pad(image, pad_width=crop_size, mode="constant")

        for centroid in centroids.tolist():
            # Crop cell images
            crop = image[
                centroid[0] : centroid[0] + 2 * crop_size,
                centroid[1] : centroid[1] + 2 * crop_size,
            ]

            # Resize the cropped images
            crop = skimage.transform.resize(
                crop, resize_shape, anti_aliasing=True
            )
            crops.append(crop)

        # Add frame column to the dataframe
        df.insert(loc=0, column="frame", value=frame_idx)
        nodes.append(df)

    crops = np.stack(crops, axis=-1)

    # Concatenate the dataframes in a single
    #  dataframe for the whole sequence
    nodes = pd.concat(nodes)

    # Add parent column to the dataframe
    nodes["parent"] = 0

    # Add parent ids to each element in the dataframe
    # if the id is different than zero
    parenthood = graph.values
    for child, parent in parenthood:
        nodes.loc[
            (nodes["label"] == child) & (nodes["frame"] > 0), "parent"
        ] = parent

    # Returns a solution for each node. If a node has a parent,
    # the solution is [0, 1], otherwise it is [1, 0] representing
    # cells that did not divide.
    def GetSolution(x):
        return tonehot(1, 2) if x.parent != 0 else tonehot(0, 2)

    return (
        AppendSolution(nodes.reset_index(drop=True), GetSolution),
        parenthood,
        crops,
        properties,
    )


def GetEdge(
    df: pd.DataFrame,
    start: int,
    end: int,
    radius: int,
    scales: list,
    parenthood: pd.DataFrame,
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
            regex=("label|index|feature")
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
    # array or if label_x == label_y, the solution is [0, 1], otherwise
    # it is [1, 0] representing edges are not connected.
    def GetSolution(df):
        if (np.any(np.all(df["label"][::-1] == parenthood, axis=1))) | (
            df["label_x"] == df["label_y"]
        ):
            solution = tonehot(1, 2)
        else:
            solution = tonehot(0, 2)

        return solution

    return AppendSolution(edgedf, GetSolution)


def EdgeExtractor(nodesdf, nofframes=2, **kwargs):
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

    # Create subsets from the frame list, with
    # "nofframes" elements each
    maxframe = range(0, df["frame"].max() + 1)
    windows = mit.windowed(maxframe, n=nofframes, step=1)
    windows = map(lambda x: list(filter(partial(is_not, None), x)), windows)

    # Get scales for the search radius
    scales = GetScale(nofframes)

    edgedfs = []
    for window in windows:
        # Compute the edges for each frames window
        edgedf = GetEdge(
            df, start=window[0], end=window[-1], scales=scales, **kwargs
        )
        edgedfs.append(edgedf)

    # Concatenate the dataframes in a single
    # dataframe for the whole set of edges
    edgesdfs = pd.concat(edgedfs)

    return edgesdfs


def GetScale(nofframes):
    # TODO: implement this function
    return [1] * nofframes


def AppendSolution(df, func, **kwargs):
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
    df.loc[:, "solution"] = df.apply(func, axis=1, **kwargs)
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


def GraphExtractor(sequence: dt.Feature = None, **kwargs):
    """
    Extracts the graph from a sequence of frames
    Parameters
    ----------
    sequence: dt.Feature
        A sequence of frames.
    """
    # Extract nodes from the sequence
    nodesdf, parenthood, _, properties = NodeExtractor(sequence, **kwargs)

    # Extract edges and edge features from nodes
    edgesdf = EdgeExtractor(
        nodesdf, nofframes=3, parenthood=parenthood, **kwargs
    )

    # Split the nodes dataframe into features and labels
    nodefeatures, nfsolution = DataframeSplitter(
        nodesdf, props=properties, **kwargs
    )

    # Split the edges dataframe into features, sparse adjacency
    # matrix, and labels
    edgefeatures, sparseadjmtx, efsolution = DataframeSplitter(
        edgesdf, props=("feature",), **kwargs
    )

    return (
        (nodefeatures, edgefeatures, sparseadjmtx),
        (nfsolution, efsolution),
    )


def SelfDuplicateEdgeAugmentation(edges, maxnofedges=None, idxs=None):
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
        itr, edge = items
        edge = np.array(edge)

        # Computes the number of additional edges to add
        nofedges = np.shape(edge)[0]
        offset = maxnofedges - nofedges

        # Randomly selects edges to duplicate
        if use_idxs:
            idx = idxs[itr]
        else:
            idx = np.random.choice(nofedges, offset, replace=False)
            idxs.append(idx)

        # Initiliazes the weights to 1
        w = np.ones(nofedges)

        # Add indexes to the weight matrix
        w = np.stack((np.arange(0, w.shape[0]), w), axis=1)

        # Balances repeated edges
        w[idx, 1] = 0.5

        # Augment the weights
        w = np.concatenate((w, w[idx]), axis=0)
        weights.append(w)

        # Duplicate the edges
        edge = np.concatenate((edge, edge[idx, :]), axis=0)
        return edge

    return list(map(inner, enumerate(edges))), weights, idxs
