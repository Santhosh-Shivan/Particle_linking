from numpy import random
from . import deeptrack as dt
import numpy as np

import itertools
from typing import List
import scipy
import pandas as pd

import tqdm
from . import graphs

import andi

import graphtrack as gt
from .graphs import NodeExtractor, EdgeExtractor, GraphExtractor
from .extractors import from_multi_masks

import os
import glob

import re

import untangle


def process_dataset(dataset, seq_length, scale_traj, shift_traj):
    trajectories = []
    trajs = dataset[:, 2:]

    # scale, shift, strip
    for i in range(len(trajs)):
        trajs[i][:seq_length] = (
            scale_traj * (trajs[i][:seq_length])
            + np.random.rand(1) * shift_traj
        )  # scale x
        trajs[i][seq_length : 2 * seq_length] = (
            scale_traj * (trajs[i][seq_length : 2 * seq_length])
            + np.random.rand(1) * shift_traj
        )  # scale y
        trajs[i][2 * seq_length :] = scale_traj * (
            trajs[i][2 * seq_length :]
        ) + np.random.rand(1) * (
            shift_traj
        )  # scale z
        trajectories.append(
            np.transpose(
                [
                    trajs[i][:seq_length],
                    trajs[i][seq_length : 2 * seq_length],
                    trajs[i][2 * seq_length :],
                ]
            )
        )

    return trajectories


def DataSimulator(_type):
    if _type == "diffusion_type":
        root = dt.Arguments(
            N=lambda: np.random.randint(3, 7),
            alpha=1,  # lambda: np.round(0.4 + 1.2 * np.random.rand(), 1),
            model=lambda: np.random.randint(
                0, 3, 1
            ),  # change it a int for alpha prediction
            seq_length=100,  # lambda: np.random.randint(50, 80),
            scale_traj=lambda: np.random.uniform(2, 5),
            shift_traj=100,
            dataset=lambda seq_length, N, alpha, model: andi.andi_datasets().create_dataset(
                T=seq_length,
                N=N,
                exponents=alpha,
                models=list(model),
                dimension=3,
            ),
            trajectories=lambda dataset, seq_length, scale_traj, shift_traj: process_dataset(
                dataset, seq_length, scale_traj, shift_traj
            ),
            traj_pointer=0,
            ri=lambda N: np.random.uniform(1.35, 1.55, N),
        )
    elif _type == "alpha":
        root = dt.Arguments(
            N=lambda: np.random.randint(3, 10),
            alpha=lambda: np.round(0.1 + 1.8 * np.random.rand(), 2),
            model=2,
            seq_length=100,
            scale_traj=lambda: np.random.uniform(2, 5),
            shift_traj=100,
            dataset=lambda seq_length, N, alpha, model: andi.andi_datasets().create_dataset(
                T=seq_length, N=N, exponents=alpha, models=model, dimension=3
            ),
            trajectories=lambda dataset, seq_length, scale_traj, shift_traj: process_dataset(
                dataset, seq_length, scale_traj, shift_traj
            ),
            traj_pointer=0,
            ri=lambda N: np.random.uniform(1.35, 1.55, N),
        )
    else:
        raise NotImplementedError

    optics = dt.Brightfield(
        wavelength=633e-9,
        NA=0.3,
        resolution=3.6e-6,
        magnification=5,
        refractive_index_medium=1.33,
        aberration=dt.GaussianApodization(sigma=0.06),
        upscale=4,
        output_region=(0, 0, 128, 128),
    )

    _particle = dt.Sphere(
        trajectories=root.trajectories,
        trajectory=lambda replicate_index, trajectories: dt.units.pixel
        * trajectories[replicate_index[-1]],
        traj_pointer=lambda replicate_index: replicate_index[-1],
        diffusion_type=root.model,
        alpha=root.alpha,
        number_of_particles=root.N,
        traj_length=root.seq_length,
        position=lambda trajectory: trajectory[0],
        position_unit="pixel",
        radius=2e-6,  # lambda: 1e-6 + 10e-6*np.random.rand(),
        ri=root.ri,
        refractive_index=lambda: np.random.uniform(1.35, 1.45, 1),
    )

    particle = dt.Sequential(
        _particle,
        position=lambda trajectory, sequence_step: trajectory[sequence_step],
    )

    sample = (
        root
        >> optics(particle ^ _particle.number_of_particles)
        >> dt.Gaussian(mu=0, sigma=0.025)
    )

    return dt.Sequence(
        sample,
        trajectory=root.trajectories,
        alpha=_particle.alpha,
        sequence_length=lambda trajectory: len(trajectory[0]),
    )


def get_targets(image_sequence):
    seq_length = image_sequence[0].get_property("sequence_length")
    n_particles = image_sequence[0].get_property("number_of_particles")
    m = image_sequence[0].get_property("magnification")
    labels = np.zeros([seq_length, n_particles, 128, 128, 3])

    X, Y = np.meshgrid(
        np.arange(0, labels.shape[2]), np.arange(0, labels.shape[3])
    )

    for i in range(seq_length):
        pos = image_sequence[i].get_property("position", get_one=False)
        ri = image_sequence[i].get_property("refractive_index", get_one=False)

        for j in range(n_particles):
            z_dist = pos[j][2]
            distance_map = (X - pos[j][1]) ** 2 + (Y - pos[j][0]) ** 2
            labels[i][j][distance_map < (z_dist) * m, 0] = 1
            labels[i][j][distance_map < 3 ** 2, 1] = pos[j][2]
            labels[i][j][distance_map < 3 ** 2, 2] = ri[j]

    return labels


def LoadGraphExponent(
    dataset_size,
    return_frames=False,
    global_prop_name="diffusion_type",
    **kwargs,
):
    loader = DataSimulator(_type=global_prop_name)

    def load(loader, dataset_size, desc="data"):
        images, masks, global_props = [], [], []
        for s in tqdm.tqdm(range(dataset_size), desc=f"Loading {desc}"):
            sequence = loader.update().resolve()
            mask = get_targets(sequence)

            # Global property
            global_prop = sequence[0].get_property(global_prop_name)

            images.append(sequence)
            masks.append(mask)
            global_props.append(global_prop)

        return (images, masks), global_props

    sequences, global_props = load(loader, dataset_size, "images")
    graph = GraphExtractor(
        sequences,
        extractor_function=from_multi_masks,
        global_property=global_props,
        **kwargs,
    )
    if return_frames:
        return graph, sequences
    else:
        return graph


_default_properties = {"intensity": 70.0}


def NodeExtractor(
    paths=None,
    properties: dict = _default_properties,
    extract_solution=False,
    **kwargs,
):
    def to_frame(xml):
        particles = xml.root.GlobalHeterogenousInfo.particle

        detection_list = []
        for p in range(len(particles)):
            detections = particles[p].detection
            for d in range(len(detections)):
                detection_att = {
                    "frame": int(detections[d]["frame"]),
                    "centroid_x": float(detections[d]["x"]),
                    "centroid_y": float(detections[d]["y"]),
                    "intensity": float(detections[d]["intensity"]),
                    "radius": float(detections[d]["radius"]),
                    "label": int(p + 1),
                }
                if extract_solution:
                    detection_att["solution"] = float(
                        detections[d]["solution"]
                    )

                detection_list.append(detection_att)
        df = (
            pd.DataFrame.from_dict(detection_list)
            .sort_values(by=["frame"])
            .reset_index(drop=True)
        )
        df["solution"] = 0.0
        return df, np.array([0.0])

    _properties = {
        "label": 1,
        "centroid": np.array([128.0, 128.0]).astype(np.float32),
    }
    _properties.update(properties)

    dfs, global_property = [], []
    for batch, path in tqdm.tqdm(
        enumerate(paths), total=len(paths), desc="Loading xmls"
    ):
        df, global_prop = to_frame(untangle.parse(path))

        # Normalize features
        df.loc[:, df.columns.str.contains("centroid")] = np.round(
            df.loc[:, df.columns.str.contains("centroid")]
            / _properties["centroid"],
            3,
        )
        df["intensity"] = np.round(
            df["intensity"] / _properties["intensity"], 3
        )
        df["radius"] = np.round(df["radius"] / _properties["radius"], 3)

        # Append solution
        if not ("solution" in df.columns):
            df["solution"] = 0.0

        # Append set
        df["set"] = batch
        dfs.append(df)
        global_property.append(global_prop)

    dfs = pd.concat(dfs)
    # dfs = dfs.clip(lower=0)

    return dfs, list(_properties.keys()), global_property


_path_to_xml = os.path.join(
    ".", "xml-generators", "xml_data", "{_type}", "{mode}", "*.xml"
)


def LoadGraphXml(_type="mixed_alpha", mode="training", **kwargs):
    PATH_TO_DATASET = glob.glob(_path_to_xml.format(_type=_type, mode=mode))
    nodesdf, props, global_property = NodeExtractor(PATH_TO_DATASET, **kwargs)

    print(PATH_TO_DATASET)

    graph = graphs.GraphExtractor(
        nodesdf=(nodesdf, props), global_property=global_property, **kwargs
    )
    return graph
