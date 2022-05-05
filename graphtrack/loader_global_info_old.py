from PIL.Image import alpha_composite
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


def process_dataset(dataset, seq_length, scale_traj, shift_traj):
    trajectories = []
    trajs = dataset[:, 2:]

    # scale, shift, strip
    for i in range(len(trajs)):
        trajs[i][:seq_length] = (
            scale_traj * (trajs[i][:seq_length])
            + np.random.rand(1) * shift_traj
        )
        trajs[i][seq_length:] = (
            scale_traj * (trajs[i][seq_length:])
            + np.random.rand(1) * shift_traj
        )
        trajectories.append(
            np.transpose([trajs[i][:seq_length], trajs[i][seq_length:]])
        )
    return trajectories


def DataSimulator():
    root = dt.Arguments(
        N=lambda: np.random.randint(3, 10),
        alpha=lambda: np.round(0.4 + 1.2 * np.random.rand(), 1),
        model=2,
        seq_length=50,
        scale_traj=5,
        shift_traj=100,
        dataset=lambda seq_length, N, alpha, model: andi.andi_datasets().create_dataset(
            T=seq_length, N=N, exponents=alpha, models=model, dimension=2
        ),
        trajectories=lambda dataset, seq_length, scale_traj, shift_traj: process_dataset(
            dataset, seq_length, scale_traj, shift_traj
        ),
        traj_pointer=0,
        ri=lambda N: np.random.uniform(1.35, 1.55, N),
    )

    optics = dt.Brightfield(
        wavelength=633e-9,
        NA=1,
        resolution=3.6e-6,
        magnification=1,
        refractive_index_medium=1.33,
        upscale=4,
        output_region=(0, 0, 128, 128),
    )

    _particle = dt.Sphere(
        trajectories=root.trajectories,
        trajectory=lambda replicate_index, trajectories: trajectories[
            replicate_index[-1]
        ],
        traj_pointer=lambda replicate_index: replicate_index[-1],
        alpha=root.alpha,
        number_of_particles=root.N,
        traj_length=root.seq_length,
        position=lambda trajectory: trajectory[0],
        z=1000 / 3.6,
        position_unit="pixel",
        radius=5e-6,
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
    labels = np.zeros([seq_length, n_particles, 128, 128, 3])

    X, Y = np.meshgrid(
        np.arange(0, labels.shape[2]), np.arange(0, labels.shape[3])
    )

    for i in range(seq_length):
        pos = image_sequence[i].get_property("position", get_one=False)
        ri = image_sequence[i].get_property("refractive_index", get_one=False)
        particle_index = image_sequence[i].get_property(
            "traj_pointer", get_one=False
        )

        for j in range(n_particles):
            distance_map = (X - pos[j][1]) ** 2 + (Y - pos[j][0]) ** 2
            labels[i][j][distance_map < 3 ** 5, 0] = 1
            labels[i][j][distance_map < 3 ** 2, 1] = particle_index[j] + 1
            labels[i][j][distance_map < 3 ** 2, 2] = ri[j]

    return labels


def LoadGraphExponent(dataset_size, return_frames=False, **kwargs):
    loader = DataSimulator()

    def load(loader, dataset_size, desc="data"):
        images, masks, alphas = [], [], []
        for s in tqdm.tqdm(range(dataset_size), desc=f"Loading {desc}"):
            sequence = loader.update().resolve()
            mask = get_targets(sequence)

            # Global property
            alpha = sequence[0].get_property("alpha")

            images.append(sequence)
            masks.append(mask)
            alphas.append(alpha)

        return (images, masks), alphas

    sequences, alphas = load(loader, dataset_size, "images")
    graph = GraphExtractor(
        sequences,
        extractor_function=from_multi_masks,
        global_property=alphas,
        **kwargs,
    )
    if return_frames:
        return graph, sequences
    else:
        return graph
