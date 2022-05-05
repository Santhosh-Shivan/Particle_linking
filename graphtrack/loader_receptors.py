from . import deeptrack as dt
import os
import glob
import pandas as pd
import numpy as np
import untangle
import tqdm

from . import graphs

_path_to_images = os.path.join(
    "..",
    "datasets",
    "{dataset}",
    "{dataset} snr {snr} density {density}",
    "{dataset} snr {snr} density {density} t{t} z0.tif",
)
_path_to_xml = os.path.join(
    "..",
    "datasets",
    "{dataset}",
    "{dataset} snr {snr} density {density}",
    "{dataset} snr {snr} density {density}.xml",
)


def ReceptorsGraphLoader(
    dataset="RECEPTOR",
    snr=7,
    density="mid",
    return_frames=False,
    **kwargs,
):
    dtspaths, df = GetAvalaibleData(dataset, snr, density=density)

    def load(paths, desc="data"):
        output = None
        idx = 0
        for path in tqdm.tqdm(paths, desc=f"Loading {desc}"):
            res = dt.LoadImage(path)()._value
            res = np.squeeze(res)
            if output is None:
                output = np.zeros((*res.shape, len(paths)))

            output[..., idx] = res
            idx += 1

        return output

    images = load(dtspaths, desc="images")
    nodesdf, props = NodeExtractor(images, df, **kwargs)

    graph = graphs.GraphExtractor(nodesdf=(nodesdf, props), **kwargs)
    if return_frames:
        return graph, images
    else:
        return graph


_default_properties = {"intensity": 100}


def NodeExtractor(
    images, df, properties: dict = _default_properties, **kwargs
):
    _properties = {"label": 1, "centroid": np.shape(images)[0]}
    _properties.update(properties)

    # Embed intesities
    pix = np.flip(df.filter(regex="centroid|frame").to_numpy(), axis=1).astype(
        int
    )
    df.loc[:, df.columns.str.contains("centroid")] /= _properties["centroid"]
    df["intensity"] = (
        images[pix[:, 0], pix[:, 1], pix[:, 2]] / _properties["intensity"]
    )
    # Append solution
    df["solution"] = 0
    return df, list(_properties.keys())


def GetAvalaibleData(dataset, snr, density):
    def to_frame(xml):
        particles = xml.root.TrackContestISBI2012.particle

        detection_list = []
        for p in range(len(particles)):
            detections = particles[p].detection
            for d in range(len(detections)):
                detection_att = {
                    "frame": int(detections[d]["t"]),
                    "centroid_x": float(detections[d]["x"]),
                    "centroid_y": float(detections[d]["y"]),
                    "label": int(p + 1),
                }

                detection_list.append(detection_att)
        df = (
            pd.DataFrame.from_dict(detection_list)
            .sort_values(by=["frame"])
            .reset_index(drop=True)
        )
        return df

    path = _path_to_images.format(
        dataset=dataset, snr=snr, density=density, t="*"
    )
    df = to_frame(
        untangle.parse(
            _path_to_xml.format(dataset=dataset, snr=snr, density=density)
        )
    )
    return glob.glob(path), df
