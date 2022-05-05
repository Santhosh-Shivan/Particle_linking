import skimage
import numpy as np
import pandas as pd

import tqdm


def from_masks(images, masks, properties, **kwargs):
    nodes = []

    # Convert images and masks to numpy arrays
    images, masks = np.array(images), np.array(masks)

    # Properties to be extracted from the cell images and masks.
    # A normalization factor is also defined for each property.
    # By default label and centroid are extracted
    _properties = {"label": 1, "centroid": np.shape(images[..., 0])}
    _properties.update(properties)

    # Extract the names of the properties
    properties_names = list(_properties.keys())

    # Roll the axis -1 backwards before the for loop
    iterator = map(lambda x: np.rollaxis(x, -1), (images, masks))

    for frame_idx, (image, mask) in enumerate(zip(*iterator)):
        # Compute image properties and return them as a pandas-compatible table
        props = skimage.measure.regionprops_table(
            mask.astype(int),
            intensity_image=image,
            properties=properties_names,
        )

        # Create dataframe with the properties
        df = pd.DataFrame(props)

        # normalize the properties
        for prop in properties_names:
            df_filtered = df.filter(like=prop) / _properties[prop]
            df.loc[:, df_filtered.columns] = df_filtered

        # Cast label to int
        df["label"] = df["label"].astype(int)
        df["set"] = 0

        # Add frame column to the dataframe
        df.insert(loc=0, column="frame", value=frame_idx)
        nodes.append(df)

    return (nodes, properties_names)


def from_multi_masks(images, masks, properties, extra_properties, **kwargs):
    nodes = []

    # Properties to be extracted from the cell images and masks.
    # A normalization factor is also defined for each property.
    # By default label and centroid are extracted

    # Extract the names of the extra properties
    extra_properties_names = list(extra_properties.keys())
    _extra_properties_names = extra_properties_names.copy()

    if "centroid_z" in extra_properties.keys():
        _properties = {
            "label": 1,
            "centroid": (
                *np.shape(masks[0][..., 0])[-2:],
                extra_properties["centroid_z"],
            ),
        }
        _extra_properties_names.remove("centroid_z")
    else:
        _properties = {"label": 1, "centroid": np.shape(masks[0][..., 0])[-2:]}

    _properties.update(properties)

    # Extract the names of the properties
    properties_names = list(_properties.keys())

    _properties.update(extra_properties)

    for batch, (_images, _masks) in tqdm.tqdm(
        enumerate(zip(images, masks)), "Building nodes", total=len(masks)
    ):

        _masks_merged = np.argmax(
            np.concatenate(
                [
                    np.zeros((np.shape(_masks)[0], 1, *np.shape(_masks)[-3:])),
                    _masks,
                ],
                axis=1,
            ),
            axis=1,
        )

        # Roll the axis -1 backwards before the for loop
        # iterator = map(lambda x: np.rollaxis(x, 0), (_images, _masks_merged))

        for frame_idx, (image, mask) in enumerate(zip(_images, _masks_merged)):

            # Convert images and masks to numpy arrays
            image = np.array(image)

            # Compute image properties and return them as a pandas-compatible table
            # TODO: compute the centroids from extra masks intead for more accurate results
            props = skimage.measure.regionprops_table(
                mask[..., 0].astype(int),
                intensity_image=image[..., 0],
                properties=properties_names,
            )

            # Create dataframe with the properties
            df = pd.DataFrame(props)

            # Retrieve extra properties
            extra_props = []
            for i, extra_prop_name in enumerate(extra_properties_names):
                extra_masks = _masks[frame_idx, ..., i + 1].flatten()
                _, idx = np.unique(extra_masks, return_index=True)
                extra_props.append(
                    dict(
                        zip(
                            [extra_prop_name, "label"],
                            [
                                extra_masks[np.sort(idx)][1:],
                                np.arange(1, len(idx)),
                            ],
                        )
                    )
                )

            # Append extra properties to the dataframe
            df = (
                pd.concat(
                    [
                        df,
                        *[
                            pd.DataFrame(extra_prop, columns=extra_prop.keys())
                            for extra_prop in extra_props
                        ],
                    ],
                )
                .groupby("label")
                .agg(["first"])
                .reset_index()
            )
            df.columns = df.columns.droplevel(level=1)

            # normalize the properties
            for prop in properties_names + _extra_properties_names:
                df_filtered = df.filter(like=prop) / _properties[prop]
                df.loc[:, df_filtered.columns] = df_filtered

            # Cast label to int
            df["label"] = df["label"].astype(int)
            df["set"] = batch

            # Add frame column to the dataframe
            df.insert(loc=0, column="frame", value=frame_idx)

            # Replace NaNs with mean of column
            # TODO: check nans from simulation
            feature_means = df.mean()
            df = df.fillna(feature_means)

            nodes.append(df)

    return (nodes, properties_names + extra_properties_names)
