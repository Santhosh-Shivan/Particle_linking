""" Generators to continuously resolve features.

Classes
-------
Generator
    Base class for a generator.
ContinuousGenerator
    Generator that asynchronously expands the dataset
"""

from .augmentations import Affine
import numpy as np

from typing import List

import numpy as np
import tensorflow.keras as keras

from .features import Feature
from .image import Image, array
import threading
import random
import time


class Generator(keras.utils.Sequence):
    """Base class for a generator.

    Generators continously update and resolve features, and allow other
    frameworks to continuously access new features.
    """

    def generate(
        self,
        feature,
        label_function=None,
        batch_function=lambda image: image,
        batch_size=1,
        repeat_batch=1,
        ndim=4,
        shuffle_batch=True,
        ensure_contains_classes=[],
        feature_kwargs={},
    ):
        """Create generator instance.

        Parameters
        ----------
        feature : Feature
            The feature to resolve images from.
        label_function : Callable[Image] -> array_like
            Function that returns the label corresponding to an image.
        batch_size : int
            Number of images per batch.
        repeat_batch : int
            How many times to reuse the same batch before creating a new batch.
        shuffle_batch : bool
            If True, the batches are shuffled before outputting.
        ndim : int
            Expected number of dimensions of the output.
        feature_kwargs : dict
            Set of options to pass to the feature when resolving
        ensure_contains_classes : list
            Ensures each batch contains all classes. Label is assumed to be sparse.
            ´batch_size´ needs to be larger than the number of classes
        """

        get_one = self._get_from_map(feature, feature_kwargs)
        while True:
            with threading.Lock():
                batch = []
                labels = []
                # Yield batch_size results

                contains_class = [[] for _ in range(len(ensure_contains_classes))]
                while (not all(contains_class)) or len(batch) < batch_size:
                    image = next(get_one)
                    batch.append(batch_function(image))

                    if label_function:
                        labels.append(label_function(image))

                    for list_index, class_list in enumerate(contains_class):
                        if np.any(labels[-1] == ensure_contains_classes[list_index]):
                            class_list.append(len(batch) - 1)

                number_of_sub_batches = len(batch) // batch_size
                for _ in range(number_of_sub_batches):
                    sub_batch = []
                    sub_labels = []

                    if number_of_sub_batches > 1:
                        for sample_from in contains_class + [
                            list(range(len(batch)))
                        ] * (batch_size - len(contains_class)):
                            index = np.random.choice(sample_from)
                            sub_batch.append(batch[index])
                            sub_labels.append(labels[index])
                    else:
                        sub_batch = batch
                        sub_labels = labels

                    if shuffle_batch:
                        self._shuffle(sub_batch, sub_labels)

                    sub_batch = array(sub_batch)
                    sub_labels = array(sub_labels)

                    # Console found batch_size with results
                    if sub_batch.ndim > ndim:
                        dims_to_remove = sub_batch.ndim - ndim
                        sub_batch = np.reshape(
                            sub_batch,
                            (-1, *sub_batch.shape[dims_to_remove + 1 :]),
                        )
                        sub_labels = np.reshape(
                            sub_labels,
                            (-1, *sub_labels.shape[dims_to_remove + 1 :]),
                        )

                    elif sub_batch.ndim < ndim:
                        Warning(
                            "Incorrect number of dimensions. Found {0} with {1} dimensions, expected {2}.".format(
                                sub_batch.shape, sub_batch.ndim, ndim
                            )
                        )
                    for _ in range(repeat_batch):
                        if label_function:
                            yield sub_batch, sub_labels
                        else:
                            yield sub_batch

    def _get(self, features: Feature or List[Feature], feature_kwargs) -> Image:
        # Updates and resolves a feature or list of features.
        if isinstance(features, List):
            for feature in features:
                feature.update()
            return [feature.resolve(**feature_kwargs) for feature in reversed(features)]
        else:
            features.update()
            return features.resolve(**feature_kwargs)

    def _shuffle(self, x, y):
        # Shuffles the batch and labels equally along the first dimension
        import random

        start_state = random.getstate()
        random.shuffle(x)
        random.setstate(start_state)
        random.shuffle(y)

    def _get_from_map(self, features, feature_kwargs):
        # Continuously yield the output of _get
        while True:
            yield self._get(features, feature_kwargs)


class ContinuousGenerator(keras.utils.Sequence):

    """Generator that asynchronously expands the dataset.

    Generator that aims to speed up the training of networks by striking a
    balance between the generalization gained by generating new images
    and the speed gained from reusing images. The generator will continuously
    create new training data during training, until `max_data_size` is reached,
    at which point the oldest data point is replaced.

    The generator is expected to be used with the python "with" statement, which
    ensures that the generator worker is consumed correctly.

    Parameters
    ----------
    feature : Feature
        The feature to resolve images from.
    label_function : Callable[Image or list of Image] -> array_like
        Function that returns the label corresponding to a feature output.
    batch_function : Callable[Image or list of Image] -> array_like, optional
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
    """

    def __init__(
        self,
        feature,
        label_function=lambda image: image[1]
        if isinstance(image, (list, tuple))
        else None,
        batch_function=lambda image: image[0]
        if isinstance(image, (list, tuple))
        else image,
        augmentation=None,
        min_data_size=None,
        max_data_size=np.inf,
        batch_size=32,
        shuffle_batch=True,
        ndim=4,
        max_epochs_per_sample=np.inf,
        verbose=1,
    ):

        if min_data_size is None:
            min_data_size = min(batch_size * 10, max_data_size - 1)

        assert (
            min_data_size < max_data_size
        ), "max_data_size needs to be larger than min_data_size"

        self.min_data_size = min_data_size
        self.max_data_size = max_data_size

        self.feature = feature
        self.label_function = label_function
        self.batch_function = batch_function
        self.batch_size = batch_size
        self.shuffle_batch = shuffle_batch
        self.max_epochs_per_sample = max_epochs_per_sample
        self.ndim = ndim
        self.augmentation = augmentation

        self.lock = threading.Lock()
        self.data = []
        self.batch = []
        self.labels = []
        self.exit_signal = False
        self.epoch = 0
        self._batch_size = 32
        self.verbose = verbose
        self.data_generation_thread = threading.Thread(
            target=self._continuous_get_training_data, daemon=True
        )
        self.new_epoch = True

    def __enter__(self):
        try:
            self.epoch = 0
            self.exit_signal = False
            try:
                self.data_generation_thread.start()
            except RuntimeError:
                self.data_generation_thread = threading.Thread(
                    target=self._continuous_get_training_data, daemon=True
                )
                self.data_generation_thread.start()

            while len(self.data) < self.min_data_size:
                if self.verbose > 0:
                    print(
                        "Generating {0} / {1} samples before starting training".format(
                            len(self.data), self.min_data_size
                        ),
                        end="\r",
                    )
                time.sleep(0.5)

            print(
                "Generating {0} / {1} samples before starting training".format(
                    len(self.data), self.min_data_size
                )
            )

            self.on_epoch_end()
        except (KeyboardInterrupt, Exception) as e:
            self.__exit__()
            raise e

        return self

    def __exit__(self, *args):
        self.exit_signal = True
        self.data_generation_thread.join()
        return False

    def on_epoch_end(self):
        # Grab a copy
        current_data = list(self.data)

        self.new_epoch = True
        if self.augmentation:
            for data_point in current_data:
                data_point["data"] = self.augmentation.update().resolve(
                    data_point["data"]
                )

        if self.shuffle_batch:
            random.shuffle(current_data)

        self.current_data = current_data

        self.epoch += 1

        if callable(self.batch_size):
            self._batch_size = self.batch_size(self.epoch, len(current_data))
        else:
            self._batch_size = self.batch_size

        while len(self.data) < self.min_data_size:
            print("Awaiting dataset to reach minimum size...", end="\r")
            time.sleep(0.1)

    def __getitem__(self, idx):

        batch_size = self._batch_size

        subset = self.current_data[idx * batch_size : (idx + 1) * batch_size]

        data = [self.batch_function(d["data"]) for d in subset]
        labels = [self.label_function(d["data"]) for d in subset]
        return np.array(data), np.array(labels)

    def __len__(self):
        steps = int((len(self.current_data) // self._batch_size))
        assert (
            steps > 0
        ), "There needs to be at least batch_size number of datapoints. Try increasing min_data_size."
        return steps

    def _continuous_get_training_data(self):
        index = 0
        while True:
            # Stop generator
            if self.exit_signal:
                break

            new_image = self._get(self.feature)

            datapoint = self.construct_datapoint(new_image)
            if len(self.data) >= self.max_data_size:
                self.data.pop(0)
            else:
                self.data.append(datapoint)

            self.cleanup()

            index += 1

            if index % self.max_data_size == 0:

                while not self.new_epoch and not self.exit_signal:
                    time.sleep(0.1)

            self.new_epoch = False

    def construct_datapoint(self, image):

        return {"data": image, "usage": 0}

    def cleanup(self):
        self.data = [
            sample
            for sample in self.data
            if sample["usage"] < self.max_epochs_per_sample
        ]

    def _get(self, features: Feature or List[Feature]) -> Image:
        # Updates and resolves a feature or list of features.
        if isinstance(features, list):

            for feature in features:
                feature.update()

            return [feature.resolve() for feature in features]
        else:
            features.update()
            return features.resolve()


# class CappedContinuousGenerator(ContinuousGenerator):

#     """Generator that asynchronously expands the dataset.

#     Generator that aims to speed up the training of networks by striking a
#     balance between the generalization gained by generating new images
#     and the speed gained from reusing images. The generator will continuously
#     create new training data during training, until `max_data_size` is reached,
#     at which point the oldest data point is replaced.

#     Unlike the `ContinuousGenerator`, this generator will purge any data that
#     has been seen more than `max_sample_exposure` times.

#     The generator is expected to be used with the python "with" statement, which
#     ensures that the generator worker is consumed correctly.

#     Parameters
#     ----------
#     feature : Feature
#         The feature to resolve images from.
#     label_function : Callable[Image or list of Image] -> array_like
#         Function that returns the label corresponding to a feature output.
#     batch_function : Callable[Image or list of Image] -> array_like, optional
#         Function that returns the training data corresponding a feature output.
#     min_data_size : int
#         Minimum size of the training data before training starts
#     max_data_set : int
#         Maximum size of the training data before old data is replaced.
#     max_sample_exposure : int
#         Any sample that has been seen for more than `max_sample_exposure` will be removed from the dataset
#     batch_size : int or Callable[int, int] -> int
#         Number of images per batch. A function is expected to accept the current epoch
#         and the size of the training data as input.
#     shuffle_batch : bool
#         If True, the batches are shuffled before outputting.
#     feature_kwargs : dict or list of dicts
#         Set of options to pass to the feature when resolving
#     ndim : int
#         Number of dimensions of each batch (including the batch dimension).
#     max_sample_exposure : int
#         Any sample that has been seen for more than `max_sample_exposure` will be removed from the dataset
#     """

#     def __init__(self, *args, max_sample_exposure=np.inf, **kwargs):
#         import warnings

#         warnings.warn(
#             "CappedContinuousGenerator is deprecated in favor of ContinuousGenerator with the max_epochs_per_sample argument.",
#             DeprecationWarning,
#         )
#         self.max_sample_exposure = max_sample_exposure
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, idx):

#         # TODO: Use parent method
#         batch_size = self._batch_size
#         subset = self.current_data[idx * batch_size : (idx + 1) * batch_size]
#         for a in subset:
#             a[-1] += 1
#         outputs = [array(a) for a in list(zip(*subset))]
#         outputs = (outputs[0], *outputs[1:-1])
#         return outputs

#     def construct_datapoint(self, image, label):
#         return [image, label, 0]

#     def cleanup(self):
#         self.data = [
#             sample for sample in self.data if sample[-1] < self.max_sample_exposure
#         ]

#     def on_epoch_end(self):

#         while len(self.data) < self.min_data_size:
#             print("Awaiting dataset to reach minimum size...", end="\r")
#             time.sleep(0.1)

#         return super().on_epoch_end()


class AutoTrackGenerator(ContinuousGenerator):
    def __init__(self, *args, symmetries=1, **kwargs):
        self.symmetries = symmetries
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):

        aug = None
        if aug is None:
            aug = Affine(
                translate=lambda: np.random.randn(2) * 2,
                # scale=lambda: np.random.choice([-1, 1], size=(2,)),
                rotate=lambda: np.random.rand() * np.pi * 2,
            )

        x = self.current_data[idx]["data"]
        x = np.array(x)

        # x = (
        #     (
        #         Affine(
        #             scale=lambda: np.random.choice([-1, 1], size=(2,)),
        #             rotate=lambda: np.random.rand() * np.pi * 2,
        #         )
        #         >> Gaussian(sigma=lambda: np.random.rand() * 0.01)
        #     )
        #     .update()
        #     .resolve(x)
        # )
        sample = np.array(x)
        batch = [aug.update().resolve(sample) for _ in range(self.batch_size)]

        labels = np.array(
            [self.get_transform_matrix(batch[0], b).reshape((-1,)) for b in batch]
        )

        return np.array(batch), np.array(labels)

    def __len__(self):
        return len(self.current_data)

    def get_transform_matrix(self, base_image, new_image):
        t0 = base_image.get_property("translate")[2::-1]
        r0 = base_image.get_property("rotate") * self.symmetries
        s0 = base_image.get_property("scale")

        t1 = new_image.get_property("translate")[2::-1]
        r1 = new_image.get_property("rotate") * self.symmetries
        s1 = new_image.get_property("scale")

        rmat0 = np.array([[np.cos(r0), np.sin(r0)], [-np.sin(r0), np.cos(r0)]]) * s0
        rmat1 = np.array([[np.cos(r1), np.sin(r1)], [-np.sin(r1), np.cos(r1)]]) * s1

        rmat = np.linalg.inv(rmat0) @ rmat1
        dt = (np.array(t1) - t0) @ rmat1

        return np.array(
            [rmat[0, 0], rmat[0, 1], dt[0], rmat[1, 0], rmat[1, 1], dt[1], 0, 0]
        )
