from abc import abstractmethod
import h5py
import os
import requests
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import audeer

from .define import (
    _FACTORS_IN_ORDER,
    _NUM_VALUES_PER_FACTOR,
    SERIES
)
from .utils import get_index


class Shapes3d(torch.utils.data.Dataset):
    r"""3D Shapes Dataset.

    Contains 480,000 images of size (64, 64) with 3 channels as
    uint in the range [0, 255].
    The targets consist of 6 labels, one for each different latent factor..

    source: https://github.com/deepmind/3d-shapes

    Args:
        series: index contains path to image files,
            labels represent configuration of six factors, i.e.
            floor hue, wall hue, object hue, scale, shape and orientation
        normalize: rescale pixel values to [0, 1]

    The expected data table (`series`) looks e.g. like:

    ```
    index
    3d-shapes/00000_img.npy                   [0.0, 0.0, 0.0, 0.75, 0.0, -30.0]
    3d-shapes/00001_img.npy     [0.0, 0.0, 0.0, 0.75, 0.0, -25.714285714285715]
                                                       ...
    Name: 0, Length: 480000, dtype: object
    ``

    floor hue: 10 values linearly spaced in [0, 0.9]
    wall hue: 10 values linearly spaced in [0, 0.9]
    object hue: 10 values linearly spaced in [0, 0.9]
    scale: 8 values linearly spaced in [0.75, 1.25]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]

    Shapes:
        images: (480000, 64, 64, 3)
        labels: (480000, 6)

    """

    def __init__(self, series: pd.Series, *, normalize: bool = True):
        super().__init__()
        self.files = [os.path.abspath(os.path.expanduser(f))
                      for f in series.index]
        self.labels = series.tolist()
        self.normalize = normalize

    def __len__(self):
        return len(self.files)

    def intervene(self, factor: int, value: int):
        r"""Intervene on `factor` and fix to `value`.

        Args:
            factor: index of `_FACTORS_IN_ORDER`
            value: manifestation of factor in range of
                `_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[factor]`

        Returns:
            image with fixed factor.

        """
        factors = list(np.zeros((len(_FACTORS_IN_ORDER),), dtype=np.int32))

        for index, name in enumerate(_FACTORS_IN_ORDER):
            num_choices = _NUM_VALUES_PER_FACTOR[name]
            factors[index] = np.random.choice(num_choices)

        factors[factor] = value
        return self[get_index(factors)]

    def load_image(self, index: int):
        img = np.load(self.files[index])
        return img.transpose((2, 0, 1))  # to (c, h, w)

    def show(self, index: int):
        r"""Show image at `index`."""
        img = self.load_image(index)

        if self.normalize:
            img = img / 255

        plt.imshow(img)
        plt.show()

    @abstractmethod
    def __getitem__(self, item):
        pass


class Reconstruction(Shapes3d):
    def __init__(self, series: pd.Series, *, normalize: bool = True):
        super().__init__(series=series, normalize=normalize)

    def __getitem__(self, item):
        img = self.load_image(item)

        if self.normalize:
            img = img / 255

        img = torch.from_numpy(img).float()
        return img, img.clone().detach()


class Disentanglement(Shapes3d):
    def __init__(self, series: pd.Series, *, normalize: bool = True):
        super().__init__(series=series, normalize=normalize)

    def __getitem__(self, item):
        img, label = self.load_image(item), self.labels[item]

        if self.normalize:
            img = img / 255

        img = torch.from_numpy(img).float()
        return img, label


def load_3dshapes(cache_root: str):
    r"""Load train, dev, test split for 3d-shapes.

    As there is no official train-dev-test split, the following splits
    adheres to https://arxiv.org/abs/2006.07796 which creates a 70-10-20
    shuffled split with a fixed random seed.

    """
    series = pd.read_pickle(
        os.path.join(audeer.safe_path(cache_root), SERIES),
        compression='xz'
    )

    series = series.sample(frac=1, replace=False, random_state=12345)

    # train on standard 70-10-20 train-dev-test split
    train_series = series[:int(0.7 * len(series))]
    dev_series = series[int(0.7 * len(series)):int(0.8 * len(series))]
    test_series = series[int(0.8 * len(series)):]

    return train_series, dev_series, test_series
