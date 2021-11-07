from typing import (List, Union)

import numpy as np
import matplotlib.pyplot as plt
import torch

from .define import (_FACTORS_IN_ORDER, _NUM_VALUES_PER_FACTOR)


def get_index(factors: Union[np.ndarray, List[int]]):
    r"""Get index in dataset for factor intervention.

    Args:
        factors: array of shape [batch_size, 6].
            intervention[:, i] takes integer values in range from
            `_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]`

      Returns:
        indices: np array shape [batch_size]

    """

    indices = 0
    base = 1

    if isinstance(factors, np.ndarray):
        factors = factors.transpose(1, 0)

    for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
        indices += factors[factor] * base
        base *= _NUM_VALUES_PER_FACTOR[name]
    return indices


def all_indices_of_factor(factor: int, value: int):
    values_per_factor = _NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[factor]]

    if value not in range(values_per_factor):
        raise ValueError(
            f'`value` not in range({values_per_factor} of '
            f'factor `{_FACTORS_IN_ORDER[factor]}` ')

    ranges = tuple(_NUM_VALUES_PER_FACTOR.values())
    indices = np.arange(np.prod(ranges)).reshape(ranges)
    return indices.take(value, axis=factor).reshape(-1)


def show_images(imgs: Union[np.ndarray, torch.Tensor]):
    """Show images of 3D shapes.

    Args:
        imgs: images of 3D shapes

    Shapes:
        * imgs: (N, C, H, W)

    """
    num_images = imgs.shape[0]

    ncols = int(np.ceil(num_images**0.5))
    nrows = int(np.ceil(num_images / ncols))
    _, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
    axes = axes.flatten()

    for ax_i, ax in enumerate(axes):
        if ax_i < num_images:
            ax.imshow(imgs[ax_i], cmap='Greys_r', interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.show()
