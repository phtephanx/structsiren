import typing

import torch


def get_meshgrid(
        side_lengths: typing.Union[int, typing.List, typing.Tuple]
) -> torch.Tensor:
    r"""Generates a flattened grid of (x, y, ...) coordinates between -1 and 1.

    Args:
        side_lengths: side lengths of dimensions

    Returns:
        mesh grid

    """
    if isinstance(side_lengths, int):
        side_lengths = [side_lengths]

    num_dims = len(side_lengths)
    tensors = [torch.linspace(-1, 1, steps=lengths)
               for lengths in side_lengths]
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    return mgrid.reshape(-1, num_dims)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i],
            x,
            torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(
        y,
        [x],
        grad_outputs=grad_outputs, create_graph=True)[0]
    return grad
