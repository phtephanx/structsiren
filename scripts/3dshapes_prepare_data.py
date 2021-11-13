import argparse
import os
import requests
import sys
import typing

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from structsiren.datasets.shapes3d import URL, H5, SERIES


def _download(url: str, filename: str):
    r"""Download content at `url` to `filename`."""
    print(f'downloading from {url} to {filename}')

    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(
                    chunk_size=max(int(total / 1000), 1024 * 1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total)
                sys.stdout.write(
                    '\r[{}{}]'.format('█' * done, '.' * (50 - done)))
                sys.stdout.flush()
    sys.stdout.write('\n')


def _cache_3d_shapes(
        cache_root: str,
        *,
        h5: typing.Optional[str] = None,
        download: bool = False
):
    r"""Cache corpus of 3D-Shapes.

    Args:
        cache_root: folder to save images to
        h5: file with images
        download: option whether to download data if not available

    """
    cache_root = os.path.abspath(os.path.expanduser(cache_root))
    if not os.path.exists(cache_root):
        os.mkdir(cache_root)

    if not h5:
        h5 = os.path.abspath(os.path.expanduser(H5))

    if not os.path.exists(h5):
        if not download:
            raise RuntimeError(
                f'h5 file not available at `{h5}`, '
                f'yet, download disabled.')

        _download(URL, h5)

    with h5py.File(h5, "r") as f:
        images = np.array(f['images'])
        labels = np.array(f['labels'])

    filenames = []

    for i, img in tqdm(
            enumerate(images),
            desc='Cache 3D-Shapes',
            total=images.shape[0]
    ):
        fname = os.path.join(cache_root, f'{str(i).zfill(5)}_img.npy')
        filenames.append(fname)
        np.save(fname, arr=img)

    series = pd.Series(data=labels.tolist(), index=filenames)
    series.to_pickle(
        os.path.join(cache_root, SERIES),
        compression='xz'
    )


def main(args: argparse.Namespace):

    _cache_3d_shapes(
        cache_root=args.cache_root,
        h5=args.h5,
        download=args.download
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'cache_root',
        type=str,
        help='path to folder where 3dshape content '
             'should be stored'
    )

    parser.add_argument(
        '--h5',
        type=str,
        help='path to h5-file with 3dshape content. '
             f'If not specified, it is downloaded if '
             f'download option is activated',
        default=''
    )

    parser.add_argument(
        '--download',
        action='store_true',
        help='option whether to download `h5` file if not available'
    )

    main(parser.parse_args())
