import argparse
import os

import audeer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from structsiren.models import (
    EfficientNetSirenSAE,
    SirenSAE
)
from structsiren.datasets import (
    Reconstruction,
    load_3dshapes
)
import structsiren.datasets.shapes3d


SHAPES3D_COLORS = [
    'darkblue', 'blue', 'darkviolet', 'magenta',
    'crimson', 'firebrick', 'lightsalmon', 'chocolate',
    'darkorange', 'goldenrod', 'gold', 'olive',
    'yellowgreen', 'lawngreen', 'forestgreen'
]

X = 'x'
Y = 'y'
DIM = 'dim'


def main(args: argparse.Namespace):

    ckpt_folder = audeer.safe_path(args.ckpt_folder)
    colors = SHAPES3D_COLORS
    factors = shapes3d.FACTORS

    # Load Embeddings
    frame = pd.read_pickle(
        os.path.join(ckpt_folder, 'test_codes.pkl'),
        compression='xz'
    )
    if 'labels' in frame.columns:
        frame = frame.drop('labels', axis=1)

    # Prepare Dataframe

    # Map to Categorical Values
    palette = {}

    for factor in factors:
        values = frame[factor].unique()
        name_map = {i: factor+str(j+1)
                    for i, j in zip(values, range(len(values)))}
        frame[factor] = frame[factor].apply(lambda row: name_map[row])
        palette.update(
            {n: colors[i]
             for i, n in enumerate(name_map.values())}
        )

    id_vars = []
    for i in range(args.num_codes):
        id_vars.extend([X + str(i+1), Y + str(i+1)])

    codes = pd.DataFrame.from_records(
        frame['codes'],
        columns=id_vars,
        index=frame.index
    )

    frame = pd.concat(
        (frame, codes),
        axis=1
    ).drop('codes', axis=1)

    print(frame)
    print(factors)
    print(DIM)

    frame = pd.wide_to_long(
        frame,
        stubnames=[X, Y],
        i=factors,
        j=DIM
    ).reset_index()

    frame = pd.melt(
        frame,
        id_vars=[X, Y, DIM],
        value_vars=factors,
        var_name='factor',
        value_name='value'
    )

    sns.relplot(
        data=frame,
        x=X,
        y=Y,
        hue='value',
        row='factor',
        col=DIM,
        palette=palette,
        facet_kws={'sharey': False, 'sharex': False}
    )

    plt.savefig(
        os.path.join(ckpt_folder, 'codes-to-factors.png')
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ckpt_folder',
        type=str,
        help='path to checkpoint folder with embeddings tables'
    )

    parser.add_argument(
        'num_codes',
        type=int,
        help='number of codes used to represent latent factors. '
             'The number of dimensions per code needs to be `2` '
             'in order to plot the latent space in 2D'
    )

    main(parser.parse_args())
