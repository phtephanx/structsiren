import argparse
import json
import os

import audeer
from disentanglement_lib.evaluation.metrics.dci import _compute_dci
from disentanglement_lib.evaluation.metrics.irs import (
    _drop_constant_dims,
    scalable_disentanglement_score
)
import numpy as np
import pandas as pd

from structsiren.datasets import shapes3d


def main(args: argparse.Namespace):

    ckpt_folder = audeer.safe_path(args.ckpt_folder)
    factors = shapes3d.FACTORS

    train_codes = pd.read_pickle(
        os.path.join(ckpt_folder, 'train_codes.pkl'),
        compression='xz'
    )
    dev_codes = pd.read_pickle(
        os.path.join(ckpt_folder, 'dev_codes.pkl'),
        compression='xz'
    )
    test_codes = pd.read_pickle(
        os.path.join(ckpt_folder, 'test_codes.pkl'),
        compression='xz'
    )

    frame = pd.concat((train_codes, dev_codes, test_codes), axis=0)

    # from continuous to categorical
    for f in factors:
        values = frame[f].unique()
        name_map = {i: j for i, j in zip(values, range(len(values)))}
        frame[f] = frame[f].apply(lambda row: name_map[row])

    # column with filenames is called `index`
    frame = frame.reset_index()
    train_frame = frame[frame['index'].isin(train_codes.index.tolist())]
    test_frame = frame[frame['index'].isin(test_codes.index.tolist())]

    test_codes = np.stack(test_frame['codes'], axis=1)  # (#codes, bs)
    test_factors = test_frame[factors].values.T  # (#factors, bs)
    train_codes = np.stack(train_frame['codes'], axis=1)  # (#codes, bs)
    train_factors = train_frame[factors].values.T  # (#factors, bs)

    scores_long = {}

    dci = _compute_dci(
        mus_train=train_codes,
        ys_train=train_factors,
        mus_test=test_codes,
        ys_test=test_factors
    )

    for k in dci:
        dci[k] = round(dci[k], 3)

    scores_long["DCI"] = dci

    # drop codes which are constant over all samples
    active_codes = _drop_constant_dims(test_codes)
    irs = scalable_disentanglement_score(
        gen_factors=test_factors.T,
        latents=active_codes.T,
        diff_quantile=0.99
    )
    for k in irs:
        v = irs[k]
        if isinstance(v, np.ndarray):
            irs[k] = np.round(v, 3).tolist()
        else:
            irs[k] = round(v, 3)

    scores_long["IRS"] = irs
    scores_long["num_active_dims"] = round(np.sum(active_codes), 3)

    scores = dci.copy()
    scores.update({'irs': irs['avg_score']})
    scores = pd.DataFrame(scores, index=range(1))

    print(f'Disentanglement Scores: \n{scores}')
    scores.to_csv(os.path.join(ckpt_folder, 'disentanglement.csv'))
    with open(os.path.join(ckpt_folder, 'disentanglement_long.txt'), 'w') as f:
        json.dump(obj=scores_long, fp=f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'ckpt_folder',
        type=str,
        help='path to checkpoint folder with code tables'
    )

    main(parser.parse_args())
