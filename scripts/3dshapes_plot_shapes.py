import argparse
import os

import audeer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths
import torch


num_workers_default = 0
device_default = 'cpu'


def plot_img(a: np.ndarray, dst: str, count: int = 0):
    r"""Plot image and save to `dst`."""
    a = np.clip(a, 0, 1).astype('float32')
    a = (a * 255).astype(np.uint8)
    im = Image.fromarray(a)
    im.save(os.path.join(dst, f'{count}.png'))


def main(args: argparse.Namespace):
    best_folder = os.path.abspath(os.path.expanduser(args.best_folder))

    predictions = torch.load(os.path.join(best_folder, 'outputs_test.pt'))
    predictions = predictions.cpu().numpy()
    targets = torch.load(os.path.join(best_folder, 'targets_test.pt'))
    targets = targets.cpu().numpy()

    img_folder = audeer.mkdir(os.path.join(best_folder, 'imgs'))
    img_p = audeer.mkdir(os.path.join(img_folder, 'predictions'))
    img_t = audeer.mkdir(os.path.join(img_folder, 'targets'))

    params = [([predictions[i], img_p, i], {})
              for i in range(predictions.shape[0])]
    audeer.run_tasks(
        task_func=plot_img,
        params=params,
        num_workers=args.num_workers,
        multiprocessing=True,
        progress_bar=True,
        task_description='Plot predictions'
    )

    params = [([targets[i], img_t, i], {}) for i in range(targets.shape[0])]
    audeer.run_tasks(
        task_func=plot_img,
        params=params,
        num_workers=args.num_workers,
        multiprocessing=True,
        progress_bar=True,
        task_description='Plot targets'
    )

    # https://github.com/mseitzer/pytorch-fid/blob/master/src/
    # pytorch_fid/fid_score.py#L61
    fid = calculate_fid_given_paths(
        [img_p, img_t],
        batch_size=2,
        device=args.device,
        dims=2048
    )

    print(f'Frechet Inception Distance: {fid}')
    with open(os.path.join(best_folder, 'fid.txt'), 'w') as f:
        f.write(f'fid: {fid}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'best_folder',
        type=str,
        help='path to folder with model, predictions and targets of '
             'best performing model, named `best` by default'
    )

    parser.add_argument(
        '--num_workers', '-w',
        type=int,
        default=num_workers_default,
        help=f'number of workers. Default: `{num_workers_default}`'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=device_default
    )

    main(parser.parse_args())
