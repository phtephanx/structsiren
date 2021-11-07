import argparse
import os
import typing

import pandas as pd
import torch

import audeer
from structsiren.models import (
    EfficientNetSirenSAE,
    SirenSAE
)
from structsiren.datasets import (
    Reconstruction,
    load_3dshapes
)


@torch.no_grad()
def retrieve_codes(
        model: SirenSAE,
        loader: torch.utils.data.DataLoader,
        *,
        device: typing.Union[str, torch.device] = 'cpu'):
    all_codes = []
    model.to(device)
    model.eval()

    for index, (imgs, _) in audeer.progress_bar(
            enumerate(loader),
            desc='Retrieve codes',
            total=len(loader)
    ):
        imgs = imgs.to(device)
        codes = model(imgs, return_codes=True)[1]
        codes = codes.cpu()
        all_codes.append(codes)

    return torch.cat(all_codes, dim=0)


def main(args: argparse.Namespace):

    cache_root = audeer.safe_path(args.cache_root)
    model_file = audeer.safe_path(args.model_file)
    train_series, dev_series, test_series = load_3dshapes(cache_root)

    train_loader = torch.utils.data.DataLoader(
        Reconstruction(train_series),
        collate_fn=None,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    dev_loader = torch.utils.data.DataLoader(
        Reconstruction(dev_series),
        collate_fn=None,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    test_loader = torch.utils.data.DataLoader(
        Reconstruction(test_series),
        collate_fn=None,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    model = torch.load(model_file)

    ckpt = torch.load(
        os.path.join(args.ckpt, 'ckpt.pth.tar'),
        map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model_state_dict'])

    # train codes
    train_codes = retrieve_codes(
        model, train_loader, device=args.device)
    train_frame = train_series.to_frame('labels')
    train_frame[['floor', 'wall', 'object', 'scale', 'shape', 'orient']] = \
        pd.DataFrame(train_frame.labels.tolist(), index=train_frame.index)
    train_frame['codes'] = train_codes.tolist()
    train_frame.to_pickle(
        os.path.join(args.ckpt, 'train_codes.pkl'), compression='xz')
    torch.save(
        train_codes, os.path.join(args.ckpt, f'train_codes.pt'))

    # dev codes
    dev_codes = retrieve_codes(
        model, dev_loader, device=args.device)
    dev_frame = dev_series.to_frame('labels')
    dev_frame[['floor', 'wall', 'object', 'scale', 'shape', 'orient']] = \
        pd.DataFrame(dev_frame.labels.tolist(), index=dev_frame.index)
    dev_frame['codes'] = dev_codes.tolist()
    dev_frame.to_pickle(
        os.path.join(args.ckpt, 'dev_codes.pkl'), compression='xz')
    torch.save(
        dev_frame, os.path.join(args.ckpt, f'dev_codes.pt'))

    # test codes
    test_codes = retrieve_codes(
        model, test_loader, device=args.device)
    test_frame = test_series.to_frame('labels')
    test_frame[['floor', 'wall', 'object', 'scale', 'shape', 'orient']] = \
        pd.DataFrame(test_frame.labels.tolist(), index=test_frame.index)
    test_frame['codes'] = test_codes.tolist()
    test_frame.to_pickle(
        os.path.join(args.ckpt, 'test_codes.pkl'), compression='xz')
    torch.save(
        test_codes, os.path.join(args.ckpt, f'test_codes.pt'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model_file',
        type=str,
        help='path to model.pth.tar to retrieve model definition'
    )

    parser.add_argument(
        'ckpt',
        type=str,
        help='path to folder with `state.pth.tar`'
    )

    parser.add_argument(
        'cache_root',
        type=str,
        help='path to root with data'
    )

    parser.add_argument(
        'batch_size',
        type=int,
        default=16
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=0
    )

    main(parser.parse_args())
