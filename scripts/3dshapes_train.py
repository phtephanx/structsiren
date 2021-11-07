import argparse
import os
import random
import shutil
import typing

import numpy as np
import pandas as pd
import torch
from torch.utils import tensorboard
from tqdm import tqdm

from structsiren.models import (
    EfficientNetSirenSAE,
    StructSirenSAE
)
from structsiren.datasets import (
    load_3dshapes,
    Reconstruction
)


epochs_default = 60
batch_size_default = 64
learning_rate_default = float(1e-4)
num_workers_default = 0
device_default = 'cpu'
checkpoint_frequency_default = 1
random_seed_default = 12345


def safe_path(path):
    return os.path.abspath(os.path.expanduser(path))


def train_for_one_epoch(
        model: StructSirenSAE,
        train_loader: torch.utils.data.DataLoader,
        optimizer,
        criterion,
        device
) -> float:
    r"""Train model for one epoch."""
    model = model.to(device)
    model.train()

    total_loss = 0

    for (imgs, targets) in tqdm(
            train_loader,
            total=len(train_loader),
            desc='Training steps'
    ):
        optimizer.zero_grad()
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = model(imgs)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.data.cpu()

    return total_loss


@torch.no_grad()
def evaluate_on_loader(
        model: torch.nn.Module,
        loader: torch.data.utils.DataLoader,
        *,
        device: typing.Union[str, torch.device] = 'cpu'
):
    predictions = []
    targets = []

    model.to(device)
    model.eval()

    for imgs, target in tqdm(
            loader,
            desc='Evaluation',
            total=len(loader)
    ):
        imgs = imgs.to(device)
        prediction = model(imgs)
        predictions.append(prediction.cpu())
        targets.append(target.cpu())

    return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)


def evaluate(
        model: torch.nn.Module,
        output_folder: str,
        dev_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        metrics: typing.Dict[str, typing.Callable],
        criterion: str,
        ascending: bool,
        *,
        device: torch.device = 'cpu'
):
    checkpoints = sorted(
        [os.path.join(output_folder, folder)
         for folder in os.listdir(output_folder) if 'Epoch' in folder])

    for checkpoint in tqdm(
            checkpoints,
            desc='Checkpoint',
            total=len(checkpoints)
    ):
        if os.path.exists(os.path.join(checkpoint, f'outputs_dev.pt')):
            continue

        model.cpu()
        ckpt = torch.load(
            os.path.join(checkpoint, 'ckpt.pth.tar'),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(ckpt['model_state_dict'], strict=True)

        outputs, targets = evaluate_on_loader(model, dev_loader, device)
        torch.save(outputs, os.path.join(checkpoint, f'outputs_dev.pt'))
        torch.save(targets, os.path.join(checkpoint, f'targets_dev.pt'))

    results = {}

    # select best model on dev set
    for ckpt in tqdm(
            checkpoints,
            desc='Model Selection',
            total=len(checkpoints)
    ):
        results[ckpt.split('/')[-1]] = {}
        for set in ['dev']:
            outputs = torch.load(os.path.join(ckpt, f'outputs_{set}.pt'))
            targets = torch.load(os.path.join(ckpt, f'targets_{set}.pt'))
            results[ckpt.rsplit('/', 1)[1]][set] = {
                key: metrics[key](targets, outputs)
                for key in metrics.keys()
            }

    results_df = pd.DataFrame.from_dict(
        {(i, j): results[i][j]
         for i in results.keys()
         for j in results[i].keys()
         },
        orient='index')

    results_df.reset_index(inplace=True)
    results_df = results_df.rename(
        columns={'level_0': 'Epoch', 'level_1': 'Partition'})
    results_df['Epoch'] = results_df['Epoch'].apply(
        lambda x: int(x.strip('Epoch_')))
    results_df.set_index('Partition', inplace=True)
    print(results_df)

    if not isinstance(results_df.loc['dev'], pd.Series):
        best_epoch = int(results_df.loc['dev'].sort_values(
            by=criterion,
            ascending=ascending
        ).iloc[-1]['Epoch'])
        results_df = results_df.sort_values(by='Epoch')
    else:
        best_epoch = int(results_df.loc['dev']['Epoch'])

    best_folder = os.path.join(output_folder, 'best')
    if os.path.exists(best_folder):
        shutil.rmtree(best_folder)

    shutil.copytree(
        src=os.path.join(output_folder, f'Epoch_{best_epoch}'),
        dst=best_folder
    )

    print(f'\nBest epoch found at: {best_epoch}')
    print(f'Dev set results are:')
    print(results_df.loc[results_df['Epoch'] == best_epoch])
    results_df.to_csv(
        os.path.join(output_folder, 'dev_results.csv'),
        index=True
    )

    # Evaluate on test set
    ckpt = torch.load(
        os.path.join(best_folder, 'ckpt.pth.tar'),
        map_location=torch.device('cpu')
    )
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    outputs, targets = evaluate_on_loader(model, test_loader, device)
    test_results = {key: metrics[key](targets, outputs)
                    for key in metrics.keys()}
    test_results = pd.DataFrame(test_results, index=[f'epoch {best_epoch}'])
    print('\nTest results are:')
    print(test_results)
    test_results.to_csv(
        os.path.join(output_folder, 'test_results.csv'),
        index=False
    )

    torch.save(outputs, os.path.join(best_folder, 'outputs_test.pt'))
    torch.save(targets, os.path.join(best_folder, 'targets_test.pt'))
    return best_folder


def main(args: argparse.Namespace):

    cache_root = safe_path(args.cache_root)
    output_folder = safe_path(args.output_folder)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    seed = args.random_seed

    if seed is not None:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    args.h_dim = 64
    args.w_dim = 64
    args.num_channels = 3

    model = EfficientNetSirenSAE(
        h_dim=args.h_dim,
        w_dim=args.w_dim,
        num_channels=args.num_channels,
        num_factors=6,
        dim_per_factor=2,
        encoder_hidden_dims=[64],
        siren_hidden_dim=128,
        sigmoid=True,
        name='efficientnet-b0',
        pretrained=True
    )

    torch.save(model, os.path.join(output_folder, 'model.pth.tar'))
    model = model.to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    if args.checkpoint:
        ckpt_file = os.path.join(safe_path(args.checkpoint), 'ckpt.pth.tar')
        ckpt = torch.load(ckpt_file, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    train_series, dev_series, test_series = load_3dshapes(cache_root)

    train_loader = torch.utils.data.DataLoader(
        Reconstruction(train_series),
        collate_fn=None,
        shuffle=True,
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

    device = args.device
    epochs = args.epochs

    mse = torch.nn.MSELoss()
    metric = 'MSE'
    metrics = {'MSE': lambda x, y: mse(x, y).item()}
    ascending = False

    torch.save(model.cpu(), os.path.join(output_folder, 'model.pth.tar'))
    torch.save(model.cpu().state_dict(), os.path.join(
        output_folder, 'init.pth.tar'))

    writer = tensorboard.SummaryWriter(output_folder)

    for epoch in range(1, epochs + 1):
        epoch_folder = os.path.join(output_folder, f'Epoch_{epoch}')

        if epoch % args.checkpoint_frequency == 0:
            os.mkdir(epoch_folder)

        if os.path.exists(os.path.join(epoch_folder, 'ckpt.pth.tar')):
            continue

        total_loss = train_for_one_epoch(
            model, train_loader, optimizer, mse, device)
        avg_loss = total_loss / len(train_loader)
        print(f'Avg. loss in epoch {epoch}: {avg_loss}')
        writer.add_scalar("Loss/train", avg_loss, epoch)

        if epoch % args.checkpoint_frequency == 0:
            torch.save(
                {
                    'model_state_dict': model.cpu().state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                },
                os.path.join(epoch_folder, 'ckpt.pth.tar')
            )

    writer.flush()

    evaluate(
        model=model,
        output_folder=output_folder,
        dev_loader=dev_loader,
        test_loader=test_loader,
        metrics=metrics,
        criterion=metric,
        ascending=ascending,
        device=device
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training')

    parser.add_argument(
        'cache_root',
        type=str,
    )

    parser.add_argument(
        '-o', '--output_folder',
        type=str,
        help='path to output folder'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='path to checkpoint folder with state dict. '
             f'Default: `{None}`',
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=batch_size_default,
        help=f'batch size. Default: `{batch_size_default}`'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=epochs_default
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=learning_rate_default
    )

    parser.add_argument(
        '--device',
        type=str,
        default=device_default
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=num_workers_default
    )

    parser.add_argument(
        '--random_seed',
        type=int,
        default=random_seed_default
    )

    parser.add_argument(
        '-f', '--checkpoint_frequency',
        type=int,
        default=checkpoint_frequency_default,
        help='Frequency in epochs to save model and optimizer state. '
             f'Default: {checkpoint_frequency_default}'
    )

    parser.add_argument(
        '--plot-img',
        action='store_true'
    )

    main(parser.parse_args())
