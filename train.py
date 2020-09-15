import argparse
from datetime import datetime
import gc
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import dataset
from lib import nets
from lib import spec_utils


def train_inner_epoch(X, y, model, device, optimizer, batchsize):
    model.train()
    sum_loss = 0
    crit = nn.L1Loss()
    perm = np.random.permutation(len(X))

    for i in range(0, len(X), batchsize):
        local_perm = perm[i: i + batchsize]

        X_batch = X[local_perm]
        y_batch = y[local_perm]

        X_mag = np.abs(X_batch)
        y_mag = np.abs(y_batch)

        X_mag = torch.from_numpy(X_mag).to(device)
        y_mag = torch.from_numpy(y_mag).to(device)

        model.zero_grad()
        pred, aux1, aux2 = model(X_mag)

        loss = crit(pred, y_mag) * 0.8
        loss += crit(aux1, y_mag) * 0.1
        loss += crit(aux2, y_mag) * 0.1

        loss.backward()
        optimizer.step()

        sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(X)


def val_inner_epoch(dataloader, model, device):
    model.eval()
    sum_loss = 0
    crit = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model.predict(X_batch)

            y_batch = spec_utils.crop_center(y_batch, pred)
            loss = crit(pred, y_batch)

            sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(dataloader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--seed', '-s', type=int, default=2019)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--dataset', '-d', required=True)
    p.add_argument('--split_mode', '-S', type=str, choices=['random', 'subdirs'], default='random')
    p.add_argument('--learning_rate', '-l', type=float, default=0.001)
    p.add_argument('--lr_min', type=float, default=0.0001)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=2)
    p.add_argument('--val_cropsize', '-C', type=int, default=512)
    p.add_argument('--epoch', '-E', type=int, default=60)
    p.add_argument('--inner_epoch', '-e', type=int, default=4)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    val_filelist = []
    if args.val_filelist is not None:
        with open(args.val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    train_filelist, val_filelist = dataset.train_val_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode,
        val_rate=args.val_rate,
        val_filelist=val_filelist)

    if args.debug:
        print('### DEBUG MODE')
        train_filelist = train_filelist[:1]
        val_filelist = val_filelist[:1]
    elif args.val_filelist is None:
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        print(i + 1, os.path.basename(X_fname), os.path.basename(y_fname))

    device = torch.device('cpu')
    model = nets.CascadedASPPNet(args.n_fft)
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True)

    val_dataset = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        offset=model.offset)

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=4)

    bins = args.n_fft // 2 + 1
    freq_to_bin = 2 * bins / args.sr
    unstable_bins = int(160 * freq_to_bin)
    reduction_bins = int(16000 * freq_to_bin)
    reduction_mask = np.concatenate([
        np.linspace(0, 1, unstable_bins)[:, None],
        np.linspace(1, 0, reduction_bins - unstable_bins)[:, None],
        np.zeros((bins - reduction_bins, 1))
    ], axis=0) * args.reduction_level

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        X_train, y_train = dataset.make_training_set(
            filelist=train_filelist,
            cropsize=args.cropsize,
            patches=args.patches,
            sr=args.sr,
            hop_length=args.hop_length,
            n_fft=args.n_fft,
            offset=model.offset)

        X_train, y_train = dataset.augment(
            X_train, y_train,
            reduction_rate=args.reduction_rate,
            reduction_mask=reduction_mask,
            mixup_rate=args.mixup_rate,
            mixup_alpha=args.mixup_alpha)

        print('# epoch', epoch)
        for inner_epoch in range(args.inner_epoch):
            print('  * inner epoch {}'.format(inner_epoch))

            train_loss = train_inner_epoch(
                X_train, y_train,
                model=model,
                device=device,
                optimizer=optimizer,
                batchsize=args.batchsize)

            val_loss = val_inner_epoch(val_dataloader, model, device)

            print('    * training loss = {:.6f}, validation loss = {:.6f}'
                  .format(train_loss, val_loss))

            scheduler.step(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                print('    * best validation loss')
                model_path = 'models/model_iter{}.pth'.format(epoch)
                torch.save(model.state_dict(), model_path)

            log.append([train_loss, val_loss])
            with open('log_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
                json.dump(log, f, ensure_ascii=False)

        del X_train, y_train
        gc.collect()


if __name__ == '__main__':
    main()
