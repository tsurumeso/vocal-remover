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


def train_inner_epoch(X, y, model, optimizer, batchsize, max_reduction_rate):
    sum_loss = 0
    crit = nn.L1Loss()
    aux_crit = nn.L1Loss()

    unstable_region_offset = 4
    start = X.shape[2] // 4
    weight = np.concatenate([
        np.zeros((unstable_region_offset, 1)),
        np.ones((start - unstable_region_offset, 1)),
        np.linspace(1, 0, start)[:, None],
        np.zeros((X.shape[2] - 2 * start, 1))
    ], axis=0)

    perm = np.random.permutation(len(X))
    for i in range(0, len(X), batchsize):
        local_perm = perm[i: i + batchsize]

        X_batch = X[local_perm]
        y_batch = y[local_perm]

        X_mag = np.abs(X_batch)
        y_mag = np.abs(y_batch)
        v_mag = np.abs(X_batch - y_batch)

        reduction_rate = np.random.uniform(0, max_reduction_rate, len(X_batch))
        v_mag *= (v_mag > y_mag) * reduction_rate[:, None, None, None] * weight

        X_mag = torch.from_numpy(X_mag).cuda()
        y_mag_org = torch.from_numpy(y_mag).cuda()
        y_mag_sub = torch.from_numpy(np.clip(y_mag - v_mag, 0, np.inf)).cuda()

        model.zero_grad()
        pred, aux = model(X_mag)

        loss = crit(pred, y_mag_sub) * 0.9
        loss += aux_crit(aux, y_mag_org) * 0.1

        loss.backward()
        optimizer.step()

        sum_loss += loss.item() * len(X_batch)

    return sum_loss / len(X)


def val_inner_epoch(dataloader, model):
    sum_loss = 0
    crit = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.cuda()
            y_batch = y_batch.cuda()

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
    p.add_argument('--hop_length', '-l', type=int, default=1024)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--mixtures', '-m', required=True)
    p.add_argument('--instruments', '-i', required=True)
    p.add_argument('--learning_rate', '-L', type=float, default=0.001)
    p.add_argument('--lr_min', type=float, default=0.0001)
    p.add_argument('--lr_decay_factor', type=float, default=0.9)
    p.add_argument('--lr_decay_patience', type=int, default=6)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=256)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.1)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=4)
    p.add_argument('--val_cropsize', '-C', type=int, default=512)
    p.add_argument('--epoch', '-E', type=int, default=50)
    p.add_argument('--inner_epoch', '-e', type=int, default=4)
    p.add_argument('--max_reduction_rate', '-R', type=float, default=0.18)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', '-d', action='store_true')
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
        mix_dir=args.mixtures,
        inst_dir=args.instruments,
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

    mean, std = dataset.get_statistics(
        filelist=train_filelist,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft)

    model = nets.CascadedASPPNet(args.n_fft, mean, std)
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model))
    if args.gpu >= 0:
        model.cuda()

    model.mean.requires_grad = False
    model.std.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
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

        X_train, y_train = dataset.mixup_generator(
            X_train, y_train,
            rate=args.mixup_rate,
            alpha=args.mixup_alpha)

        print('# epoch', epoch)
        for inner_epoch in range(args.inner_epoch):
            print('  * inner epoch {}'.format(inner_epoch))

            model.train()
            train_loss = train_inner_epoch(
                X_train, y_train,
                model=model,
                optimizer=optimizer,
                batchsize=args.batchsize,
                max_reduction_rate=args.max_reduction_rate)

            model.eval()
            val_loss = val_inner_epoch(val_dataloader, model)

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
