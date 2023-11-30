import argparse
from datetime import datetime
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from lib import dataset
from lib import nets
from lib import spec_utils


def setup_logger(name, logfile='LOGFILENAME.log'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fh = logging.FileHandler(logfile, encoding='utf8')
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def to_wave(spec, n_fft, hop_length, window):
    B, _, N, T = spec.shape
    wave = spec.reshape(-1, N, T)
    wave = torch.istft(wave, n_fft, hop_length, window=window)
    wave = wave.reshape(B, 2, -1)

    return wave


def sdr_loss(y, y_pred, eps=1e-8):
    sdr = (y * y_pred).sum()
    sdr /= torch.linalg.norm(y) * torch.linalg.norm(y_pred) + eps

    return -sdr


def weighted_sdr_loss(y, y_pred, n, n_pred, eps=1e-8):
    y_sdr = (y * y_pred).sum()
    y_sdr /= torch.linalg.norm(y) * torch.linalg.norm(y_pred) + eps

    noise_sdr = (n * n_pred).sum()
    noise_sdr /= torch.linalg.norm(n) * torch.linalg.norm(n_pred) + eps

    a = torch.sum(y ** 2)
    a /= torch.sum(y ** 2) + torch.sum(n ** 2) + eps

    loss = a * y_sdr + (1 - a) * noise_sdr

    return -loss


def train_epoch(dataloader, model, device, optimizer, accumulation_steps):
    model.train()
    # n_fft = model.n_fft
    # hop_length = model.hop_length
    # window = torch.hann_window(n_fft).to(device)

    sum_loss = 0
    crit_l1 = nn.L1Loss()

    for itr, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        mask = model(X_batch)

        # y_pred = X_batch * mask
        # y_wave_batch = to_wave(y_batch, n_fft, hop_length, window)
        # y_wave_pred = to_wave(y_pred, n_fft, hop_length, window)

        # loss = crit_l1(torch.abs(y_batch), torch.abs(y_pred))
        # loss += sdr_loss(y_wave_batch, y_wave_pred) * 0.01
        loss = crit_l1(mask * X_batch, y_batch)

        accum_loss = loss / accumulation_steps
        accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        sum_loss += loss.item() * len(X_batch)

    # the rest batch
    if (itr + 1) % accumulation_steps != 0:
        optimizer.step()
        model.zero_grad()

    return sum_loss / len(dataloader.dataset)


def validate_epoch(dataloader, model, device):
    model.eval()
    # n_fft = model.n_fft
    # hop_length = model.hop_length
    # window = torch.hann_window(n_fft).to(device)

    sum_loss = 0
    crit_l1 = nn.L1Loss()

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model.predict(X_batch)

            y_batch = spec_utils.crop_center(y_batch, y_pred)
            # y_wave_batch = to_wave(y_batch, n_fft, hop_length, window)
            # y_wave_pred = to_wave(y_pred, n_fft, hop_length, window)

            # loss = crit_l1(torch.abs(y_batch), torch.abs(y_pred))
            # loss += sdr_loss(y_wave_batch, y_wave_pred) * 0.01
            loss = crit_l1(y_pred, y_batch)

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
    p.add_argument('--accumulation_steps', '-A', type=int, default=1)
    p.add_argument('--cropsize', '-C', type=int, default=256)
    p.add_argument('--patches', '-p', type=int, default=16)
    p.add_argument('--val_rate', '-v', type=float, default=0.2)
    p.add_argument('--val_filelist', '-V', type=str, default=None)
    p.add_argument('--val_batchsize', '-b', type=int, default=4)
    p.add_argument('--val_cropsize', '-c', type=int, default=256)
    p.add_argument('--num_workers', '-w', type=int, default=4)
    p.add_argument('--epoch', '-E', type=int, default=200)
    p.add_argument('--reduction_rate', '-R', type=float, default=0.0)
    p.add_argument('--reduction_level', '-L', type=float, default=0.2)
    p.add_argument('--mixup_rate', '-M', type=float, default=0.0)
    p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
    p.add_argument('--pretrained_model', '-P', type=str, default=None)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    logger.debug(vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    val_filelist = []
    if args.val_filelist is not None:
        with open(args.val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    train_filelist, val_filelist = dataset.train_val_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode,
        val_rate=args.val_rate,
        val_filelist=val_filelist
    )

    if args.debug:
        logger.info('### DEBUG MODE')
        train_filelist = train_filelist[:1]
        val_filelist = val_filelist[:1]
    elif args.val_filelist is None and args.split_mode == 'random':
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        logger.info('{} {} {}'.format(i + 1, os.path.basename(X_fname), os.path.basename(y_fname)))

    bins = args.n_fft // 2 + 1
    freq_to_bin = 2 * bins / args.sr
    unstable_bins = int(200 * freq_to_bin)
    stable_bins = int(22050 * freq_to_bin)
    reduction_weight = np.concatenate([
        np.linspace(0, 1, unstable_bins, dtype=np.float32)[:, None],
        np.linspace(1, 0, stable_bins - unstable_bins, dtype=np.float32)[:, None],
        np.zeros((bins - stable_bins, 1), dtype=np.float32),
    ], axis=0) * args.reduction_level

    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128)
    if args.pretrained_model is not None:
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        threshold=1e-6,
        min_lr=args.lr_min,
        verbose=True
    )

    training_set = dataset.make_training_set(
        filelist=train_filelist,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft
    )

    train_dataset = dataset.VocalRemoverTrainingSet(
        training_set * args.patches,
        cropsize=args.cropsize,
        reduction_rate=args.reduction_rate,
        reduction_weight=reduction_weight,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers
    )

    patch_list = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        offset=model.offset
    )

    val_dataset = dataset.VocalRemoverValidationSet(
        patch_list=patch_list
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.val_batchsize,
        shuffle=False,
        num_workers=args.num_workers
    )

    log = []
    best_loss = np.inf
    for epoch in range(args.epoch):
        logger.info('# epoch {}'.format(epoch))
        train_loss = train_epoch(train_dataloader, model, device, optimizer, args.accumulation_steps)
        val_loss = validate_epoch(val_dataloader, model, device)

        logger.info(
            '  * training loss = {:.6f}, validation loss = {:.6f}'
            .format(train_loss, val_loss)
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            logger.info('  * best validation loss')
            model_path = 'models/model_iter{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)

        log.append([train_loss, val_loss])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)
