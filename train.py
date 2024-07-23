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
    B, C, N, T = spec.shape
    wave = spec.reshape(-1, N, T)
    wave = torch.istft(wave, n_fft, hop_length, window=window)
    wave = wave.reshape(B, C, -1)

    return wave


def train_epoch(dataloader, model, device, optimizer, accumulation_steps):
    is_complex = model.is_complex
    if is_complex:
        n_fft = model.n_fft
        hop_length = model.hop_length
        window = torch.hann_window(n_fft).to(device)

    model.train()
    crit_l1 = nn.L1Loss(reduction='none')
    sum_loss_y = sum_loss_v = 0

    for itr, (X_batch, y_batch) in enumerate(dataloader):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        mask = model(X_batch)
        y_pred = torch.cat([X_batch, X_batch], dim=1) * mask

        if is_complex:
            y_wave_batch = to_wave(y_batch, n_fft, hop_length, window)
            y_wave_pred = to_wave(y_pred, n_fft, hop_length, window)

            loss = torch.mean(crit_l1(torch.abs(y_batch), torch.abs(y_pred)), dim=(2, 3))
            loss += torch.mean(crit_l1(y_wave_batch, y_wave_pred), dim=2)
        else:
            loss = crit_l1(y_pred, y_batch)

        accum_loss = torch.mean(loss) / accumulation_steps
        accum_loss.backward()

        if (itr + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        sum_loss_y += torch.mean(loss[:, :2]).item() * len(X_batch)
        sum_loss_v += torch.mean(loss[:, 2:]).item() * len(X_batch)

    avg_loss_y = sum_loss_y / len(dataloader.dataset)
    avg_loss_v = sum_loss_v / len(dataloader.dataset)

    return avg_loss_y, avg_loss_v


def validate_epoch(dataloader, model, device):
    is_complex = model.is_complex
    if is_complex:
        n_fft = model.n_fft
        hop_length = model.hop_length
        window = torch.hann_window(n_fft).to(device)

    model.eval()
    sum_loss_y = sum_loss_v = 0
    crit_l1 = nn.L1Loss(reduction='none')

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model.predict(X_batch)
            y_batch = spec_utils.crop_center(y_batch, y_pred)

            if is_complex:
                y_wave_batch = to_wave(y_batch, n_fft, hop_length, window)
                y_wave_pred = to_wave(y_pred, n_fft, hop_length, window)

                loss = torch.mean(crit_l1(torch.abs(y_batch), torch.abs(y_pred)), dim=(2, 3))
                loss += torch.mean(crit_l1(y_wave_batch, y_wave_pred), dim=2)
            else:
                loss = crit_l1(y_pred, y_batch)

            sum_loss_y += torch.mean(loss[:, :2]).item() * len(X_batch)
            sum_loss_v += torch.mean(loss[:, 2:]).item() * len(X_batch)

    avg_loss_y = sum_loss_y / len(dataloader.dataset)
    avg_loss_v = sum_loss_v / len(dataloader.dataset)

    return avg_loss_y, avg_loss_v


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
    p.add_argument('--complex', '-X', action='store_true')
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

    trn_filelist, val_filelist = dataset.train_val_split(
        dataset_dir=args.dataset,
        split_mode=args.split_mode,
        val_rate=args.val_rate,
        val_filelist=val_filelist
    )

    if args.debug:
        logger.info('### DEBUG MODE')
        trn_filelist = trn_filelist[:1]
        val_filelist = val_filelist[:1]
    elif args.val_filelist is None and args.split_mode == 'random':
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)

    for i, (X_fname, y_fname, _) in enumerate(val_filelist):
        logger.info('{} {} {}'.format(i + 1, os.path.basename(X_fname), os.path.basename(y_fname)))

    reduction_weight = spec_utils.get_reduction_weight(args.n_fft, args.sr, args.reduction_level)

    device = torch.device('cpu')
    model = nets.CascadedNet(args.n_fft, args.hop_length, 32, 128, args.complex)
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
    )

    trn_set = dataset.make_training_set(
        filelist=trn_filelist,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft
    )

    trn_dataset = dataset.VocalRemoverTrainingSet(
        training_set=trn_set * args.patches,
        cropsize=args.cropsize,
        reduction_rate=args.reduction_rate,
        reduction_weight=reduction_weight,
        mixup_rate=args.mixup_rate,
        mixup_alpha=args.mixup_alpha,
        is_complex=args.complex
    )

    trn_dataloader = torch.utils.data.DataLoader(
        dataset=trn_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_set = dataset.make_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        offset=model.offset
    )

    val_dataset = dataset.VocalRemoverValidationSet(
        validation_set=val_set,
        is_complex=args.complex
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
        trn_loss_y, trn_loss_v = train_epoch(trn_dataloader, model, device, optimizer, args.accumulation_steps)
        val_loss_y, val_loss_v = validate_epoch(val_dataloader, model, device)

        logger.info(
            '  * training loss (y, v) = ({:.6f}, {:.6f}), validation loss (y, v) = ({:.6f}, {:.6f})'
            .format(trn_loss_y, trn_loss_v, val_loss_y, val_loss_v)
        )

        trn_loss = trn_loss_y + trn_loss_v
        val_loss = val_loss_y + val_loss_v
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            logger.info('  * best validation loss')
            model_path = 'models/model_iter{}.pth'.format(epoch)
            torch.save(model.state_dict(), model_path)

        log.append([trn_loss, val_loss])
        with open('loss_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(log, f, ensure_ascii=False)


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    logger = setup_logger(__name__, 'train_{}.log'.format(timestamp))

    try:
        main()
    except Exception as e:
        logger.exception(e)
