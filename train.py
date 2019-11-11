import argparse
from datetime import datetime as dt
import gc
import json
import os
import random

import chainer
import chainer.functions as F
import numpy as np

from lib import dataset
from lib import spec_utils
from lib import unet


p = argparse.ArgumentParser()
p.add_argument('--gpu', '-g', type=int, default=-1)
p.add_argument('--seed', '-s', type=int, default=2019)
p.add_argument('--sr', '-r', type=int, default=44100)
p.add_argument('--hop_length', '-l', type=int, default=1024)
p.add_argument('--mixture_dataset', '-m', required=True)
p.add_argument('--instrumental_dataset', '-i', required=True)
p.add_argument('--validation_rate', '-v', type=float, default=0.1)
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--lr_min', type=float, default=0.0001)
p.add_argument('--lr_decay', type=float, default=0.9)
p.add_argument('--lr_decay_interval', type=int, default=6)
p.add_argument('--batchsize', '-B', type=int, default=8)
p.add_argument('--val_batchsize', '-b', type=int, default=8)
p.add_argument('--val_filelist', '-V', type=str, default=None)
p.add_argument('--cropsize', '-c', type=int, default=448)
p.add_argument('--val_cropsize', '-C', type=int, default=896)
p.add_argument('--patches', '-p', type=int, default=16)
p.add_argument('--epoch', '-E', type=int, default=100)
p.add_argument('--inner_epoch', '-e', type=int, default=4)
p.add_argument('--oracle_rate', '-O', type=float, default=0)
p.add_argument('--oracle_drop_rate', '-o', type=float, default=0.5)
p.add_argument('--mixup', '-M', action='store_true')
p.add_argument('--mixup_alpha', '-a', type=float, default=1.0)
p.add_argument('--pretrained_model', '-P', type=str, default=None)
args = p.parse_args()


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(args.seed)
        chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024)
    chainer.global_config.autotune = True
    timestamp = dt.now().strftime('%Y%m%d%H%M%S')

    model = unet.MultiBandUNet()
    if args.pretrained_model is not None:
        chainer.serializers.load_npz(args.pretrained_model, model)
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = model.xp

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)

    input_exts = ['.wav', '.m4a', '.3gp', '.oma', '.mp3', '.mp4']
    X_list = sorted(
        [os.path.join(args.mixture_dataset, fname)
         for fname in os.listdir(args.mixture_dataset)
         if os.path.splitext(fname)[1] in input_exts])
    y_list = sorted(
        [os.path.join(args.instrumental_dataset, fname)
         for fname in os.listdir(args.instrumental_dataset)
         if os.path.splitext(fname)[1] in input_exts])
    filelist = list(zip(X_list, y_list))

    val_filelist = []
    if args.val_filelist is not None:
        with open(args.val_filelist, 'r', encoding='utf8') as f:
            val_filelist = json.load(f)

    random.shuffle(filelist)
    if len(val_filelist) == 0:
        val_size = int(len(filelist) * args.validation_rate)
        train_filelist = filelist[:-val_size]
        val_filelist = filelist[-val_size:]
        with open('val_{}.json'.format(timestamp), 'w', encoding='utf8') as f:
            json.dump(val_filelist, f, ensure_ascii=False)
    else:
        train_filelist = [
            pair for pair in filelist
            if list(pair) not in val_filelist]

    for i, (X_fname, y_fname) in enumerate(val_filelist):
        print(i + 1, os.path.basename(X_fname), os.path.basename(y_fname))

    X_valid, y_valid = dataset.create_validation_set(
        filelist=val_filelist,
        cropsize=args.val_cropsize,
        sr=args.sr,
        hop_length=args.hop_length,
        offset=model.offset)

    log = []
    oracle_X = None
    oracle_y = None
    best_count = 0
    best_loss = np.inf
    for epoch in range(args.epoch):
        X_train, y_train = dataset.create_training_set(
            filelist=train_filelist,
            cropsize=args.cropsize,
            patches=args.patches,
            sr=args.sr,
            hop_length=args.hop_length)
        if args.mixup:
            X_train, y_train = dataset.mixup_generator(
                X_train, y_train, args.mixup_alpha)
        if oracle_X is not None and oracle_y is not None:
            perm = np.random.permutation(len(oracle_X))
            X_train[perm] = oracle_X
            y_train[perm] = oracle_y
        print('# epoch', epoch)
        instance_loss = np.zeros(len(X_train), dtype=np.float32)
        for inner_epoch in range(args.inner_epoch):
            sum_loss = 0
            best_count += 1
            perm = np.random.permutation(len(X_train))
            print('  * inner epoch {}'.format(inner_epoch))
            for i in range(0, len(X_train), args.batchsize):
                local_perm = perm[i: i + args.batchsize]
                X_batch = xp.asarray(X_train[local_perm])
                y_batch = xp.asarray(y_train[local_perm])

                model.cleargrads()
                mask = model(X_batch)

                X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
                y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

                abs_diff = F.absolute_error(X_batch * mask, y_batch)
                loss = F.mean(abs_diff)
                loss.backward()
                optimizer.update()

                il = abs_diff.data.mean(axis=(1, 2, 3))
                instance_loss[local_perm] += chainer.backends.cuda.to_cpu(il)
                sum_loss += float(loss.data) * len(X_batch)

            train_loss = sum_loss / len(X_train)

            sum_loss = 0
            perm = np.random.permutation(len(X_valid))
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                for i in range(0, len(X_valid), args.val_batchsize):
                    local_perm = perm[i: i + args.val_batchsize]
                    X_batch = xp.asarray(X_valid[local_perm])
                    y_batch = xp.asarray(y_valid[local_perm])

                    mask = model(X_batch)
                    X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
                    y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

                    loss = F.mean_absolute_error(X_batch * mask, y_batch)
                    sum_loss += float(loss.data) * len(X_batch)

            valid_loss = sum_loss / len(X_valid)
            print('    * training loss = {:.6f}, validation loss = {:.6f}'
                  .format(train_loss * 1000, valid_loss * 1000))

            log.append([train_loss, valid_loss])
            np.save('log_{}.npy'.format(timestamp), np.asarray(log))

            if valid_loss < best_loss:
                best_count = 0
                best_loss = valid_loss
                print('    * best validation loss')
                model_path = 'models/model_iter{}.npz'.format(epoch)
                chainer.serializers.save_npz(model_path, model)
            if epoch > 1 and best_count >= args.lr_decay_interval:
                best_count = 0
                optimizer.alpha *= args.lr_decay
                if optimizer.alpha < args.lr_min:
                    optimizer.alpha = args.lr_min
                else:
                    print('    * learning rate decay: {:.6f}'
                          .format(optimizer.alpha))

        if args.oracle_rate > 0:
            instance_loss /= args.inner_epoch
            oracle_X, oracle_y, idx = dataset.get_oracle_data(
                X_train, y_train, instance_loss, args.oracle_rate, args.oracle_drop_rate)
            print('  * oracle loss = {:.6f}'.format(instance_loss[idx].mean()))

        del X_train, y_train
        gc.collect()
