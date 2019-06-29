import argparse
import gc
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
p.add_argument('--mixture_dataset', '-m', required=True)
p.add_argument('--instrumental_dataset', '-i', required=True)
p.add_argument('--learning_rate', type=float, default=0.001)
p.add_argument('--lr_min', type=float, default=0.00001)
p.add_argument('--lr_decay', type=float, default=0.9)
p.add_argument('--lr_decay_interval', type=int, default=5)
p.add_argument('--batchsize', '-B', type=int, default=32)
p.add_argument('--val_batchsize', '-b', type=int, default=32)
p.add_argument('--cropsize', '-c', type=int, default=512)
p.add_argument('--epoch', '-E', type=int, default=50)
p.add_argument('--inner_epoch', '-e', type=int, default=4)
p.add_argument('--mixup', '-M', action='store_true')
p.add_argument('--mixup_alpha', '-a', type=float, default=0.4)
args = p.parse_args()


if __name__ == '__main__':
    random.seed(args.seed)
    np.random.seed(args.seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(args.seed)
    chainer.global_config.autotune = True

    model = unet.SpecUNet()
    if args.gpu >= 0:
        chainer.backends.cuda.check_cuda_available()
        chainer.backends.cuda.get_device(args.gpu).use()
        model.to_gpu()

    optimizer = chainer.optimizers.Adam(args.learning_rate)
    optimizer.setup(model)
    # optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(0.0001))

    input_exts = ['.wav', '.m4a', '.3gp', '.oma', '.mp3', '.mp4']
    X_list = sorted(
        [fname for fname in os.listdir(args.mixture_dataset)
         if os.path.splitext(fname)[1] in input_exts])
    y_list = sorted(
        [fname for fname in os.listdir(args.instrumental_dataset)
         if os.path.splitext(fname)[1] in input_exts])

    filelist = []
    for X_fname, y_fname in zip(X_list, y_list):
        filelist.append((os.path.join(args.mixture_dataset, X_fname),
                         os.path.join(args.instrumental_dataset, y_fname)))

    random.shuffle(filelist)
    train_filelist = filelist[:-20]
    valid_filelist = filelist[-20:]
    X_valid, y_valid = dataset.create_dataset(
        valid_filelist, args.cropsize, validation=True)

    best_count = 0
    best_loss = np.inf
    for epoch in range(args.epoch):
        random.shuffle(train_filelist)
        X_train, y_train = dataset.create_dataset(
            train_filelist[:100], args.cropsize)
        if args.mixup:
            X_train, y_train = dataset.mixup_generator(
                X_train, y_train, args.mixup_alpha)
        print('# epoch', epoch)
        for inner_epoch in range(args.inner_epoch):
            sum_loss = 0
            best_count += 1
            perm = np.random.permutation(len(X_train))
            for i in range(0, len(X_train), args.batchsize):
                local_perm = perm[i: i + args.batchsize]
                X_batch = model.xp.asarray(X_train[local_perm])
                y_batch = model.xp.asarray(y_train[local_perm])

                model.cleargrads()
                mask = model(X_batch)
                X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
                y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

                loss = F.mean_absolute_error(X_batch * mask, y_batch)
                loss.backward()
                optimizer.update()

                sum_loss += float(loss.data) * len(X_batch)

            train_loss = sum_loss / len(X_train)

            sum_loss = 0
            perm = np.random.permutation(len(X_valid))
            with chainer.no_backprop_mode(), chainer.using_config('train', False):
                for i in range(0, len(X_valid), args.val_batchsize):
                    local_perm = perm[i: i + args.val_batchsize]
                    X_batch = model.xp.asarray(X_valid[local_perm])
                    y_batch = model.xp.asarray(y_valid[local_perm])

                    mask = model(X_batch)
                    X_batch = spec_utils.crop_and_concat(mask, X_batch, False)
                    y_batch = spec_utils.crop_and_concat(mask, y_batch, False)

                    inst_loss = F.mean_squared_error(X_batch * mask, y_batch)
                    vocal_loss = F.mean_squared_error(
                        X_batch * (1 - mask), X_batch - y_batch)
                    loss = inst_loss + vocal_loss
                    sum_loss += float(loss.data) * len(X_batch)

            valid_loss = sum_loss / len(X_valid)
            print('  * inner epoch {}'.format(inner_epoch))
            print('    * training loss = {:.6f}, validation loss = {:.6f}'
                  .format(train_loss * 100, valid_loss * 1000))

            if valid_loss < best_loss:
                best_count = 0
                best_loss = valid_loss
                print('    * best validation loss')
                model_path = 'models/model_iter{}.npz'.format(epoch)
                chainer.serializers.save_npz(model_path, model)
            if best_count >= args.lr_decay_interval:
                best_count = 0
                optimizer.alpha *= args.lr_decay
                if optimizer.alpha < args.lr_min:
                    optimizer.alpha = args.lr_min
                else:
                    print('    * learning rate decay: {:.6f}'
                          .format(optimizer.alpha))

        del X_train, y_train
        gc.collect()
