import json
import sys

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='utf8') as f:
        log = np.asarray(json.load(f))
    print(np.min(log, axis=0))
    log = log[:(log.shape[0] // 4) * 4]
    split_trn = np.array_split(log[:, 0], log.shape[0] // 4)
    split_val = np.array_split(log[:, 1], log.shape[0] // 4)

    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    mean_val = np.mean(split_val, axis=1)
    min_val = np.min(split_val, axis=1)
    std_val = np.std(split_val, axis=1)
    x_val = np.arange(len(mean_val))
    plt.fill_between(
        x_val, mean_val - std_val, mean_val + std_val, alpha=0.5, color='r')
    plt.plot(x_val, mean_val, label='validation mean', c='r')
    plt.plot(x_val, min_val, label='validation min', c='k', ls='--')

    mean_trn = np.mean(split_trn, axis=1)
    std_trn = np.std(split_trn, axis=1)
    x_trn = np.arange(len(mean_trn))
    plt.fill_between(
        x_trn, mean_trn - std_trn, mean_trn + std_trn, alpha=0.5, color='b')
    plt.plot(x_trn, mean_trn, label='training mean', c='b')

    plt.grid(which='both', color='gray', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(edgecolor='white')
    plt.show()
