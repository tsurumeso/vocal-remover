import json
import sys

import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    with open(sys.argv[1], 'r', encoding='utf8') as f:
        log = np.asarray(json.load(f))
    print(np.min(log, axis=0))
    trn_loss = log[:, 0]
    val_loss = log[:, 1]

    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    x_val = np.arange(len(val_loss))
    plt.plot(x_val, val_loss, label='validation loss', c='r')

    x_trn = np.arange(len(trn_loss))
    plt.plot(x_trn, trn_loss, label='training loss', c='b')

    plt.grid(which='both', color='gray', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(edgecolor='white')
    plt.show()
