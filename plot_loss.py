import numpy as np
import argparse
import matplotlib.pyplot as plt
import os

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="usage: %(prog)s [opts]")
    parser.add_argument('--version', action='version', version='%prog 1.0')
    parser.add_argument('-g', '--global_name',  action='store', type=str, dest='global_name',  default='Test', help='Global name for identifying this run - used in folder naming and output naming')
    opts = parser.parse_args()

    train_loss = f"models/{opts.global_name}/loss_train_{opts.global_name}.npy"
    val_loss = f"models/{opts.global_name}/loss_val_{opts.global_name}.npy"

    train_loss = np.load(train_loss)
    val_loss = np.load(val_loss)
    
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.ylabel("loss")
    plt.legend(frameon=False, title="")
    #plt.show()
    os.makedirs(f"plots/{opts.global_name}/",exist_ok=True)
    plt.savefig(f"plots/{opts.global_name}/train_val_loss.png")
