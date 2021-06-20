import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

mapping = {
    "sift": "SIFT",
    "orb": "ORB",
    "lfnet": "LF-Net",
    "superpoint": "SuperPoint",
    "d2net": "D2-Net",
    "r2d2": "R2D2",
    "aslfeat": "ASLFeat",
}
def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dir',
        help='directory of inputs',
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    )
    parser.add_argument(
        '--descriptors',
        help='types of descriptors',
        nargs='*',
        default=['sift', 'orb', 'lfnet', 'superpoint', 'd2net', 'r2d2', 'aslfeat'],
    )
    args = parser.parse_args()
    return args

def _main():
    args = _parse_arguments()
    
    thresholds = range(1, 11)
    ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for descr in args.descriptors:
        mma_all = np.load(os.path.join(args.input_dir, f'mma_{descr}', 'all.npy'))
        plt.plot(thresholds, mma_all, label=mapping[descr])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('threshold (px)')
    plt.ylabel('MMA (%)')
    plt.xticks(thresholds)
    plt.yticks(ticks)
    plt.title('MMA_all')
    plt.savefig(os.path.join(args.input_dir, 'mma_all.png'))
    plt.clf()

    for descr in args.descriptors:
        mma_illu = np.load(os.path.join(args.input_dir, f'mma_{descr}', 'illu.npy'))
        plt.plot(thresholds, mma_illu, label=mapping[descr])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('threshold (px)')
    plt.ylabel('MMA (%)')
    plt.xticks(thresholds)
    plt.yticks(ticks)
    plt.title('MMA_illumination')
    plt.savefig(os.path.join(args.input_dir, 'mma_illu.png'))
    plt.clf()

    for descr in args.descriptors:
        mma_view = np.load(os.path.join(args.input_dir, f'mma_{descr}', 'view.npy'))
        plt.plot(thresholds, mma_view, label=mapping[descr])
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel('threshold (px)')
    plt.ylabel('MMA (%)')
    plt.xticks(thresholds)
    plt.yticks(ticks)
    plt.title('MMA_viewpoint')
    plt.savefig(os.path.join(args.input_dir, 'mma_view.png'))
    plt.clf()

if __name__ == '__main__':
    _main()
