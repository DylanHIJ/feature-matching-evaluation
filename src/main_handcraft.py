import os
import glob
import argparse
import cv2
import numpy as np
import logging 

from utils.hpatches import *
from utils.matcher import BruteForceMatcher
from utils.metrics import calc_proj_errs
from descriptors.sift import SIFT
from descriptors.orb import ORB

THRESHOLDS = range(1, 11)

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    q    help="directory of data"
    )
    parser.add_argument(
        "--result-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")),
        help="directory of results"
    )
    parser.add_argument(
        "--descriptor",
        help="type of feature descriptor"
    )
    args = parser.parse_args()
    return args

def _main():
    args = _parse_arguments()
    
    seq_dirs = glob.glob(os.path.join(args.data_dir, "*"))
    seq_dirs = [os.path.abspath(path) for path in seq_dirs]
    print(f"Number of sequences: {len(seq_dirs)}")
    

    if args.descriptor == "sift":
        descriptor = SIFT()
    elif args.descriptor == "orb":
        descriptor = ORB()
        
    matcher = BruteForceMatcher()

    accs_all, accs_illu, accs_view = np.zeros(len(THRESHOLDS)), np.zeros(len(THRESHOLDS)), np.zeros(len(THRESHOLDS))
    features_nums, matches_nums = [], []
    num_v, num_i = 0, 0
    for seq_dir in seq_dirs:
        seq = HPatchesSequence(seq_dir)
        print(seq.name)

        if seq.name[0] == "v":
            num_v += 1
        elif seq.name[0] == "i":
            num_i += 1

        ref_kps, ref_descrs = descriptor.detect_and_compute(seq.ref_img)
        ref_descrs = np.float32(ref_descrs)
        features_nums.append(len(ref_descrs))
        

        for idx, tar_img in enumerate(seq.tar_imgs):
            correct = np.zeros(len(THRESHOLDS))
            tar_kps, tar_descrs = descriptor.detect_and_compute(tar_img)
            tar_descrs = np.float32(tar_descrs)
            features_nums.append(len(tar_descrs))

            matches = matcher.mutual_match(tar_descrs, ref_descrs)
            matches = np.array([[match.queryIdx, match.trainIdx] for match in matches])
            matches_nums.append(len(matches))
            
        
            match_ref_kps = np.array(ref_kps)[matches[:, 0]]
            match_ref_kps = np.array([list(kp.pt) for kp in match_ref_kps])
            match_tar_kps = np.array(tar_kps)[matches[:, 1]]
            match_tar_kps = np.array([list(kp.pt) for kp in match_tar_kps])
    
            errs = calc_proj_errs(match_ref_kps, match_tar_kps, seq.homos[idx])
            if errs.shape[0] == 0:
                errs = np.array([float("inf")])

            for t_i, thresh in enumerate(THRESHOLDS):
                accs_all[t_i] += np.mean(errs <= thresh)
                if seq.name[0] == "v":
                    accs_view[t_i] += np.mean(errs <= thresh)
                elif seq.name[0] == "i":
                    accs_illu[t_i] += np.mean(errs <= thresh)
            
    
    mma_all = accs_all / ((num_i + num_v) * 5)
    mma_view = accs_view / (num_v * 5)
    mma_illu = accs_illu / (num_i * 5)

    mean_features_num = np.mean(np.array(features_nums))
    mean_matches_num = np.mean(np.array(matches_nums))
    
    print(f"--- Descriptor: {args.descriptor} ---")
    print(f"MMA (all): {mma_all}")
    print(f"MMA (illumination): {mma_illu}")
    print(f"MMA (viewpoint): {mma_view}")
    print(f"Mean number of featues: {mean_features_num}")
    print(f"Mean number of matches: {mean_matches_num}")
    
    mma_dir = os.path.join(args.result_dir, f"mma_{args.descriptor}")
    if not os.path.exists(mma_dir):
        os.makedirs(mma_dir)
    with open(os.path.join(mma_dir, "all.npy"), "wb") as f:
        np.save(f, mma_all)
    with open(os.path.join(mma_dir, "illu.npy"), "wb") as f:
        np.save(f, mma_illu)
    with open(os.path.join(mma_dir, "view.npy"), "wb") as f:
        np.save(f, mma_view)

if __name__ == '__main__':
    _main()
