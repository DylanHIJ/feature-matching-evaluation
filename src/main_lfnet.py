import os
import glob
import argparse
import numpy as np 

from utils.matcher import BruteForceMatcher
from utils.metrics import calc_proj_errs

THRESHOLDS = range(1, 11)

def load_seq_npz(seq_dir):
    ref_path = os.path.join(seq_dir, "1.ppm.npz")
    ref = np.load(ref_path)
    
    tars = []
    for i in range(5):
        tar_path = os.path.join(seq_dir, f"{i+2}.ppm.npz")
        tar = np.load(tar_path)
        tars.append(tar)
    
    return ref, tars

def load_seq_homos(seq_dir):
    homos = []
    for i in range(5):
        homo_path = os.path.join(seq_dir, f"H_1_{i+2}")
        with open(homo_path, "r") as homo_f:
            homo = homo_f.read()
            homo = np.array([float(x) for x in homo.strip().split()]).reshape(3, 3)
        homos.append(homo)
    return homos

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        help="directory of hpatches data",
    )
    parser.add_argument(
        "--result-dir",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results")),
        help="directory of results"
    )
    args = parser.parse_args()
    return args

def _main():
    args = _parse_arguments()
    
    seq_dirs = glob.glob(os.path.join(args.data_dir, "*"))
    seq_dirs = [os.path.abspath(path) for path in seq_dirs]
    print(f"Number of sequences: {len(seq_dirs)}")
    
    matcher = BruteForceMatcher()

    accs_all, accs_illu, accs_view = np.zeros(len(THRESHOLDS)), np.zeros(len(THRESHOLDS)), np.zeros(len(THRESHOLDS))
    features_nums, matches_nums = [], []
    num_v, num_i = 0, 0
    for seq_dir in seq_dirs:
        seq_name = os.path.basename(seq_dir)
        print(seq_name)
        
        if seq_name[0] == "v":
            num_v += 1
        elif seq_name[0] == "i":
            num_i += 1
        
        ref, tars = load_seq_npz(seq_dir)
        ref_kps, ref_descrs = ref['kpts'], ref['feats']
        features_nums.append(ref_descrs.shape[0])

        homos = load_seq_homos(seq_dir)
        
        for idx, tar in enumerate(tars):
            tar_kps, tar_descrs = tar['kpts'], tar['feats']
            features_nums.append(tar_descrs.shape[0])
            
            matches = matcher.mutual_match(tar_descrs, ref_descrs)
            matches = np.array([[match.queryIdx, match.trainIdx] for match in matches])
            matches_nums.append(len(matches))
            
            match_ref_kps = ref_kps[matches[:, 0], :2] 
            match_tar_kps = tar_kps[matches[:, 1], :2]
    
            errs = calc_proj_errs(match_ref_kps, match_tar_kps, homos[idx])
            if errs.shape[0] == 0:
                errs = np.array([float("inf")])

            for t_i, thresh in enumerate(THRESHOLDS):
                accs_all[t_i] += np.mean(errs <= thresh)
                if seq_name[0] == "v":
                    accs_view[t_i] += np.mean(errs <= thresh)
                elif seq_name[0] == "i":
                    accs_illu[t_i] += np.mean(errs <= thresh)
         
    mma_all = accs_all / ((num_i + num_v) * 5)
    mma_view = accs_view / (num_v * 5)
    mma_illu = accs_illu / (num_i * 5)

    mean_features_num = np.mean(np.array(features_nums))
    mean_matches_num = np.mean(np.array(matches_nums))

    print(f"--- Descriptor: LF-Net ---")
    print(f"MMA (all): {mma_all}")
    print(f"MMA (illumination): {mma_illu}")
    print(f"MMA (viewpoint): {mma_view}")
    print(f"Mean number of featues: {mean_features_num}")
    print(f"Mean number of matches: {mean_matches_num}")

    mma_dir = os.path.join(args.result_dir, "mma_lfnet")
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