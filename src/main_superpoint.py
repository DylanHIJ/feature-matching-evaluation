import os 
import glob
import argparse 
import numpy as np

from utils.matcher import BruteForceMatcher
from utils.metrics import calc_proj_errs

THRESHOLDS = range(1, 11)

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        help="directory of input image pairs (keypoints and descriptors)",
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
    
    pathnames = glob.glob(os.path.join(args.input_dir, "*"))
    print(f"Number of image pairs: {len(pathnames)}")

    matcher = BruteForceMatcher()

    matches_nums, ref_features_nums, tar_features_nums = [], [], []
    accs = np.zeros(len(THRESHOLDS))
    num = 0

    for pathname in pathnames:
        if os.path.splitext(pathname)[1] != ".npz":
            continue
        num += 1
        pred = np.load(pathname)
        ref_kps, ref_descrs = pred['prob'], pred['desc']
        tar_kps, tar_descrs = pred['warped_prob'], pred['warped_desc']
        homo = pred['homography']
        ref_features_nums.append(len(ref_descrs))
        tar_features_nums.append(len(tar_descrs))
        

        matches = matcher.mutual_match(tar_descrs, ref_descrs)
        matches = np.array([[match.queryIdx, match.trainIdx] for match in matches])
        matches_nums.append(len(matches))

        match_ref_kps = ref_kps[matches[:, 0], :2] 
        match_tar_kps = tar_kps[matches[:, 1], :2]
    
        errs = calc_proj_errs(match_ref_kps, match_tar_kps, homo)
        if errs.shape[0] == 0:
            errs = np.array([float("inf")])

        for t_i, thresh in enumerate(THRESHOLDS):
            accs[t_i] += np.mean(errs <= thresh)

    
    mma = accs / num
    mean_features_num = (np.mean(ref_features_nums) + np.mean(tar_features_nums) * 4) / 5   
    mean_matches_num = np.mean(matches_nums)
    
    mma_dir = os.path.join(args.result_dir, "mma_superpoint")
    if not os.path.exists(mma_dir):
        os.makedirs(mma_dir)

    print(f"--- Descriptor: SuperPoint ---") 
    if args.input_dir.split("/")[-2].split("_")[-1] == "all":
        print(f"MMA (all): {mma}")
        with open(os.path.join(mma_dir, "all.npy"), "wb") as f:
            np.save(f, mma)
    elif args.input_dir.split("/")[-2].split("_")[-1] == "v":
        print(f"MMA (viewpoint): {mma}")
        with open(os.path.join(mma_dir, "view.npy"), "wb") as f:
            np.save(f, mma)
    else:
        print(f"MMA (illumination): {mma}")
        with open(os.path.join(mma_dir, "illu.npy"), "wb") as f:
            np.save(f, mma)
    print(f"Mean number of featues: {mean_features_num}")
    print(f"Mean number of matches: {mean_matches_num}")
    

        
if __name__ == "__main__":
    _main()
