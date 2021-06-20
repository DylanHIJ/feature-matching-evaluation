import numpy as np


def calc_proj_errs(ref_pts, tar_pts, homo):
    ref_pts_extend = np.concatenate((ref_pts, np.ones((ref_pts.shape[0], 1))), axis=1)
    proj_pts_extend = np.transpose(homo @ np.transpose(ref_pts_extend))
    proj_pts = proj_pts_extend[:, :2] / proj_pts_extend[:, 2:]
    errs = np.sqrt(np.sum((proj_pts - tar_pts) ** 2, axis=1))
    return errs 