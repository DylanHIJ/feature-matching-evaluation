import os
import cv2
import re
import numpy as np


class HPatchesSequence: 
    def __init__(self, seq_dir):
        self.name = os.path.basename(seq_dir)
        self.seq_dir = seq_dir
        self.tar_imgs = []
        self.homos = []

        for i in range(6): 
            img_path = os.path.join(seq_dir, f"{i+1}.ppm")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if i == 0:
                self.ref_img = img
            else:
                self.tar_imgs.append(img)
        
        for i in range(5):
            homo_path = os.path.join(seq_dir, f"H_1_{i+2}")
            with open(homo_path, "r") as homo_f:
                homo = homo_f.read()
                homo = np.array([float(x) for x in homo.strip().split()]).reshape(3, 3)
            self.homos.append(homo)


    def get_homo(self, idx):
        return getattr(self, f"homo_{idx}")

def _main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    seq_dir = os.path.join(data_dir, 'v_coffeehouse')
    seq = HPatchesSequence(seq_dir)
    cv2.imwrite("v_coffeehouse_ref.png", seq.ref_img)
    cv2.imwrite("v_coffeehouse_tar_2.png", seq.tar_imgs[0])
    
if __name__ == '__main__':
    _main()