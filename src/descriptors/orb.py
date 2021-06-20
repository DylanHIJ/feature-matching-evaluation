import cv2

class ORB:
    def __init__(self):
        self.orb = cv2.ORB_create()
    
    def detect_and_compute(self, img):
        keypoints, descriptors = self.orb.detectAndCompute(img, None)
        return keypoints, descriptors