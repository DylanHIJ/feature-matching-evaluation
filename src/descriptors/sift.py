import cv2

class SIFT:
    def __init__(self):
        self.sift = cv2.SIFT_create()
    
    def detect_and_compute(self, img):
        keypoints, descriptors = self.sift.detectAndCompute(img, None)
        return keypoints, descriptors