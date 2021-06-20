import cv2

class FlannMatcher:
    def __init__(self, index_params):
        search_params = dict(checks=50)  
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def mutual_match(self, query_descrs, train_descrs, ratio_thresh=0.7):
        matches_1 = self.flann.knnMatch(train_descrs, query_descrs, k=2)
        matches_2 = self.flann.knnMatch(query_descrs, train_descrs, k=2) 

        mutual_matches = []
        for i, match_1 in enumerate(matches_1):
            j = match_1[0].trainIdx
            # if match_1[0].distance < ratio_thresh * match_1[1].distance and\
            #     matches_2[j][0].trainIdx == i and\
            #     matches_2[j][0].distance < ratio_thresh * matches_2[j][1].distance:
            #     mutual_matches.append(match_1[0])
            if matches_2[j][0].trainIdx == i:
                mutual_matches.append(match_1[0])
        return mutual_matches


class BruteForceMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    
    def mutual_match(self, query_descrs, train_descrs, ratio_thresh=0.7):
        matches = self.bf.match(train_descrs, query_descrs)
        return matches
        
        


