import cv2 as cv


def matchWithRatioTest(
    matcher: cv.DescriptorMatcher, desc1, desc2, nn_match_ratio_th=0.8
):
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    ratioMatched = []
    for m, n in nn_matches:
        if m.distance < n.distance * nn_match_ratio_th:
            ratioMatched.append(m)

    return ratioMatched
