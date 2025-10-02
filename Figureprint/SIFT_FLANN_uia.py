import os
import cv2
import numpy as np

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def sift_flann_match(img1_gray, img2_gray, ratio=0.75, checks=64):
    sift = cv2.SIFT_create(nfeatures=2000)

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return [], kp1, kp2, None, None, 0

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=checks)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn = flann.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    H, inliers = None, None
    inlier_count = 0
    if len(good) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)
        if inliers is not None:
            inlier_count = int(inliers.sum())

    return good, kp1, kp2, H, inliers, inlier_count

def draw_matches(img1_gray, img2_gray, kp1, kp2, matches, inliers_mask=None, max_draw=200):
    if inliers_mask is not None and len(inliers_mask) == len(matches):
        # Keep only inlier matches for drawing (looks cleaner)
        inlier_matches = [m for m, keep in zip(matches, inliers_mask.ravel().tolist()) if keep]
        draw_these = inlier_matches[:max_draw] if len(inlier_matches) > 0 else matches[:max_draw]
    else:
        draw_these = matches[:max_draw]

    vis = cv2.drawMatches(
        img1_gray, kp1,
        img2_gray, kp2,
        draw_these, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis

if __name__ == "__main__":
    image = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\data-check_uia\UiA front1.png"
    image2 = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\data-check_uia\UiA front3.jpg"

    out_dir = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\results_folder_sift_uia"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "UiA_SIFT_FLANN_matches.png")

    img1 = load_gray(image)
    img2 = load_gray(image2)

    good, kp1, kp2, H, inliers_mask, inlier_count = sift_flann_match(img1, img2, ratio=0.75, checks=64)

    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")
    print(f"Good matches after ratio test: {len(good)}")
    if H is not None:
        print(f"Inlier matches (RANSAC): {inlier_count}")
    else:
        print("Homography not estimated (not enough matches).")

    vis = draw_matches(img1, img2, kp1, kp2, good, inliers_mask)
    cv2.imwrite(out_path, vis)
    print(f"Saved to here:: {out_path}")

    cv2.imshow("SIFT + FLANN matches", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()