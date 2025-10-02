import os
import cv2
import numpy as np

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Couldnt save the image: {path}")
    return img

def orb_bf_match(img1_gray, img2_gray, ratio=0.7):
    orb = cv2.ORB_create(nfeatures=2000)

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)
    if des1 is None or des2 is None:
        return [], kp1, kp2, None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    H, mask = None, None
    inliers = 0
    if len(good) >= 4:
        src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
        if mask is not None:
            inliers = int(mask.sum())

    return good, kp1, kp2, H, mask, inliers

def draw_matches(img1_gray, img2_gray, kp1, kp2, matches, mask=None, max_draw=200):
    if mask is not None and len(mask) == len(matches):
        inlier_matches = [m for m, keep in zip(matches, mask.ravel().tolist()) if keep]
        to_draw = inlier_matches[:max_draw] if inlier_matches else matches[:max_draw]
    else:
        to_draw = matches[:max_draw]

    vis = cv2.drawMatches(
        img1_gray, kp1, img2_gray, kp2, to_draw, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return vis

if __name__ == "__main__":
    image = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\data-check_uia\UiA front1.png"
    image2 = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\data-check_uia\UiA front3.jpg"

    out_dir = r"C:\Users\space\OneDrive\Skrivebord\Figureprint\results_folder_orb_uia"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "UiA_ORB_BF_matches.png")

    img1 = load_gray(image)
    img2 = load_gray(image2)

    good, kp1, kp2, H, mask, inliers = orb_bf_match(img1, img2, ratio=0.7)

    print(f"Keypoints: img1={len(kp1)}, img2={len(kp2)}")
    print(f"Good matches (ratio test): {len(good)}")
    print(f"Inlier matches (RANSAC): {inliers}")

    decision = "MATCHED" if inliers >= 15 else "NOT MATCHED"
    print(f"Decision (inliers>=15): {decision}")

    vis = draw_matches(img1, img2, kp1, kp2, good, mask)
    cv2.imwrite(out_path, vis)
    print(f"Saved here: {out_path}")

    # Show window (optional)
    cv2.imshow(f"ORB+BF: {decision}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
