import numpy as np
from skimage.io import imread

def extract_features(image_path):
    """
    Extracts sliding window features from a binarized word image.
    Each 1-pixel wide column yields:
    - Lower contour (bottom-most black pixel)
    - Upper contour (top-most black pixel)
    - Fraction of black pixels
    - Number of black/white transitions
    - Gradient of LC
    - Gradient of UC
    """
    image = imread(image_path, as_gray=True)
    image = (image < 0.5).astype(np.uint8)

    h, w = image.shape
    features = []

    for x in range(w):
        column = image[:, x]
        black_pixels = np.where(column == 1)[0]

        lc = black_pixels[-1] if black_pixels.size > 0 else 0
        uc = black_pixels[0] if black_pixels.size > 0 else 0
        frac_black = black_pixels.size / h
        transitions = np.count_nonzero(np.diff(column))
        grad_lc = 0
        grad_uc = 0

        features.append([lc, uc, frac_black, transitions, grad_lc, grad_uc])

    features = np.array(features, dtype=np.float32)

    if features.shape[0] > 1:
        features[1:, 4] = features[1:, 0] - features[:-1, 0]  # grad_lc
        features[1:, 5] = features[1:, 1] - features[:-1, 1]  # grad_uc

    mean = features.mean(axis=0)
    std = features.std(axis=0) + 1e-6
    features = (features - mean) / std

    return features
