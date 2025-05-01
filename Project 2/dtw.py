import numpy as np

def dtw_distance(seq1, seq2):
    """
    Computes DTW distance between two feature sequences using Euclidean cost.
    """
    n, m = len(seq1), len(seq2)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = np.linalg.norm(seq1[i - 1] - seq2[j - 1])
            cost[i, j] = dist + min(
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1]
            )

    return cost[n, m]

def compute_dtw_from_images(img_path1, img_path2, extractor):
    """
    Loads  binarized word images, extracts feature sequences, and returns the DTW distance.

    Args:
        img_path1 (str): Path to first binarized word image
        img_path2 (str): Path to second binarized word image
        extractor (function): Feature extraction function

    Returns:
        float: DTW distance between the feature sequences
    """
    f1 = extractor(img_path1)
    f2 = extractor(img_path2)
    return dtw_distance(f1, f2)
