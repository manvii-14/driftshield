import numpy as np
from numpy.linalg import norm
from scipy.stats import energy_distance
from sklearn.metrics.pairwise import rbf_kernel


def subsample(X, n=500, seed=42):
    """Random subset of rows, so the tests run fast."""
    if len(X) <= n:
        return X
    rng = np.random.default_rng(seed)
    return X[rng.choice(len(X), n, replace=False)]


def cosine_centroid(X, Y):
    """Baseline: cosine similarity of the two mean vectors. 1 = no drift."""
    cx, cy = X.mean(axis=0), Y.mean(axis=0)
    return float(np.dot(cx, cy) / (norm(cx) * norm(cy)))


def _mmd_from_kernel(K, a, b):
    return K[np.ix_(a, a)].mean() + K[np.ix_(b, b)].mean() - 2 * K[np.ix_(a, b)].mean()


def mmd(X, Y, gamma=None):
    """Maximum Mean Discrepancy with an RBF kernel. 0 = same distribution."""
    Z = np.vstack([X, Y])
    K = rbf_kernel(Z, Z, gamma=gamma)  # gamma=None means 1/n_features
    n = len(X)
    return float(_mmd_from_kernel(K, np.arange(n), np.arange(n, len(Z))))


def mmd_pvalue(X, Y, n_perm=200, gamma=None, seed=42):
    """Permutation test: returns (mmd_score, p_value). p < 0.05 means drift."""
    Z = np.vstack([X, Y])
    n = len(X)
    K = rbf_kernel(Z, Z, gamma=gamma)
    observed = _mmd_from_kernel(K, np.arange(n), np.arange(n, len(Z)))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(Z))
        if _mmd_from_kernel(K, p[:n], p[n:]) >= observed:
            count += 1
    return float(observed), (count + 1) / (n_perm + 1)


def energy(X, Y):
    """Average 1-D energy distance across all embedding dimensions."""
    scores = [energy_distance(X[:, i], Y[:, i]) for i in range(X.shape[1])]
    return float(np.mean(scores))


def drift_report(X, Y, n=500, seed=42, n_perm=200):
    """All three tests on a subsample of each side. Returns a dict."""
    Xs, Ys = subsample(X, n, seed), subsample(Y, n, seed)
    score, p = mmd_pvalue(Xs, Ys, n_perm=n_perm, seed=seed)
    return {
        "cosine_centroid": cosine_centroid(X, Y),
        "mmd": score,
        "mmd_p": p,
        "drift_detected": p < 0.05,
        "energy": energy(Xs, Ys),
    }