from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

def load_moons(n_samples: int = 1000, noise: float = 0.08, scale: bool = True):
    """
    Returns (X, y_true) for a 2D non-convex dataset (two interleaved moons).
    Great for HDBSCAN/OPTICS/Spectral demos.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=42)
    if scale:
        X = StandardScaler().fit_transform(X)
    return X, y

if __name__ == "__main__":
    X, y = load_moons()
    print("Moons:", X.shape, y.shape)
