from pathlib import Path
import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_olivetti_pca(n_components: int = 100, whiten: bool = True, scale: bool = True):
    """
    Returns (X_embedded, y) for Olivetti faces.
    X_embedded shape: (400, n_components)
    """
    faces = fetch_olivetti_faces()
    X, y = faces.data, faces.target
    if scale:
        X = StandardScaler().fit_transform(X)
    X = PCA(n_components=n_components, whiten=whiten, random_state=42).fit_transform(X)
    return X, y

def load_olivetti_raw():
    """Returns (X_raw, y) flattened 64x64 images -> (400, 4096)."""
    faces = fetch_olivetti_faces()
    return faces.data, faces.target

if __name__ == "__main__":
    X, y = load_olivetti_pca()
    print("Olivetti PCA:", X.shape, y.shape)
