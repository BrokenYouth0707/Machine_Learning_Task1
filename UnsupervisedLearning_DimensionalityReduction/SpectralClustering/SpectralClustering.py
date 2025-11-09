# SpectralClustering.py
import pathlib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score as ARI
from sklearn.decomposition import PCA

from Datasets.Olivetti_Faces_Dataset import load_olivetti_pca

def main(k: int, nn: int, pca_dims: int):
    # --- Data ---
    X_oli, y_oli = load_olivetti_pca(n_components=pca_dims, whiten=True, scale=True)

    # --- Model ---
    labels = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=nn,
        assign_labels="kmeans",
        random_state=42
    ).fit_predict(X_oli)

    # --- Metrics ---
    sil = float(silhouette_score(X_oli, labels)) if len(set(labels)) > 1 else float("nan")
    ari = float(ARI(y_oli, labels))

    # --- I/O ---
    out = pathlib.Path("Results")
    out.mkdir(exist_ok=True)

    # Save CSV row
    row = pd.DataFrame([{
        "dataset": f"olivetti_pca{pca_dims}",
        "n_neighbors": nn,
        "n_clusters": k,
        "silhouette": sil,
        "ARI": ari,
        "seed": 42
    }])
    csv_path = out / f"spectral_olivetti_pca{pca_dims}_metrics.csv"
    header = not csv_path.exists()
    row.to_csv(csv_path, mode="a", header=header, index=False)

    # Quick 2D plot (PCA just for visualization)
    X2 = PCA(n_components=2, random_state=42).fit_transform(X_oli)
    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=12)
    plt.title(f"Spectral (k={k}, nn={nn})  Sil={sil:.3f}  ARI={ari:.3f}")
    plt.tight_layout()
    png_path = out / f"spectral_olivetti_pca{pca_dims}_k{k}_nn{nn}.png"
    plt.savefig(png_path, dpi=180)
    plt.close()

    print("Done.")
    print("  X shape:", X_oli.shape)
    print("  Clusters:", len(set(labels)))
    print("  Silhouette:", round(sil, 3))
    print("  ARI:", round(ari, 3))
    print("  CSV:", csv_path)
    print("  Figure:", png_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40, help="n_clusters")
    ap.add_argument("--nn", type=int, default=12, help="n_neighbors")
    ap.add_argument("--pca", type=int, default=100, help="PCA dimensions for Olivetti")
    args = ap.parse_args()
    main(args.k, args.nn, args.pca)
