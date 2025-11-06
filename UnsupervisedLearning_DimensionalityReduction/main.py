from Datasets.Olivetti_Faces_Dataset import load_olivetti_pca
from Datasets.Moons_Dataset import load_moons

X_oli, y_oli = load_olivetti_pca(n_components=100)
X_m, y_m = load_moons()

print("Olivetti:", X_oli.shape, y_oli.shape)
print("Moons:", X_m.shape, y_m.shape)
