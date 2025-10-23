import numpy as np
dataset1 = np.load("Dataset_1.npz")
X1, y1 = dataset1["X"], dataset1["y"]

dataset2 = np.load("Dataset_2.npz")
X2, y2, is_outlier = dataset2["X"], dataset2["y"], dataset2["is_outlier"]