import numpy as np
from sklearn.model_selection import train_test_split



dataset1 = np.load("Dataset_1.npz")
X1, y1 = dataset1["X"], dataset1["y"]

dataset2 = np.load("Dataset_2.npz")
X2, y2, is_outlier = dataset2["X"], dataset2["y"], dataset2["is_outlier"]

X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, 
    test_size=0.3,         
    stratify=y1,           
    random_state=42       
)

assert (len(X1_train) + len(X1_test)) == (len(y1_test) + len(y1_train)) == len(X1)

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, 
    test_size=0.3,         
    stratify=y2,           
    random_state=42       
)

assert (len(X2_train) + len(X2_test)) == (len(y2_test) + len(y2_train)) == len(X2)
