import os
import json  
import re    
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path  

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


RANDOM_STATE = 42 

def _sanitize(name: str) -> str:
    # filename-safe
    return re.sub(r"[^A-Za-z0-9_.()-]+", "_", name)

def _save_text(path: Path, content: str):
    path.write_text(content, encoding="utf-8")

def _save_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

# output dir (fixed: Path math on Path object)
out_dir = current_dir / "Results"
out_dir.mkdir(parents=True, exist_ok=True)

dataset1 = np.load(current_dir / "Dataset_1.npz")
X1, y1 = dataset1["X"], dataset1["y"]

dataset2 = np.load(current_dir / "Dataset_2.npz")
X2, y2, is_outlier = dataset2["X"], dataset2["y"], dataset2["is_outlier"]

# Split Dataset 1
X1_train, X1_test, y1_train, y1_test = train_test_split(
    X1, y1, 
    test_size=0.3,         
    stratify=y1,           
    random_state=42       
)
assert (len(X1_train) + len(X1_test)) == (len(y1_test) + len(y1_train)) == len(X1)

# Split Dataset 2
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, 
    test_size=0.3,         
    stratify=y2,           
    random_state=42       
)
assert (len(X2_train) + len(X2_test)) == (len(y2_test) + len(y2_train)) == len(X2)

def build_knn_search():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    param_grid = {
        "knn__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "knn__weights": ["uniform", "distance"],
        "knn__p": [2],  # Euclidean; add 1 for Manhattan
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE) 
    gs = GridSearchCV(
        pipe, param_grid=param_grid, cv=cv,
        scoring="accuracy", n_jobs=-1, refit=True, return_train_score=False
    )
    return gs

def fit_eval_knn(X_train, y_train, X_test, y_test, label):
    gs = build_knn_search()
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    yhat_tr = best.predict(X_train)
    yhat_te = best.predict(X_test)

    acc_tr = accuracy_score(y_train, yhat_tr)
    acc_te = accuracy_score(y_test, yhat_te)
    cm_tr  = confusion_matrix(y_train, yhat_tr)
    cm_te  = confusion_matrix(y_test,  yhat_te)

    print(f"\n=== {label} — k-NN results ===")
    print("Best params:", gs.best_params_)
    print(f"Train accuracy: {acc_tr:.3f}")
    print(f"Test  accuracy: {acc_te:.3f}")
    print("Confusion matrix (train):\n", cm_tr)
    print("Confusion matrix (test):\n",  cm_te)
    print("\nClassification report (test):\n", classification_report(y_test, yhat_te, digits=3))

    tag = "D1" if "1" in label else ("D2" if "2" in label else _sanitize(label))
    metrics = {
        "label": label,
        "best_params": gs.best_params_,
        "train_accuracy": float(acc_tr),
        "test_accuracy": float(acc_te),
        "confusion_matrix_train": cm_tr.tolist(),
        "confusion_matrix_test": cm_te.tolist(),
    }
    _save_json(out_dir / f"{tag}_metrics.json", metrics)

    clf_rep = classification_report(y_test, yhat_te, digits=3)
    _save_text(out_dir / f"{tag}_classification_report.txt", clf_rep)

    np.savez(out_dir / f"{tag}_predictions.npz",
             y_train=y_train, yhat_train=yhat_tr,
             y_test=y_test,  yhat_test=yhat_te)

    return best, (acc_tr, acc_te), (cm_tr, cm_te)

def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title):
    X_all = np.vstack([X_train, X_test])
    x_min, x_max = X_all[:, 0].min() - 0.8, X_all[:, 0].max() + 0.8
    y_min, y_max = X_all[:, 1].min() - 0.8, X_all[:, 1].max() + 0.8
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.02),
        np.arange(y_min, y_max, 0.02)
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, levels=np.arange(Z.max()+2)-0.5, alpha=0.35)
    plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', s=20, label='train')
    plt.scatter(X_test[:, 0],  X_test[:, 1],  marker='x', s=20, label='test')
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend(loc="best")
    plt.tight_layout()

    fname = _sanitize(title) + ".png"
    plt.savefig(out_dir / fname, dpi=160)
    plt.show()

best1, accs1, cms1 = fit_eval_knn(X1_train, y1_train, X1_test, y1_test, "Dataset 1")
plot_decision_boundary(best1, X1_train, y1_train, X1_test, y1_test,
                       f"Dataset 1 — k-NN boundary ({best1.named_steps['knn'].n_neighbors} neighbors)")

best2, accs2, cms2 = fit_eval_knn(X2_train, y2_train, X2_test, y2_test, "Dataset 2 (with outliers)")
plot_decision_boundary(best2, X2_train, y2_train, X2_test, y2_test,
                       f"Dataset 2 — k-NN boundary ({best2.named_steps['knn'].n_neighbors} neighbors)")

print("\nSummary:")
print("Dataset 1  -> Train acc: %.3f | Test acc: %.3f" % accs1)
print("Dataset 2  -> Train acc: %.3f | Test acc: %.3f" % accs2)
print(f"\nSaved outputs in: {out_dir}")
