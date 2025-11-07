import sys
import os
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath("../Data"))

from Load_Data import *

clf = GaussianNB()
clf.fit(X1_train, y1_train)

y1_pred = clf.predict(X1_test)
print("the accry of dataset1's test set：", accuracy_score(y1_test, y1_pred))
cm1 = confusion_matrix(y1_test, y1_pred)
print("Confusion Matrix(dataset1's test set):\n", cm1)

y1_traing_pred = clf.predict(X1_train)
print("the accry of dataset1's traing set：", accuracy_score(y1_train, y1_traing_pred))
cm2 = confusion_matrix(y1_train, y1_traing_pred)
print("Confusion Matrix(dataset1's training set):\n", cm2)


clf2 = GaussianNB()
clf2.fit(X2_train, y2_train)
y2_pred = clf.predict(X2_test)
print("the accry of dataset1's test set：", accuracy_score(y2_test, y2_pred))
cm3 = confusion_matrix(y2_test, y2_pred)
print("Confusion Matrix(dataset2's test set):\n", cm3)


y2_traing_pred = clf.predict(X2_train)
print("the accry of dataset2's traing set：", accuracy_score(y2_train, y2_traing_pred))
cm4 = confusion_matrix(y2_train, y2_traing_pred)
print("Confusion Matrix(dataset2's training set):\n", cm4)

def plot_decision_boundary_with_hits(clf, X_train, y_train, X_test=None, y_test=None, title="Decision Boundary"):
    Xt = X_train.values if hasattr(X_train, "values") else X_train
    yt = y_train.values if hasattr(y_train, "values") else y_train
    Xv = None if X_test is None else (X_test.values if hasattr(X_test, "values") else X_test)
    yv = None if y_test is None else (y_test.values if hasattr(y_test, "values") else y_test)


    X_all = Xt if Xv is None else np.vstack([Xt, Xv])
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5

    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(6.5, 5.5))
    plt.contourf(xx, yy, Z, alpha=0.25)


    y_pred_train = clf.predict(Xt)
    ok_tr = (y_pred_train == yt)
    plt.scatter(Xt[ok_tr, 0],  Xt[ok_tr, 1],  marker='o',  s=28, edgecolors='k', linewidths=0.6, label='train ✓', alpha=0.9)
    plt.scatter(Xt[~ok_tr, 0], Xt[~ok_tr, 1], marker='x',  s=48,                 linewidths=1.2, label='train ✗')


    if Xv is not None and yv is not None:
        y_pred_test = clf.predict(Xv)
        ok_te = (y_pred_test == yv)
        plt.scatter(Xv[ok_te, 0],  Xv[ok_te, 1],  marker='^', s=36, edgecolors='k', linewidths=0.6, label='test ✓', alpha=0.9)
        plt.scatter(Xv[~ok_te, 0], Xv[~ok_te, 1], marker='v', s=36, edgecolors='k', linewidths=0.6, label='test ✗', alpha=0.9)

    plt.xlabel("x1"); plt.ylabel("x2")
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


    plot_decision_boundary_with_hits(
    clf,
    X1_train, y1_train,
    X1_test,  y1_test,
    title="GaussianNB decision boundary for dataset1"
)


plot_decision_boundary_with_hits(
    clf2,
    X2_train, y2_train,
    X2_test,  y2_test,
    title="GaussianNB decision boundary for dataset2"
)