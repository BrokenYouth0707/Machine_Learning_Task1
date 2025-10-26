import numpy as np
import sys
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt

# add Data folder to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Data'))
import Load_Data

# Access Dataset 1
X1_train, X1_test = Load_Data.X1_train, Load_Data.X1_test
y1_train, y1_test = Load_Data.y1_train, Load_Data.y1_test

# Access Dataset 2
X2_train, X2_test = Load_Data.X2_train, Load_Data.X2_test
y2_train, y2_test = Load_Data.y2_train, Load_Data.y2_test


######################################################################
# ========== Decision Tree Models ==========
print("\n" + "="*50)
print("Decision Tree Classifier")
print("="*50)

# Decision Tree for Dataset 1
print("\n--- Dataset 1 ---")
# use 'gini' criterion for splitting and set max_depth of tree to 10 preventing overfitting
# set random_state for reproducibility
dt1 = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=10)
dt1.fit(X1_train, y1_train)
y1_pred_dt = dt1.predict(X1_test)
print(f"Train Accuracy: {dt1.score(X1_train, y1_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y1_test, y1_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y1_test, y1_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred_dt))

# Decision Tree for Dataset 2
print("\n--- Dataset 2 ---")
dt2 = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=10)
dt2.fit(X2_train, y2_train)
y2_pred_dt = dt2.predict(X2_test)
print(f"Train Accuracy: {dt2.score(X2_train, y2_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y2_test, y2_pred_dt):.4f}")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred_dt))
print("Confusion Matrix:")
print(confusion_matrix(y2_test, y2_pred_dt))


######################################################################
# ========== Random Forest Models ==========
print("\n" + "="*50)
print("Random Forest Classifier")
print("="*50)

# Random Forest for Dataset 1
print("\n--- Dataset 1 ---")
# use 100 decision trees
rf1 = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42, max_depth=10)
rf1.fit(X1_train, y1_train)
y1_pred_rf = rf1.predict(X1_test)
print(f"Train Accuracy: {rf1.score(X1_train, y1_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y1_test, y1_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y1_test, y1_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y1_test, y1_pred_rf))

# Random Forest for Dataset 2
print("\n--- Dataset 2 ---")
rf2 = RandomForestClassifier(criterion='gini', n_estimators=100, random_state=42, max_depth=10)
rf2.fit(X2_train, y2_train)
y2_pred_rf = rf2.predict(X2_test)
print(f"Train Accuracy: {rf2.score(X2_train, y2_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y2_test, y2_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y2_test, y2_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y2_test, y2_pred_rf))

# ========== Summary Comparison ==========
print("\n" + "="*50)
print("Summary Comparison")
print("="*50)
print(f"{'Model':<20} {'Dataset':<10} {'Train Acc':<12} {'Test Acc':<12}")
print("-"*54)
print(f"{'Decision Tree':<20} {'1':<10} {dt1.score(X1_train, y1_train):<12.4f} {accuracy_score(y1_test, y1_pred_dt):<12.4f}")
print(f"{'Decision Tree':<20} {'2':<10} {dt2.score(X2_train, y2_train):<12.4f} {accuracy_score(y2_test, y2_pred_dt):<12.4f}")
print(f"{'Random Forest':<20} {'1':<10} {rf1.score(X1_train, y1_train):<12.4f} {accuracy_score(y1_test, y1_pred_rf):<12.4f}")
print(f"{'Random Forest':<20} {'2':<10} {rf2.score(X2_train, y2_train):<12.4f} {accuracy_score(y2_test, y2_pred_rf):<12.4f}")


######################################################################
# ========== Decision Boundary Visualization ==========
print("\n" + "="*50)
print("Generating Decision Boundary Plots")
print("="*50)

# Create output directory for plots
output_dir = os.path.join(os.path.dirname(__file__), 'decision_boundaries')
os.makedirs(output_dir, exist_ok=True)

# Function to plot decision boundaries
def plot_decision_boundary(model, X_train, y_train, X_test, y_test, title, filename):
    """
    Plots the decision boundary of a classifier along with training and test data points.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # plot decision boundary
    DecisionBoundaryDisplay.from_estimator(
        model,
        X_train,
        cmap=plt.cm.RdYlBu,
        alpha=0.3,
        ax=ax,
        response_method="predict",
        plot_method="contourf"
    )
    
    # plot training data points
    scatter_train = ax.scatter(
        X_train[:, 0], X_train[:, 1],
        c=y_train,
        cmap=plt.cm.RdYlBu,
        edgecolors='k',
        s=50,
        alpha=0.6,
        label='Training data'
    )
    
    # plot test data points
    scatter_test = ax.scatter(
        X_test[:, 0], X_test[:, 1],
        c=y_test,
        cmap=plt.cm.RdYlBu,
        edgecolors='k',
        s=100,
        alpha=0.8,
        marker='^',
        label='Test data'
    )
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved {filename}")


# Plot Decision Tree boundaries
plot_decision_boundary(
    dt1, X1_train, y1_train, X1_test, y1_test,
    'Decision Tree - Dataset 1',
    'dt_dataset1.png'
)

plot_decision_boundary(
    dt2, X2_train, y2_train, X2_test, y2_test,
    'Decision Tree - Dataset 2',
    'dt_dataset2.png'
)

# Plot Random Forest boundaries
plot_decision_boundary(
    rf1, X1_train, y1_train, X1_test, y1_test,
    'Random Forest - Dataset 1',
    'rf_dataset1.png'
)

plot_decision_boundary(
    rf2, X2_train, y2_train, X2_test, y2_test,
    'Random Forest - Dataset 2',
    'rf_dataset2.png'
)

# Create comparison plot (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Decision Boundaries Comparison', fontsize=16, fontweight='bold')

models = [(dt1, 'Decision Tree - Dataset 1'), 
          (dt2, 'Decision Tree - Dataset 2'),
          (rf1, 'Random Forest - Dataset 1'), 
          (rf2, 'Random Forest - Dataset 2')]
datasets = [(X1_train, y1_train, X1_test, y1_test),
            (X2_train, y2_train, X2_test, y2_test),
            (X1_train, y1_train, X1_test, y1_test),
            (X2_train, y2_train, X2_test, y2_test)]

for idx, (ax, (model, title), (X_tr, y_tr, X_te, y_te)) in enumerate(zip(axes.flat, models, datasets)):
    DecisionBoundaryDisplay.from_estimator(
        model, X_tr,
        cmap=plt.cm.RdYlBu,
        alpha=0.3,
        ax=ax,
        response_method="predict",
        plot_method="contourf"
    )
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap=plt.cm.RdYlBu, 
               edgecolors='k', s=30, alpha=0.6)
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap=plt.cm.RdYlBu,
               edgecolors='k', s=60, alpha=0.8, marker='^')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

plt.tight_layout()
comparison_file = os.path.join(output_dir, 'comparison_all.png')
plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
plt.close()

print(f"[plot] saved comparison_all.png")

print(f"\nAll decision boundary plots saved in: {output_dir}")


