"""
Multi-Layer Perceptron Training for THA2

- Loads THA2train.xlsx / THA2validate.xlsx
- Standardises input features
- Trains MLP with backprop + (mini-batch) SGD
- Runs small hyperparameter sweeps
- Produces train/val loss plots for each learning-rate experiment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from MLP_Model import MLP


# =========================================================
# 1. Data Loading
# =========================================================
def load_tha2_data(train_path="THA2train.xlsx", val_path="THA2validate.xlsx"):
    """
    Load THA2 train and validation sets from Excel.
    Assumes header row: X_0, X_1, y
    """
    df_train = pd.read_excel(train_path)
    df_val = pd.read_excel(val_path)

    # Features
    X_train = df_train[["X_0", "X_1"]].values.astype(np.float32)
    X_val = df_val[["X_0", "X_1"]].values.astype(np.float32)

    # Labels (0/1)
    y_train_int = df_train["y"].values.astype(int)
    y_val_int = df_val["y"].values.astype(int)

    # Standardise features using training statistics
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    # One-hot encode labels
    def one_hot(y, num_classes=2):
        y_oh = np.zeros((len(y), num_classes), dtype=np.float32)
        y_oh[np.arange(len(y)), y] = 1.0
        return y_oh

    y_train = one_hot(y_train_int)
    y_val = one_hot(y_val_int)

    return X_train, y_train, y_train_int, X_val, y_val, y_val_int


# =========================================================
# 2. Loss, metrics, plots
# =========================================================
def compute_loss(y_true, y_pred):
    """
    Cross-entropy loss for one-hot labels.
    y_true, y_pred: (n_samples, n_classes)
    """
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


def confusion_matrix_2x2(y_true, y_pred):
    """
    Simple 2x2 confusion matrix.
    Rows = true labels, Cols = predicted labels.
    """
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_losses(train_losses, val_losses, title="Training vs Validation Loss",
                save_path=None):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


# =========================================================
# 3. Training loop with validation (mini-batch SGD)
# =========================================================
def train_with_validation(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=200,
    batch_size=None,
    verbose=True,
):
    """
    Train the MLP using mini-batch SGD and track train/val loss.

    - If batch_size is None or > n_train, full-batch GD is used.
    """
    n_train = X_train.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size > n_train:
        batch_size = n_train  # full-batch

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Shuffle training data every epoch
        perm = np.random.permutation(n_train)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        epoch_loss = 0.0

        # Mini-batch loop
        for start in range(0, n_train, batch_size):
            end = start + batch_size
            xb = X_sh[start:end]
            yb = y_sh[start:end]

            # Forward on batch
            y_pred_batch = model.forward(xb)
            loss_batch = compute_loss(yb, y_pred_batch)

            # Backward + parameter update
            model.backward(xb, yb, y_pred_batch)

            # Accumulate loss, weighted by batch size
            epoch_loss += loss_batch * len(xb)

        # Average loss over all samples
        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        # Validation loss (always on full val set)
        y_pred_val = model.forward(X_val)
        loss_val = compute_loss(y_val, y_pred_val)
        val_losses.append(loss_val)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1):
            acc_val = np.mean(
                np.argmax(y_pred_val, axis=1) == np.argmax(y_val, axis=1)
            )
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train Loss: {epoch_loss:.4f} | "
                f"Val Loss: {loss_val:.4f} | "
                f"Val Acc: {acc_val:.4f}"
            )

    return train_losses, val_losses


# =========================================================
# 4. Small helper for experiments (LR / batch / init)
# =========================================================
def run_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    lr,
    batch_size,
    init_scale,
    epochs=150,
    verbose=False,
    make_plots=False,
    plot_prefix=None,
):
    """
    Train a model with given hyperparameters and report final metrics.

    If make_plots=True, a train/val loss plot is shown and optionally saved.
    """
    mlp = MLP(
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=lr,
        activation="relu",
        init_scale=init_scale,
    )

    print(
        f"\n=== Experiment: lr={lr}, batch_size={batch_size}, init_scale={init_scale} ==="
    )

    train_losses, val_losses = train_with_validation(
        mlp,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    # Final validation metrics
    y_pred_val = mlp.predict(X_val)
    y_prob_val = mlp.forward(X_val)
    final_loss = compute_loss(y_val, y_prob_val)
    y_val_int = np.argmax(y_val, axis=1)
    final_acc = np.mean(y_pred_val == y_val_int)

    print(f"Final Val Loss: {final_loss:.4f}, Final Val Acc: {final_acc:.4f}")

    # Optional plot (used for learning-rate experiments)
    if make_plots:
        # Create a nice title and filename
        title = f"Training vs Validation Loss (lr={lr}, batch={batch_size}, init={init_scale})"
        filename = None
        if plot_prefix is not None:
            lr_str = str(lr).replace(".", "p")
            init_str = str(init_scale).replace(".", "p")
            bs_str = "full" if batch_size is None else str(batch_size)
            filename = f"{plot_prefix}_lr{lr_str}_bs{bs_str}_init{init_str}.png"
        plot_losses(train_losses, val_losses, title=title, save_path=filename)

    return {
        "lr": lr,
        "batch_size": batch_size,
        "init_scale": init_scale,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "final_loss": final_loss,
        "final_acc": final_acc,
    }


# =========================================================
# 5. Main
# =========================================================
if __name__ == "__main__":
    print("=== THA2: MLP Training (Part A.3) ===")

    # Turn this on to re-run sweeps & generate plots per LR
    RUN_HYPERPARAM_EXPERIMENTS = True

    # Load data
    X_train, y_train, y_train_int, X_val, y_val, y_val_int = load_tha2_data()
    print(f"Training samples:   {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}\n")

    # -----------------------------------------------------
    # Hyperparameter sweeps
    # -----------------------------------------------------
    if RUN_HYPERPARAM_EXPERIMENTS:
        # ---------- 1) Learning-rate sweep (plots here) ----------
        results_lr = []
        for lr in [0.1, 0.01, 0.001]:
            res = run_experiment(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=lr,
                batch_size=None,      # full-batch for LR sweep
                init_scale=0.1,
                epochs=200,
                verbose=False,
                make_plots=True,      # <- THIS makes a plot per learning rate
                plot_prefix="loss_lr" # saved as PNGs
            )
            results_lr.append(res)

        best_lr = min(results_lr, key=lambda r: r["final_loss"])["lr"]
        print(f"\n[SWEEP] Best learning rate (by val loss): {best_lr}")

        # ---------- 2) Batch-size sweep ----------
        results_bs = []
        for bs in [None, 16, 4]:
            res = run_experiment(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=best_lr,
                batch_size=bs,
                init_scale=0.1,
                epochs=200,
                verbose=False,
                make_plots=False,
            )
            results_bs.append(res)

        best_bs = min(results_bs, key=lambda r: r["final_loss"])["batch_size"]
        print(f"[SWEEP] Best batch size (by val loss): {best_bs}")

        # ---------- 3) Init-scale sweep ----------
        results_init = []
        for init_scale in [0.01, 0.1, 1.0]:
            res = run_experiment(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=best_lr,
                batch_size=best_bs,
                init_scale=init_scale,
                epochs=200,
                verbose=False,
                make_plots=False,
            )
            results_init.append(res)

        best_init = min(results_init, key=lambda r: r["final_loss"])["init_scale"]
        print(f"[SWEEP] Best init scale (by val loss): {best_init}")

        CHOSEN_LR = best_lr
        CHOSEN_BS = best_bs
        CHOSEN_INIT = best_init
    else:
        # If you don't want to rerun sweeps, hard-code the chosen values:
        CHOSEN_LR = 0.1
        CHOSEN_BS = 16
        CHOSEN_INIT = 0.1

    # -----------------------------------------------------
    # Final training run with chosen hyperparameters
    # -----------------------------------------------------
    print(
        f"\n=== Final training with lr={CHOSEN_LR}, "
        f"batch_size={CHOSEN_BS}, init_scale={CHOSEN_INIT} ==="
    )

    mlp = MLP(
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=CHOSEN_LR,
        activation="relu",
        init_scale=CHOSEN_INIT,
    )

    mlp.get_network_info()

    train_losses, val_losses = train_with_validation(
        mlp,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=CHOSEN_BS,
        verbose=True,
    )

    # Plot final loss curves (for A.3 main figure)
    plot_losses(train_losses, val_losses,
                title="Training vs Validation Loss (final model)",
                save_path="loss_final_model.png")

    # Final evaluation on validation set
    print("\n--- Final Evaluation on Validation Set ---")
    y_pred_val = mlp.predict(X_val)
    y_prob_val = mlp.forward(X_val)
    final_loss = compute_loss(y_val, y_prob_val)
    final_acc = np.mean(y_pred_val == y_val_int)
    cm = confusion_matrix_2x2(y_val_int, y_pred_val)

    print(f"Final Validation Loss: {final_loss:.4f}")
    print(f"Final Validation Accuracy: {final_acc:.4f}")
    print("Confusion Matrix (rows = true, cols = pred):")
    print(cm)
