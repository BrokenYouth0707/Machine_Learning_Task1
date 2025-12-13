"""
Multi-Layer Perceptron Training for THA2

Part A:
- Loads THA2train.xlsx / THA2validate.xlsx
- Standardises input features
- Trains MLP with backprop + (mini-batch) SGD
- Runs hyperparameter sweeps for:
    * learning rate
    * batch size
    * init scale
- Trains final "best" model and plots its loss.

Part B.1:
- Recreates the best model but initializes ALL weights and biases to zero.
- Trains it with the same settings.
- Plots its loss and reports final accuracy + confusion matrix.

Part B.2:
- Grid over learning rate and initialisation variance sigma^2 for weights/biases
  drawn from N(0, sigma^2). Builds a heatmap of validation accuracy.

Part B.3:
- Takes a "best" and a "worst" model from B.2 and visualises the hidden-layer
  activations at three times: init, mid-training, end of training.
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


def plot_losses(train_losses,
                val_losses,
                title="Training vs Validation Loss",
                save_path=None):
    """Plot train/val loss curves and optionally save to file."""
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

        # Average training loss over all samples
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
# 3b. B.2 heatmap: learning rate vs init variance
# =========================================================
def run_lr_init_heatmap(
    X_train,
    y_train,
    X_val,
    y_val,
    y_val_int,
    batch_size=16,
    epochs=100,
    base_seed=42,
):
    """
    For several learning rates and initialization variances sigma^2,
    train the same MLP and record validation accuracy.
    Returns the accuracy matrix and the grids of learning rates and sigma^2 values.
    """

    # Learning rates
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    # Variances sigma^2 from the statement
    sigma_sq_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    acc_matrix = np.zeros((len(learning_rates), len(sigma_sq_values)), dtype=float)

    for i, lr in enumerate(learning_rates):
        for j, sigma_sq in enumerate(sigma_sq_values):
            # same seed for fairness
            np.random.seed(base_seed)

            # variance -> std
            if sigma_sq == 0.0:
                init_scale = 0.0
            else:
                init_scale = np.sqrt(sigma_sq)

            mlp = MLP(
                input_size=2,
                hidden_sizes=[10, 10],
                output_size=2,
                learning_rate=lr,
                activation="relu",
                init_scale=init_scale,
            )

            # Train for fixed number of epochs
            train_with_validation(
                mlp,
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=False,
            )

            # Validation accuracy after training
            y_pred_val = mlp.predict(X_val)
            acc = np.mean(y_pred_val == y_val_int)
            acc_matrix[i, j] = acc

            print(
                f"[B.2] lr={lr:.0e}, sigma^2={sigma_sq:.1f} -> val acc={acc:.4f}"
            )

    # ---- Plot heatmap ----
    plt.figure(figsize=(8, 5))
    im = plt.imshow(
        acc_matrix,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=0.5,
        vmax=1.0,
    )
    plt.colorbar(im, label="Validation accuracy")

    # Tick labels
    plt.xticks(
        ticks=np.arange(len(sigma_sq_values)),
        labels=[f"{s:.1f}" for s in sigma_sq_values],
    )
    plt.yticks(
        ticks=np.arange(len(learning_rates)),
        labels=[f"{lr:.0e}" for lr in learning_rates],
    )

    plt.xlabel("σ² (variance of N(0, σ²))")
    plt.ylabel("Learning rate η")
    plt.title("Validation accuracy vs learning rate and initialization variance")
    plt.tight_layout()
    plt.savefig("heatmap_lr_init_accuracy.png", dpi=300)
    plt.show()

    return learning_rates, sigma_sq_values, acc_matrix


# =========================================================
# 3c. B.3 helpers – hidden activations
# =========================================================
def get_hidden_outputs(model, X):
    """
    Run a forward pass and return hidden-layer activations.
    Assumes model.activations = [input, h0, h1, output]
    """
    _ = model.forward(X)
    h0 = model.activations[1].copy()
    h1 = model.activations[2].copy()
    return h0, h1


def plot_hidden_heatmap(h, title, save_path=None):
    """
    Heatmap of hidden activations.
    Rows: validation samples (sorted by class).
    Columns: hidden neurons.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(h, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Activation value")
    plt.xlabel("Hidden neuron index")
    plt.ylabel("Validation sample (sorted by class)")
    plt.title(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()


def train_and_capture_activations_for_model(
    lr,
    sigma_sq,
    X_train,
    y_train,
    X_val_sorted,
    y_val_sorted,
    y_val_int_sorted,
    epochs=100,
    batch_size=16,
    label_prefix="b3_model",
    seed=123,
):
    """
    Train a model with given (lr, sigma^2) and save hidden activations
    at three times: init, mid-training, end of training.
    """

    # variance -> std used by MLP init
    if sigma_sq == 0.0:
        init_scale = 0.0
    else:
        init_scale = np.sqrt(sigma_sq)

    np.random.seed(seed)

    mlp = MLP(
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=lr,
        activation="relu",
        init_scale=init_scale,
    )

    n_train = X_train.shape[0]
    if batch_size is None or batch_size <= 0 or batch_size > n_train:
        batch_size = n_train

    snapshots = {}

    # ---- 1) Before training (after initialisation) ----
    h0_init, h1_init = get_hidden_outputs(mlp, X_val_sorted)
    snapshots["init"] = (h0_init, h1_init)

    # ---- 2) Training loop with capture at mid and end ----
    mid_epoch = epochs // 2

    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        X_sh = X_train[perm]
        y_sh = y_train[perm]

        for start in range(0, n_train, batch_size):
            end = start + batch_size
            xb = X_sh[start:end]
            yb = y_sh[start:end]
            y_pred_batch = mlp.forward(xb)
            mlp.backward(xb, yb, y_pred_batch)

        if (epoch + 1) == mid_epoch:
            h0_mid, h1_mid = get_hidden_outputs(mlp, X_val_sorted)
            snapshots["mid"] = (h0_mid, h1_mid)

    # Snapshot at the end of training
    h0_final, h1_final = get_hidden_outputs(mlp, X_val_sorted)
    snapshots["final"] = (h0_final, h1_final)

    # ---- Plot all snapshots ----
    for phase, (h0, h1) in snapshots.items():
        plot_hidden_heatmap(
            h0,
            title=f"{label_prefix} – hidden layer 1 ({phase})",
            save_path=f"activations_{label_prefix}_layer1_{phase}.png",
        )
        plot_hidden_heatmap(
            h1,
            title=f"{label_prefix} – hidden layer 2 ({phase})",
            save_path=f"activations_{label_prefix}_layer2_{phase}.png",
        )

    # Return final val accuracy (on sorted val set)
    y_pred_val = mlp.predict(X_val_sorted)
    acc = np.mean(y_pred_val == y_val_int_sorted)
    return acc


# =========================================================
# 4. Helper for sweeps (Part A)
# =========================================================
def run_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    lr,
    batch_size,
    init_scale,
    epochs=200,
    verbose=False,
    make_plots=False,
    plot_prefix=None,
):
    """
    Train a model with given hyperparameters and report final metrics.
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

    # Optional plot
    if make_plots:
        bs_str = "full" if batch_size is None else str(batch_size)
        init_str = str(init_scale).replace(".", "p")
        lr_str = str(lr).replace(".", "p")

        title = (
            f"Train vs Val Loss (lr={lr}, batch={bs_str}, init={init_scale})"
        )
        filename = None
        if plot_prefix is not None:
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
    print("=== THA2: MLP Training ===")

    # Sweeps
    RUN_HYPERPARAM_EXPERIMENTS = False

    # Load data
    X_train, y_train, y_train_int, X_val, y_val, y_val_int = load_tha2_data()
    print(f"Training samples:   {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}\n")

    # -----------------------------------------------------
    # Hyperparameter sweeps (Part A.2 & A.3)
    # -----------------------------------------------------
    if RUN_HYPERPARAM_EXPERIMENTS:
        # 1) Learning-rate sweep
        results_lr = []
        for lr in [0.1, 0.01, 0.001]:
            res = run_experiment(
                X_train,
                y_train,
                X_val,
                y_val,
                lr=lr,
                batch_size=None,
                init_scale=0.1,
                epochs=200,
                verbose=False,
                make_plots=True,
                plot_prefix="loss_lr",
            )
            results_lr.append(res)
        best_lr = min(results_lr, key=lambda r: r["final_loss"])["lr"]
        print(f"\n[SWEEP] Best learning rate (by val loss): {best_lr}")

        # 2) Batch-size sweep
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
                make_plots=True,
                plot_prefix="loss_bs",
            )
            results_bs.append(res)
        best_bs = min(results_bs, key=lambda r: r["final_loss"])["batch_size"]
        print(f"[SWEEP] Best batch size (by val loss): {best_bs}")

        # 3) Init-scale sweep
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
                make_plots=True,
                plot_prefix="loss_init",
            )
            results_init.append(res)
        best_init = min(results_init, key=lambda r: r["final_loss"])["init_scale"]
        print(f"[SWEEP] Best init scale (by val loss): {best_init}")

        CHOSEN_LR = best_lr
        CHOSEN_BS = best_bs
        CHOSEN_INIT = best_init
    else:
        # Best settings we found
        CHOSEN_LR = 0.1
        CHOSEN_BS = 16
        CHOSEN_INIT = 0.1

    # -----------------------------------------------------
    # Final training run with chosen hyperparameters (Part A)
    # -----------------------------------------------------
    print(
        f"\n=== Final model from Part A "
        f"(lr={CHOSEN_LR}, batch_size={CHOSEN_BS}, init_scale={CHOSEN_INIT}) ==="
    )

    mlp_best = MLP(
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=CHOSEN_LR,
        activation="relu",
        init_scale=CHOSEN_INIT,
    )

    mlp_best.get_network_info()

    train_losses_best, val_losses_best = train_with_validation(
        mlp_best,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=CHOSEN_BS,
        verbose=True,
    )

    # Plot final loss curves (best model)
    plot_losses(
        train_losses_best,
        val_losses_best,
        title="Training vs Validation Loss (final model)",
        save_path="loss_final_model.png",
    )

    # Final evaluation on validation set (best model)
    print("\n--- Final Evaluation on Validation Set (best model) ---")
    y_pred_val_best = mlp_best.predict(X_val)
    y_prob_val_best = mlp_best.forward(X_val)
    final_loss_best = compute_loss(y_val, y_prob_val_best)
    final_acc_best = np.mean(y_pred_val_best == y_val_int)
    cm_best = confusion_matrix_2x2(y_val_int, y_pred_val_best)

    print(f"Final Validation Loss: {final_loss_best:.4f}")
    print(f"Final Validation Accuracy: {final_acc_best:.4f}")
    print("Confusion Matrix (rows = true, cols = pred):")
    print(cm_best)

    # -----------------------------------------------------
    # Zero-initialised model (Part B.1)
    # -----------------------------------------------------
    print(
        f"\n=== Zero-initialised model "
        f"(same settings as best model) ==="
    )

    mlp_zero = MLP(
        input_size=2,
        hidden_sizes=[10, 10],
        output_size=2,
        learning_rate=CHOSEN_LR,
        activation="relu",
        init_scale=CHOSEN_INIT,  # will be overwritten to zeros
    )

    # Overwrite all parameters with zeros
    for i in range(len(mlp_zero.weights)):
        mlp_zero.weights[i] = np.zeros_like(mlp_zero.weights[i])
        mlp_zero.biases[i] = np.zeros_like(mlp_zero.biases[i])

    train_losses_zero, val_losses_zero = train_with_validation(
        mlp_zero,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=200,
        batch_size=CHOSEN_BS,
        verbose=True,
    )

    # Plot zero-init loss curves
    plot_losses(
        train_losses_zero,
        val_losses_zero,
        title="Training vs Validation Loss (zero-initialised model)",
        save_path="loss_zero_init_model.png",
    )

    # Final evaluation on validation set (zero-init model)
    print("\n--- Final Evaluation on Validation Set (zero-init model) ---")
    y_pred_val_zero = mlp_zero.predict(X_val)
    y_prob_val_zero = mlp_zero.forward(X_val)
    final_loss_zero = compute_loss(y_val, y_prob_val_zero)
    final_acc_zero = np.mean(y_pred_val_zero == y_val_int)
    cm_zero = confusion_matrix_2x2(y_val_int, y_pred_val_zero)

    # -----------------------------------------------------
    # Part B.2: Learning rate vs initialization heatmap
    # -----------------------------------------------------
    RUN_HEATMAP = True
    if RUN_HEATMAP:
        print("\n=== Learning rate vs initialization heatmap ===")
        lrs, sigmasq, acc_matrix = run_lr_init_heatmap(
            X_train,
            y_train,
            X_val,
            y_val,
            y_val_int,
            batch_size=16,
            epochs=100,   # fixed number of epochs for this part
            base_seed=123,
        )

    # -----------------------------------------------------
    # Part B.3: Activation visualisation for worst & best B.2 models
    # -----------------------------------------------------
    # Sort validation samples by class (0 first, then 1)
    order = np.argsort(y_val_int)
    X_val_sorted = X_val[order]
    y_val_sorted = y_val[order]
    y_val_int_sorted = y_val_int[order]

    # Best B.2 model (example: lr=1e-2, sigma^2=0.1, acc=0.9756)
    print("\n=== Best model activations (lr=1e-2, sigma^2=0.1) ===")
    acc_best_b3 = train_and_capture_activations_for_model(
        lr=1e-2,
        sigma_sq=0.1,
        X_train=X_train,
        y_train=y_train,
        X_val_sorted=X_val_sorted,
        y_val_sorted=y_val_sorted,
        y_val_int_sorted=y_val_int_sorted,
        epochs=100,
        batch_size=16,
        label_prefix="best_b3",
    )
    print(f"Final val acc (best B.3 model): {acc_best_b3:.4f}")

    # Worst B.2 model that still trains (lr=1e-5, sigma^2=1.0, acc≈0.3171)
    print("\n=== Worst model activations (lr=1e-5, sigma^2=1.0) ===")
    acc_worst_b3 = train_and_capture_activations_for_model(
        lr=1e-5,
        sigma_sq=1.0,
        X_train=X_train,
        y_train=y_train,
        X_val_sorted=X_val_sorted,
        y_val_sorted=y_val_sorted,
        y_val_int_sorted=y_val_int_sorted,
        epochs=100,
        batch_size=16,
        label_prefix="worst_b3",
    )
    print(f"Final val acc (worst B.3 model): {acc_worst_b3:.4f}")

    # -----------------------------------------------------
    # Print zero-init summary last
    # -----------------------------------------------------
    print(f"\nFinal Validation Loss (zero init): {final_loss_zero:.4f}")
    print(f"Final Validation Accuracy (zero init): {final_acc_zero:.4f}")
    print("Confusion Matrix (rows = true, cols = pred):")
    print(cm_zero)
