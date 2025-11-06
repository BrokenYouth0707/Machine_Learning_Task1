import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'LeastSquares')))
import DataPoints as dp

# Data inverse transformation
def zscore_inverse(arr_t: np.ndarray, params: Tuple[float, float]) -> np.ndarray:
    mu, sigma = params
    return arr_t * sigma + mu

# Regression model: h_θ(x) = θ0 + θ1*x + θ2*x^2
def h_theta(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    # θ = [θ0, θ1, θ2]
    return theta[2] * np.power(x,2) + theta[1] * x + theta[0]

# Cost function: J(θ) = (1/(4n)) * Σ (y - h_θ(x))^4
def cost_function(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    n = x.shape[0]
    # ε = y - h_θ(x)
    eps = y - h_theta(theta, x)
    return float(np.sum(eps ** 4) / (4.0 * n))

# Gradient of the cost function
def grad_of_cost(theta: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # ∂J/∂θ0 = -(1/n) Σ ε^3
    # ∂J/∂θ1 = -(1/n) Σ ε^3 x
    # ∂J/∂θ2 = -(1/n) Σ ε^3 x^2
    n = x.shape[0]
    eps = y - h_theta(theta, x)
    e3 = eps ** 3
    g0 = -np.sum(e3) / n
    g1 = -np.sum(e3 * x) / n
    g2 = -np.sum(e3 * (x ** 2)) / n
    return np.array([g0, g1, g2], dtype=float)

# Gradient descent algorithm
def gradient_descent(x: np.ndarray, y: np.ndarray, alpha: float, epochs: int, 
                     seed: Optional[int] = None, 
                     snapshot_iters: Optional[List[int]] = None) -> Tuple[np.ndarray, List[float], List[Tuple[int, np.ndarray]]]:
    # Create a random number generator with the specified seed for reproducibility
    rng = np.random.default_rng(seed)
    # Initialize θ with small random values
    theta = rng.normal(loc=0.0, scale=0.1, size=3).astype(float)  # θ0, θ1, θ2
    # Initialize lists to store cost values and snapshots
    costs: List[float] = []
    snaps: List[Tuple[int, np.ndarray]] = []
    # Default snapshot iterations if none provided
    if snapshot_iters is None:
        snapshot_iters = [0, epochs - 1]

    # Store the initial value of θ
    snaps.append((0, theta.copy()))
    # Record initial cost
    costs.append(cost_function(theta, x, y))

    # Perform gradient descent
    for i in range(1, epochs+1):
        # Update new θ
        theta = theta - alpha * grad_of_cost(theta, x, y)
        # Record cost at this iteration
        costs.append(cost_function(theta, x, y))
        # Take snapshot according to specified iterations
        if i in snapshot_iters:
            # Store a copy of new θ
            snaps.append((i, theta.copy()))
    
    return theta, costs, snaps

# Plotting functions for fit progress and cost
def plot_fit_progress(x_raw: np.ndarray, y_raw: np.ndarray,
                      x_t: np.ndarray, snapshots: List[Tuple[int, np.ndarray]], 
                      output_dir: str, prefix: str):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Build a dense x-grid in transformed space
    x_grid_t = np.linspace(float(np.min(x_t)), float(np.max(x_t)), 300)
    # For plotting in original scale, we need predicted y in transformed space then inverse y
    for it, theta in snapshots:
        # Predict y in transformed space
        y_hat_t = h_theta(theta, x_grid_t)
        # Calculate z-score parameters from raw data for inverse transformation
        x_params = (float(np.mean(x_raw)), float(np.std(x_raw)))
        y_params = (float(np.mean(y_raw)), float(np.std(y_raw)))
        # map y_hat_t back to original y for plotting
        y_hat = zscore_inverse(y_hat_t, y_params)
        # map x_grid_t back to original x for plotting
        x_grid = zscore_inverse(x_grid_t, x_params)


        plt.figure()
        # Plot raw data points
        plt.scatter(x_raw, y_raw, s=18, label="data", color='blue')
        # Plot the polynomial fit
        plt.plot(x_grid, y_hat, linewidth=2, label=f"fit at iter {it}", color='black', linestyle='--')
        plt.xlabel("Temperature (X)")
        plt.ylabel("Net hourly electrical energy output (Y)")
        plt.title(f"Polynomial regression fit (iteration {it})")
        plt.legend()
        # Save the plot
        fname = os.path.join(output_dir, f"{prefix}_fit_iter_{it:06d}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {fname}")

# Plot cost over iterations
def plot_cost(costs: List[float], output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure()
    # Plot cost on a logarithmic scale
    plt.semilogy(np.arange(len(costs)), costs)
    plt.xlabel("Iteration")
    plt.ylabel("J(θ) = (1/(4n)) Σ (y - h)^4 (log scale)")
    plt.title("Cost vs iteration (quartic loss)")
    plt.tight_layout()
    # Save the plot
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved {output_path}")

def main():
    # define parameters
    alpha = 0.01
    epochs = 3000
    seed = 1
    group = 11
    outdir = "outputs_AII"
    # Snapshot list
    snapshot_iters: List[int] = [0, 100, 200, 500, 1000, 2000, 3000]

    # Load the group's raw data
    x_raw, y_raw = dp.a, dp.b

    # transformed data
    x_t, y_t = dp.x, dp.y

    # Run gradient descent
    theta_final, costs, snaps = gradient_descent(
        x=x_t, y=y_t,
        alpha=alpha, epochs=epochs, seed=seed,
        snapshot_iters=snapshot_iters
    )

    # Print required info
    print("=== A.II Polynomial Regression (quartic loss) ===")
    print(f"Group number: {group}")
    print(f"Transform: zscore")
    print(f"Learning rate alpha: {alpha}")
    print(f"Epochs: {epochs}")
    print("Initial θ was drawn ~ N(0, 0.1^2) with the provided seed.")
    print("--- Snapshots (iteration, θ0, θ1, θ2) ---")
    for it, th in snaps:
        print(f"iter {it:>6d}: [θ0, θ1, θ2] = [{th[0]: .6e}, {th[1]: .6e}, {th[2]: .6e}] | J = {cost_function(th, x_t, y_t):.6e}")
    print("--- Final parameters ---")
    print(f"[θ0, θ1, θ2] = [{theta_final[0]: .10f}, {theta_final[1]: .10f}, {theta_final[2]: .10f}]")
    print(f"Final cost: J(θ) = {cost_function(theta_final, x_t, y_t):.6e}")

    # Plots
    os.makedirs(outdir, exist_ok=True)
    plot_fit_progress(
        x_raw=x_raw, y_raw=y_raw,
        x_t=x_t,
        snapshots=snaps, output_dir=outdir, prefix=f"group{group}"
    )
    plot_cost(costs, output_path=os.path.join(outdir, f"group{group}_cost.png"))

    print(f"All done. Plots saved under: {outdir}")


if __name__ == "__main__":
    main()