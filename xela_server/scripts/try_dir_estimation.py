import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def weighted_line_fitting(cx, cy, f, taxels_x, taxels_y):
    # Flatten arrays
    taxels_x_flat = taxels_x.flatten()
    taxels_y_flat = taxels_y.flatten()
    forces_flat = f.flatten()

    # Function to minimize
    def residuals(params):
        m, b = params
        predicted_y = m * taxels_x_flat + b
        distances = np.abs(predicted_y - taxels_y_flat)
        return distances - forces_flat  # The residuals are weighted by force

    # Initial guess for slope (m) and intercept (b)
    initial_guess = [0, 0]

    # Perform least squares fitting
    result = least_squares(residuals, initial_guess)
    m, b = result.x

    return m, b


def plot_results(taxels_x, taxels_y, f, m, b):
    # Plot touch points
    plt.scatter(taxels_x, taxels_y, c=f, s=100, cmap='hot', label='Touch points')

    # Plot the fitted line
    x_vals = np.linspace(taxels_x.min(), taxels_x.max(), 100)
    y_vals = m * x_vals + b
    plt.plot(x_vals, y_vals, 'b-', label='Fitted Line')

    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.title('Fitted Line through Contact Points')
    plt.legend()
    plt.colorbar(label='Force')
    plt.show()


# Example usage
taxels_array_y = np.array([[x for x in range(4)] for _ in range(4)])
taxels_array_x = np.array([[y for x in range(4)] for y in range(4)])
forces_array = np.random.random((4, 4))  # Example forces array

# Compute contact center and weights
cx, cy = 2, 2  # Example contact center

# Fit weighted line
m, b = weighted_line_fitting(cx, cy, forces_array, taxels_array_x, taxels_array_y)

# Plot results
plot_results(taxels_array_x.flatten(), taxels_array_y.flatten(), forces_array.flatten(), m, b)
