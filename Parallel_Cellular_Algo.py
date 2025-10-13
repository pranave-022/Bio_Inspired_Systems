import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------
# Step 1: Define the Objective Function
# --------------------------------------
def objective_function(x, y):
    """We aim to minimize f(x, y) = x^2 + y^2."""
    return x**2 + y**2

# --------------------------------------
# Step 2: Initialize Parameters
# --------------------------------------
grid_size = 20          # Grid dimension (20x20)
iterations = 100        # Number of iterations
alpha = 0.3             # Movement coefficient (learning rate)
search_range = 5.0      # Range for random initialization (-5 to +5)

# Each cell has (x, y) coordinates randomly placed
x_grid = np.random.uniform(-search_range, search_range, (grid_size, grid_size))
y_grid = np.random.uniform(-search_range, search_range, (grid_size, grid_size))

# --------------------------------------
# Step 3: Define Neighborhood Function
# --------------------------------------
def get_neighbors(i, j):
    """Return the valid neighbors (Moore neighborhood)."""
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue  # skip itself
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size and 0 <= nj < grid_size:
                neighbors.append((ni, nj))
    return neighbors

# --------------------------------------
# Step 4: Optimization Loop
# --------------------------------------
for t in range(iterations):
    # Compute fitness for all cells (lower f = better)
    f_values = objective_function(x_grid, y_grid)
    fitness = -f_values  # convert to maximization (higher fitness = better)

    # Prepare new grids for updates
    new_x = np.copy(x_grid)
    new_y = np.copy(y_grid)

    # Update each cell in parallel (conceptually)
    for i in range(grid_size):
        for j in range(grid_size):
            # Get all neighbors of current cell
            neighbors = get_neighbors(i, j)
            # Find neighbor with highest fitness
            best_neighbor = max(neighbors, key=lambda n: fitness[n])
            bx = x_grid[best_neighbor]
            by = y_grid[best_neighbor]
            # Update the cell slightly toward the best neighbor
            new_x[i, j] = x_grid[i, j] + alpha * (bx - x_grid[i, j])
            new_y[i, j] = y_grid[i, j] + alpha * (by - y_grid[i, j])

    # Update the grid for the next iteration
    x_grid, y_grid = new_x, new_y

    # Track progress
    best_val = np.min(objective_function(x_grid, y_grid))
    if t % 10 == 0 or t == iterations - 1:
        print(f"Iteration {t+1:3d} → Best f(x,y) = {best_val:.6f}")

# --------------------------------------
# Step 5: Find and Display Final Result
# --------------------------------------
best_i, best_j = np.unravel_index(np.argmin(objective_function(x_grid, y_grid)), x_grid.shape)
best_x = x_grid[best_i, best_j]
best_y = y_grid[best_i, best_j]
best_f = objective_function(best_x, best_y)

print("\n✅ Final Best Solution:")
print(f"x = {best_x:.6f}, y = {best_y:.6f}, f(x, y) = {best_f:.6f}")

# --------------------------------------
# Step 6: Visualization
# --------------------------------------
plt.figure(figsize=(6,5))
plt.scatter(x_grid, y_grid, c='blue', s=15)
plt.title("Final Cell Positions (Convergence toward 0,0)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
