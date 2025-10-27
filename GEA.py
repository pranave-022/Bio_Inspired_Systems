import numpy as np
from gplearn.genetic import SymbolicRegressor
import matplotlib.pyplot as plt

# ---------------------------
# Step 1: Simulated Real-World Dataset
# ---------------------------
# Suppose house price depends on: size, rooms, and age
np.random.seed(42)
n_samples = 200

size = np.random.uniform(500, 3000, n_samples)      # square feet
rooms = np.random.randint(1, 6, n_samples)          # number of rooms
age = np.random.uniform(1, 50, n_samples)           # years

# True relationship (unknown to algorithm)
price = 50 * np.sqrt(size) + 10 * rooms - 3 * age + np.random.normal(0, 50, n_samples)

X = np.column_stack((size, rooms, age))
y = price

# ---------------------------
# Step 2: Gene Expression Algorithm (via Symbolic Regressor)
# ---------------------------
model = SymbolicRegressor(
    population_size=1000,
    generations=20,
    stopping_criteria=0.01,
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=1,
    parsimony_coefficient=0.01,
    random_state=42
)

# Train model
model.fit(X, y)

# ---------------------------
# Step 3: Evaluate
# ---------------------------
preds = model.predict(X)
mse = np.mean((y - preds)**2)
print("\n✅ Mean Squared Error:", round(mse, 3))
print("🧬 Evolved Equation:\n", model._program)

# ---------------------------
# Step 4: Visualization
# ---------------------------
plt.figure(figsize=(6,5))
plt.scatter(y, preds, c='blue', s=20, label='Predicted vs Actual')
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Fit')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.legend()
plt.title("GEA (Symbolic Regression) for House Price Prediction")
plt.grid(True)
plt.show()
