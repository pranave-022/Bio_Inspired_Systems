import numpy as np
import math

# Objective function (example: f(x) = x^2)
def f(x):
    return x**2

# Levy flight step generator (Mantegna’s algorithm)
def levy_flight(lam=1.5):
    sigma_u = (math.gamma(1+lam) * math.sin(math.pi*lam/2) /
              (math.gamma((1+lam)/2) * lam * 2**((lam-1)/2)))**(1/lam)
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step = u / (abs(v)**(1/lam))
    return step

# Cuckoo Search Algorithm
def cuckoo_search(n=10, pa=0.25, alpha=1.0, lam=1.5, max_iter=100):
    # Step 1: Initialize nests randomly in search space [-10, 10]
    nests = np.random.uniform(-10, 10, size=n)
    fitness = np.array([f(x) for x in nests])

    # Best solution
    best_idx = np.argmin(fitness)
    best = nests[best_idx]

    for t in range(max_iter):
        for i in range(n):
            # Step 2: Generate new solution by Levy flight
            step = alpha * levy_flight(lam)
            new_x = nests[i] + step
            new_fit = f(new_x)

            # Step 3: If better, replace
            if new_fit < fitness[i]:
                nests[i] = new_x
                fitness[i] = new_fit

        # Step 4: Abandon some nests with probability pa
        for i in range(n):
            if np.random.rand() < pa:
                nests[i] = np.random.uniform(-10, 10)
                fitness[i] = f(nests[i])

        # Update best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < f(best):
            best = nests[best_idx]

        # Print progress
        print(f"Iteration {t+1}: Best = {best:.6f}, f(x) = {f(best):.6f}")

    return best, f(best)

# Run demo
if __name__ == "__main__":
    best_x, best_val = cuckoo_search(n=10, pa=0.25, alpha=0.5, lam=1.5, max_iter=50)
    print("\n=== Final Result ===")
    print(f"Best solution x = {best_x:.6f}, f(x) = {best_val:.6f}")
