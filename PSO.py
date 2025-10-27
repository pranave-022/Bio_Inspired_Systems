import random

# Objective function (maximize)
def objective_function(x):
    return -x**2 + 5*x + 20

# Particle class
class Particle:
    def __init__(self, bounds):
        self.position = random.uniform(bounds[0], bounds[1])   # current position
        self.velocity = 0                                      # initial velocity
        self.pbest = self.position                             # personal best position
        self.pbest_value = objective_function(self.position)   # personal best value

    def update_velocity(self, gbest, w, c1, c2):
        r1, r2 = random.random(), random.random()
        cognitive = c1 * r1 * (self.pbest - self.position)
        social = c2 * r2 * (gbest - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def update_position(self, bounds):
        self.position += self.velocity
        # Keep particle within bounds
        self.position = max(bounds[0], min(bounds[1], self.position))
        # Update personal best if needed
        current_value = objective_function(self.position)
        if current_value > self.pbest_value:
            self.pbest = self.position
            self.pbest_value = current_value
    def __repr__(self):
        return f"Particle(pos={self.position:.4f}, vel={self.velocity:.4f}, pbest={self.pbest:.4f}, pbest_val={self.pbest_value:.4f})"

# PSO Algorithm
def particle_swarm_optimization(bounds, num_particles=10, max_iter=50, w=0.7, c1=1.5, c2=1.5):
    # Initialize swarm
    swarm = [Particle(bounds) for _ in range(num_particles)]
    #print(swarm)
    # Global best initialization
    gbest = max(swarm, key=lambda p: p.pbest_value).pbest

    for _ in range(max_iter):
        for particle in swarm:
            particle.update_velocity(gbest, w, c1, c2)
            particle.update_position(bounds)

        # Update global best
        best_particle = max(swarm, key=lambda p: p.pbest_value)
        if objective_function(best_particle.pbest) > objective_function(gbest):
            gbest = best_particle.pbest

    return gbest, objective_function(gbest)


# Run PSO on f(x) = -x^2 + 5x + 20 within [0, 10]
best_x, best_value = particle_swarm_optimization(bounds=[0, 10])
print("Best solution x =", best_x)
print("Best function value f(x) =", best_value)
