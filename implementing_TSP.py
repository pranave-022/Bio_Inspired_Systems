import math
import random

# Coordinates of cities
cities = [
    (0, 0),  # A
    (2, 3),  # B
    (5, 1),  # C
    (6, 4),  # D
    (8, 0)   # E
]

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def route_distance(route):
    total = 0
    for i in range(len(route)):
        total += distance(cities[route[i]], cities[route[(i+1) % len(route)]])
    return total

def fitness(route):
    return 1 / route_distance(route)   # smaller distance → larger fitness

def selection(population):
    fitness_values = [fitness(route) for route in population]
    total_fitness = sum(fitness_values)
    prob_values = [f / total_fitness for f in fitness_values]
    expected_output = [(f * len(population)) / total_fitness for f in fitness_values]
    actual_count = [round(val) for val in expected_output]

    max_fit = max(fitness_values)
    return prob_values, expected_output, actual_count, max_fit

def crossover(route1, route2):
    # Order crossover (OX)
    size = len(route1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[start:end] = route1[start:end]
    pos = end
    for city in route2:
        if city not in child:
            if pos >= size: pos = 0
            child[pos] = city
            pos += 1
    return child

def mutation(route, mutation_rate=0.1):
    route = route[:]
    for i in range(len(route)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(route)-1)
            route[i], route[j] = route[j], route[i]
    return route

def main():
    print("=== Genetic Algorithm for TSP ===")
    num_chromosomes = 6
    num_cities = len(cities)

    # Initial population (random permutations of cities)
    population = [random.sample(range(num_cities), num_cities) for _ in range(num_chromosomes)]

    max_fitness_old = -1
    generation = 0

    while True:
        generation += 1
        prob_values, expected_output, actual_count, max_fitness = selection(population)
        best_route = max(population, key=fitness)
        best_dist = route_distance(best_route)

        print(f"\nGeneration {generation}: Best Distance = {best_dist:.2f}, Route = {best_route}")

        # Stopping condition
        if abs(max_fitness - max_fitness_old) < 1e-6:
            print("\nBest route stabilized, stopping evolution.")
            break
        max_fitness_old = max_fitness

        # New population using crossover + mutation
        new_population = []
        for _ in range(num_chromosomes // 2):
            parents = random.sample(population, 2)
            child1 = crossover(parents[0], parents[1])
            child2 = crossover(parents[1], parents[0])
            new_population.append(mutation(child1))
            new_population.append(mutation(child2))

        population = new_population

if __name__ == "__main__":
    main()
