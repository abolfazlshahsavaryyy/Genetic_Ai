import numpy as np

# ==== Population Initialization ====
def init_random_population(pop_size, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e6, 1e6, pop_size)
    y = np.random.uniform(-1e6, 1e6, pop_size)
    return np.column_stack((x, y))


# ==== Fitness Functions ====
def fitness_one(ind, eq1, eq2):
    return abs(eq1['a1'] * ind[0] + eq1['b1'] * ind[1] + eq1['c1']) + \
           abs(eq2['a2'] * ind[0] + eq2['b2'] * ind[1] + eq2['c2'])

def fitness_total(pop, eq1, eq2):
    x = pop[:, 0]
    y = pop[:, 1]
    return np.abs(eq1['a1'] * x + eq1['b1'] * y + eq1['c1']) + \
           np.abs(eq2['a2'] * x + eq2['b2'] * y + eq2['c2'])


# ==== Selection ====
def select_parents_roulette(pop, fitness, num_parents):
    score = (1 / (1 + fitness)) ** 1.5
    probs = score / np.sum(score)
    indices = np.random.choice(len(pop), size=num_parents, p=probs)
    return pop[indices]


# ==== Crossover ====
def crossover(parents, pop_size, seed, eq1, eq2):
    np.random.seed(seed)
    new_pop = []
    num_parents = len(parents)

    for _ in range(pop_size):
        i1, i2 = np.random.choice(num_parents, 2, replace=False)
        p1, p2 = parents[i1], parents[i2]

        children = [
            np.array([p1[0], p2[1]]),
            np.array([p2[0], p1[1]]),
            np.array([p2[0], p2[1]]),
            np.array([p1[0], p1[1]])
        ]
        fitness_vals = [fitness_one(c, eq1, eq2) for c in children]
        new_pop.append(children[np.argmin(fitness_vals)])

    return np.array(new_pop)


# ==== Mutation ====
def mutate(pop, mutation_rate=0.1, mutation_strength=10.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_mutations = int(len(pop) * mutation_rate)
    indices = np.random.choice(len(pop), size=num_mutations, replace=False)
    noise = np.random.normal(0, mutation_strength, size=(num_mutations, 2))
    pop[indices] += noise

    return pop


# ==== Memetic Local Search (Optimized) ====
def memetic(pop, eq1, eq2, power, top_k=20):
    # Work only on top_k worst individuals
    fitnesses = np.array([fitness_one(ind, eq1, eq2) for ind in pop])
    worst_indices = np.argsort(fitnesses)[-top_k:]

    for i, idx in enumerate(worst_indices):
        base = pop[idx]
        noise = np.random.uniform(0, power)

        neighbors = np.array([
            base + [noise, noise],
            base + [-noise, -noise],
            base + [-noise, noise],
            base + [noise, -noise]
        ])
        fitness_vals = np.array([fitness_one(n, eq1, eq2) for n in neighbors])
        pop[idx] = neighbors[np.argmin(fitness_vals)]

    return pop


# ==== Main Solver ====
def solve_2_equations_2_unknowns(a1, b1, c1, a2, b2, c2):
    eq1 = {'a1': a1, 'b1': b1, 'c1': c1}
    eq2 = {'a2': a2, 'b2': b2, 'c2': c2}

    pop_size = 1000     # Reduced from 3000
    generations = 500   # Reduced from 2000

    population = init_random_population(pop_size, seed=42)

    for gen in range(generations):
        fitness = fitness_total(population, eq1, eq2)

        print(f"Generation {gen+1}:")
        print(f"  ➤ Min Fitness: {np.min(fitness):.6f}")
        print(f"  ➤ Max Fitness: {np.max(fitness):.6f}")

        parents = select_parents_roulette(population, fitness, 2 * pop_size)
        population = crossover(parents, pop_size, seed=42, eq1=eq1, eq2=eq2)
        population = mutate(population, mutation_rate=0.1, mutation_strength=25, seed=gen)
        population = memetic(population, eq1, eq2, power=50 * (1 / (gen + 1)), top_k=20)

    final_fitness = fitness_total(population, eq1, eq2)
    best_idx = np.argmin(final_fitness)
    best_solution = population[best_idx]
    best_fit = final_fitness[best_idx]

    print("\n✅ Best Approximate Solution:")
    print(f"  x = {best_solution[0]:.6f}")
    print(f"  y = {best_solution[1]:.6f}")
    print(f"  Fitness = {best_fit:.6f}")

    return best_solution, best_fit


# === Run Example ===
solve_2_equations_2_unknowns(30, 40, 10, 10, 10, 0)
