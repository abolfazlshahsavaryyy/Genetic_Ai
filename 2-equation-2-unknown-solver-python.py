import numpy as np

# ==== Population Initialization ====

def init_random_population(pop_size, seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y))



def standard_pop(pop,pop_size,eq1,eq2):
    params=[10**x for x in range(-8,9)]
    params+=[5*(10**x) for x in range(-8,9)]
    for i in range(pop_size):
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0]*j,pop[i][1]]))
        
        fitnesses = np.array([fitness_one(np.array([person[0],person[1]]),eq1,eq2) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][0]=sorted_pop[0][0]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1]*j]))
        
        fitnesses = np.array([fitness_one(np.array([person[0],person[1]]),eq1,eq2) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][1]=sorted_pop[0][1]
        neighbor=[]
        

    return pop
# ==== Fitness Functions ====

def fitness_one(individual, eq1, eq2):
    loss1 = abs(eq1['a1'] * individual[0] + eq1['b1'] * individual[1] + eq1['c1'])
    loss2 = abs(eq2['a2'] * individual[0] + eq2['b2'] * individual[1] + eq2['c2'])
    return loss1 + loss2

def fitness_total(pop, eq1, eq2):
    x = pop[:, 0]
    y = pop[:, 1]
    loss1 = np.abs(eq1['a1'] * x + eq1['b1'] * y + eq1['c1'])
    loss2 = np.abs(eq2['a2'] * x + eq2['b2'] * y + eq2['c2'])
    return loss1 + loss2


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
        best_child = children[np.argmin(fitness_vals)]
        new_pop.append(best_child)

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


# ==== Memetic Local Search ====

def memetic(population,equation1,equation2,population_size,power):
    fitnesses = np.array([fitness_one(person, equation1, equation2) for person in population])

    # Get the sorted indices (lowest fitness first)
    sorted_indices = np.argsort(fitnesses)

    # Sort the population accordingly
    sorted_population = population[sorted_indices]

    sorted_population=sorted_population[::-1]
    for i in range(500):
        noise = np.random.uniform(0,power)


        neighbors=[
            np.array([population[population_size-i-1][0]+noise,population[population_size-i-1][1]+noise]),
            np.array([population[population_size-i-1][0]-noise,population[population_size-i-1][1]-noise]),
            np.array([population[population_size-i-1][0]-noise,population[population_size-i-1][1]+noise]),
            np.array([population[population_size-i-1][0]+noise,population[population_size-i-1][1]-noise]),
            np.array([population[population_size-i-1][0]+noise,population[population_size-i-1][1]]),
            np.array([population[population_size-i-1][0]-noise,population[population_size-i-1][1]]),
            np.array([population[population_size-i-1][0],population[population_size-i-1][1]-noise]),
            np.array([population[population_size-i-1][0],population[population_size-i-1][1]+noise]),
        ]

        fitness=[fitness_one(person,equation1,equation2) for person in neighbors]
        index=np.argmin(fitness)
        selected_neighbor=neighbors[index]
        population[i][0]=selected_neighbor[0]
        population[i][1]=selected_neighbor[1]


    return population


# ==== Main Solver ====

def solve_2_equations_2_unknowns(a1, b1, c1, a2, b2, c2):
    eq1 = {'a1': a1, 'b1': b1, 'c1': c1}
    eq2 = {'a2': a2, 'b2': b2, 'c2': c2}

    pop_size = 3000
    generations = 2000

    population = init_random_population(pop_size, seed=42)
    population=standard_pop(population,pop_size,eq1,eq2)

    for gen in range(generations):
        fitness = fitness_total(population, eq1, eq2)

        print(f"Generation {gen+1}:")
        print(f"  ➤ Min Fitness: {np.min(fitness):.6f}")
        print(f"  ➤ Max Fitness: {np.max(fitness):.6f}")

        parents = select_parents_roulette(population, fitness, 2 * pop_size)
        population = crossover(parents, pop_size, seed=42, eq1=eq1, eq2=eq2)
        population = mutate(population, mutation_rate=0.2, mutation_strength=50, seed=gen)

        population = memetic(population, eq1, eq2, pop_size, power= np.min(fitness)* (1 / (gen + 1)**0.5))
        #population=memetic(population,eq1,eq2,pop_size,power= np.min(fitness)* (1 / (gen + 1)))
        #population=memetic(population,eq1,eq2,pop_size,power= np.min(fitness)* (1 / (gen + 1))**1.5)
        #population=memetic(population,eq1,eq2,pop_size,power= np.min(fitness)* (1 / (gen + 1))**2)

        if(np.min(fitness)<1e-4):
            break

    final_fitness = fitness_total(population, eq1, eq2)
    best_idx = np.argmin(final_fitness)
    best_solution = population[best_idx]
    best_fit = final_fitness[best_idx]

    print("\n✅ Best Approximate Solution:")
    print(f"  x = {best_solution[0]:.6f}")
    print(f"  y = {best_solution[1]:.6f}")
    print(f"  Fitness = {best_fit:.6f}")

    return best_solution, best_fit


solve_2_equations_2_unknowns(1,2,-4,4,4,-12)
