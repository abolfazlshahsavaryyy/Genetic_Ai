import numpy as np

def init(pop_size,seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    z = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y,z))

def standard_pop(pop,pop_size):
    params=[10**x for x in range(-8,9)]
    params+=[3**x for x in range(-8,9)]
    params+=[6**x for x in range(-8,9)]
    for i in range(pop_size):
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0]*j,pop[i][1],pop[i][2]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][0]=sorted_pop[0][0]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1]*j,pop[i][2]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][1]=sorted_pop[0][1]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1],pop[i][2]*j]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][2]=sorted_pop[0][2]

    return pop


        



        
def equation1(x,y,z):
    
    res=(6*x) + (-2*y) + (8*z) - 200000
    return res
def equation2(x,y,z):

    res=(y) + (8*x) * (z) +10000 #𝑦 + 8𝑥 × 𝑧 = −1
    return res

def equation3(x,y,z):

    res = (2*z)*(6/x) + (1.5*y) - 60000  
    return res

def fitness_one(x,y,z):
    loss=np.abs(equation1(x,y,z))
    loss+=np.abs(equation2(x,y,z))
    loss+=np.abs(equation3(x,y,z))
    return loss

def fitness_total(pop,pop_size):
    fitness=[fitness_one(person[0],person[1],person[2]) for person in pop]
    return np.array(fitness)

def select_parents_roulette(pop, fitness, num_parents):
    score = (1 / (1 + fitness)) ** 1.5
    probs = score / np.sum(score)
    indices = np.random.choice(len(pop), size=num_parents, p=probs)
    return pop[indices]


def crossover(parents, pop_size, seed):
    np.random.seed(seed)
    new_pop = []
    num_parents = len(parents)

    for _ in range(pop_size):
        i1, i2 = np.random.choice(num_parents, 2, replace=False)
        p1, p2 = parents[i1], parents[i2]

        children = [
            np.array([p1[0], p1[1],p1[2]]),
            np.array([p2[0], p1[1],p1[2]]),
            np.array([p1[0], p2[1],p1[2]]),
            np.array([p1[0], p1[1],p2[2]]),
            np.array([p2[0], p2[1],p1[2]]),
            np.array([p1[0], p2[1],p2[2]]),
            np.array([p2[0], p1[1],p2[2]]),
            np.array([p2[0], p2[1],p2[2]]),
        ]
        fitness_vals = [fitness_one(c[0],c[1],c[2]) for c in children]
        best_child = children[np.argmin(fitness_vals)]
        new_pop.append(best_child)

    return np.array(new_pop)   


def mutate(pop, mutation_rate=0.1, mutation_strength=10.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_mutations = int(len(pop) * mutation_rate)
    indices = np.random.choice(len(pop), size=num_mutations, replace=False)
    noise = np.random.normal(0, mutation_strength, size=(num_mutations, 3))
    pop[indices] += noise

    return pop

import numpy as np

def memetic(pop, pop_size, power):
    # Evaluate fitness for each individual in the population
    fitnesses = np.array([fitness_one(*person) for person in pop])
    
    # Sort the population by fitness (descending order)
    sorted_indices = np.argsort(fitnesses)[::-1]
    sorted_pop = pop[sorted_indices]

    # Generate a single noise value (based on original code)
    noise = np.random.uniform(0, power)

    # Define relative directions for neighbor generation
    directions = np.array([
        [ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1, -1],
        [-1, -1,  1], [-1,  1, -1], [ 1, -1, -1], [-1, -1, -1],
        [-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1],
        [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]
    ]) * noise

    # Apply local search (memetic refinement)
    for i in range(min(500, pop_size)):
        individual = sorted_pop[pop_size - i - 1]
        neighbors = individual + directions

        # Evaluate all neighbors
        neighbor_fitnesses = np.array([fitness_one(*n) for n in neighbors])
        best_neighbor = neighbors[np.argmin(neighbor_fitnesses)]

        # Update individual in the population
        pop[i] = best_neighbor

    return pop



def mean_coef(pop):
    return np.mean(pop)








######################################################################################
def solve_3equation_3unknown(pop_size,generation,seed):
    pop=init(pop_size,seed)
    pop=standard_pop(pop,pop_size)
    counter=0
    fitness_before=0
    min_fit=0
    for gen in range(generation):
        
        fitness=fitness_total(pop,pop_size)
        fitness_before=min_fit
        min_fit=np.min(fitness)
        print(f"Generation {gen+1}:")
        print(f"  ➤ Min Fitness: {min_fit:.6f}")
        print(f"  ➤ Max Fitness: {np.max(fitness):.6f}")
        if(np.abs(min_fit-fitness_before)>min_fit/200):
            fitness_before=min_fit
            counter=0
        else:
            counter+=1
                
        parents = select_parents_roulette(pop, fitness, 2 * pop_size)
        pop=crossover(parents,pop_size,seed)
        pop=mutate(pop,seed=seed)
        coef=mean_coef(pop)
        pop=memetic(pop,pop_size,min_fit*(1/(gen+1)))
        if counter == 6:
            print("⚠️  Stagnation detected. Injecting noise.")
            pop = mutate(pop, mutation_rate=1, mutation_strength=min_fit/10)
            counter = 0
        if(np.min(fitness)<0.0001):
            break

    final_fitness = fitness_total(pop,pop_size)
    best_idx = np.argmin(final_fitness)
    best_solution = pop[best_idx]
    best_fit = final_fitness[best_idx]

    print("\n✅ Best Approximate Solution:")
    print(f"  x = {best_solution[0]:.6f}")
    print(f"  y = {best_solution[1]:.6f}")
    print(f"  z = {best_solution[2]:.6f}")
    print(f"  Fitness = {best_fit:.6f}")





solve_3equation_3unknown(3000,2000,12)




#print(fitness_one(4.915641, 4.216009,-0.132744))







