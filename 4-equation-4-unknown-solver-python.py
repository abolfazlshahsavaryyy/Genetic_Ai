import numpy as np
import pandas as pd

def init(pop_size,seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    z = np.random.uniform(-1e9, 1e9, pop_size)
    t = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y,z,t))

def standard_pop(pop,pop_size):
    #init the paramether from 10^-8 to 10^8
    params=[10**x for x in range(-8,9)]
    for i in range(pop_size):
        
        ################################################
        neighbor=[]
        for j in params:
            #add all neighbor by multiply one parameter to the coef
            neighbor.append(np.array([pop[i][0]*j,pop[i][1],pop[i][2],pop[i][3]]))
        
        #compute the fitness for all neigbor
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        #sort baced on the fitness
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        # change the current value of this param to the better one
        pop[i][0]=sorted_pop[0][0]

        #########################################################


        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1]*j,pop[i][2],pop[i][3]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][1]=sorted_pop[0][1]

        ################################################


        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1],pop[i][2]*j,pop[i][3]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][2]=sorted_pop[0][2]

        ####################################################


        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1],pop[i][2],pop[i][3]*j]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][3]=sorted_pop[0][3]



    return pop

def equation1(x,y,z,t):
    
    res=(1/15)*x+(-2)*y+(-15)*z+(-4/5)*t-3
    return res
def equation2(x,y,z,t):

    res=(-2.5)*x+(-9/4)*y+(12)*z+(-1)*t-17
    return res

def equation3(x,y,z,t):

    res = (-13)*x+(0.3)*y+(-6)*z+(-2/5)*t-17  
    return res


def equation4(x,y,z,t):

    res = (1/2)*x+(2)*y+(7/4)*z+(4/3)*t+9  
    return res

def fitness_one(x,y,z,t):
    loss=np.abs(equation1(x,y,z,t))
    loss+=np.abs(equation2(x,y,z,t))
    loss+=np.abs(equation3(x,y,z,t))
    loss+=np.abs(equation4(x,y,z,t))
    return loss

def fitness_total(pop,pop_size):
    fitness=[fitness_one(person[0],person[1],person[2],person[3]) for person in pop]
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
            np.array([p1[0], p1[1],p1[2],p1[3]]), np.array([p1[0], p1[1],p1[2],p2[3]]), 
            np.array([p2[0], p1[1],p1[2],p1[3]]), np.array([p2[0], p1[1],p1[2],p2[3]]), 
            np.array([p1[0], p2[1],p1[2],p1[3]]), np.array([p1[0], p2[1],p1[2],p2[3]]), 
            np.array([p1[0], p1[1],p2[2],p1[3]]), np.array([p1[0], p1[1],p2[2],p2[3]]), 
            np.array([p2[0], p2[1],p1[2],p1[3]]), np.array([p2[0], p2[1],p1[2],p2[3]]), 
            np.array([p1[0], p2[1],p2[2],p1[3]]), np.array([p1[0], p2[1],p2[2],p2[3]]), 
            np.array([p2[0], p1[1],p2[2],p1[3]]), np.array([p2[0], p1[1],p2[2],p2[3]]), 
            np.array([p2[0], p2[1],p2[2],p1[3]]), np.array([p2[0], p2[1],p2[2],p2[3]])
        ]
        fitness_vals = [fitness_one(c[0],c[1],c[2],c[3]) for c in children]
        best_child = children[np.argmin(fitness_vals)]
        new_pop.append(best_child)

    return np.array(new_pop) 

def mutate(pop, mutation_rate=0.1, mutation_strength=10.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_mutations = int(len(pop) * mutation_rate)
    indices = np.random.choice(len(pop), size=num_mutations, replace=False)
    noise = np.random.normal(0, mutation_strength, size=(num_mutations, 4))
    pop[indices] += noise

    return pop 

import numpy as np
import itertools

def memetic(pop, pop_size, power):
    # Compute fitness for each individual
    fitnesses = np.array([fitness_one(*person) for person in pop])

    # Sort the population by fitness in descending order (best first)
    sorted_indices = np.argsort(fitnesses)[::-1]
    sorted_pop = pop[sorted_indices]

    # Generate all 81 directions for 4D where each dimension can be -1, 0, or 1
    directions = np.array(list(itertools.product([-1, 0, 1], repeat=4)))

    # Remove the zero vector (no movement)
    directions = directions[~np.all(directions == 0, axis=1)]

    # Scale all directions by the same noise power
    directions = directions * power

    # Perform local search for top individuals (up to 500 or pop_size)
    for i in range(min(500, pop_size)):
        base = sorted_pop[pop_size - i - 1]  # Choose from worst to best

        # Generate neighbors by adding all direction vectors
        neighbors = base + directions

        # Evaluate neighbors
        neighbor_fitnesses = np.array([fitness_one(*n) for n in neighbors])

        # Select the best neighbor (lowest fitness)
        best_neighbor = neighbors[np.argmin(neighbor_fitnesses)]

        # Update individual in the population
        pop[i] = best_neighbor

    return pop


def solve_3equation_3unknown(pop_size,generation,seed):
    pop=init(pop_size,seed)
    pop=standard_pop(pop,pop_size)
    counter=0
    fitness_before=0
    nois=False
    for gen in range(generation):
        
        fitness=fitness_total(pop,pop_size)
        min_fit=np.min(fitness)
        print(f"Generation {gen+1}:")
        print(f"  ➤ Min Fitness: {min_fit:.6f}")
        print(f"  ➤ Max Fitness: {np.max(fitness):.6f}")
        if(min_fit!=fitness_before):
            fitness_before=min_fit
            counter=0
        else:
            counter+=1
            if(counter==2):
                nois=True
        parents = select_parents_roulette(pop, fitness, 2 * pop_size)
        pop=crossover(parents,pop_size,seed)
        pop=mutate(pop,seed=seed)
        
        pop=memetic(pop,pop_size,np.min(fitness))
        if counter == 2:
            print("⚠️  Stagnation detected. Injecting noise.")
            pop = mutate(pop, mutation_rate=1, mutation_strength=1)
            counter = 0
        
        if(np.min(fitness)<0.01):
            break

    final_fitness = fitness_total(pop,pop_size)
    best_idx = np.argmin(final_fitness)
    best_solution = pop[best_idx]
    best_fit = final_fitness[best_idx]

    print("\n✅ Best Approximate Solution:")
    print(f"  x = {best_solution[0]:.6f}")
    print(f"  y = {best_solution[1]:.6f}")
    print(f"  z = {best_solution[2]:.6f}")
    print(f"  t = {best_solution[3]:.6f}")
    print(f"  Fitness = {best_fit:.6f}")





solve_3equation_3unknown(3000,2000,34)