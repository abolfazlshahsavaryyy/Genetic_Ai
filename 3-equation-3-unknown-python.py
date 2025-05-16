import numpy as np

def init(pop_size,seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    z = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y,z))

def standard_pop(pop,pop_size):
    params=[10**x for x in range(-8,9)]
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
    
    res=(6*x) + (-2*y) + (8*z) - 20
    return res
def equation2(x,y,z):

    res=(y) + (8*x) * (z) +1 #𝑦 + 8𝑥 × 𝑧 = −1
    return res

def equation3(x,y,z):

    res = (2*z)*(6/x) + (1.5*y) - 6  
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

def memetic(pop,pop_size,power):
    fitnesses = np.array([fitness_one(person[0],person[1],person[2]) for person in pop])
    sorted_indices = np.argsort(fitnesses)

    # Sort the population accordingly
    sorted_pop = pop[sorted_indices]

    sorted_pop=sorted_pop[::-1]
    noises=[ np.random.uniform(0,power) for _ in range(1)]
    for i in range(500):
        neighbors=[]
        for nois in noises:
            neighbors+=[
               
                np.array([sorted_pop[pop_size-i-1][0]+nois,sorted_pop[pop_size-i-1][1]+nois,sorted_pop[pop_size-i-1][2]+nois]),
                np.array([sorted_pop[pop_size-i-1][0]-nois,sorted_pop[pop_size-i-1][1]+nois,sorted_pop[pop_size-i-1][2]+nois]),
                np.array([sorted_pop[pop_size-i-1][0]+nois,sorted_pop[pop_size-i-1][1]-nois,sorted_pop[pop_size-i-1][2]+nois]),
                np.array([sorted_pop[pop_size-i-1][0]+nois,sorted_pop[pop_size-i-1][1]+nois,sorted_pop[pop_size-i-1][2]-nois]),
                np.array([sorted_pop[pop_size-i-1][0]-nois,sorted_pop[pop_size-i-1][1]-nois,sorted_pop[pop_size-i-1][2]+nois]),
                np.array([sorted_pop[pop_size-i-1][0]-nois,sorted_pop[pop_size-i-1][1]+nois,sorted_pop[pop_size-i-1][2]-nois]),
                np.array([sorted_pop[pop_size-i-1][0]+nois,sorted_pop[pop_size-i-1][1]-nois,sorted_pop[pop_size-i-1][2]-nois]),
                np.array([sorted_pop[pop_size-i-1][0]-nois,sorted_pop[pop_size-i-1][1]-nois,sorted_pop[pop_size-i-1][2]-nois]),
                np.array([sorted_pop[pop_size-i-1][0]-nois,sorted_pop[pop_size-i-1][1],sorted_pop[pop_size-i-1][2]]),
                np.array([sorted_pop[pop_size-i-1][0],sorted_pop[pop_size-i-1][1]-nois,sorted_pop[pop_size-i-1][2]]),
                np.array([sorted_pop[pop_size-i-1][0],sorted_pop[pop_size-i-1][1],sorted_pop[pop_size-i-1][2]-nois]),
                np.array([sorted_pop[pop_size-i-1][0]+nois,sorted_pop[pop_size-i-1][1],sorted_pop[pop_size-i-1][2]]),
                np.array([sorted_pop[pop_size-i-1][0],sorted_pop[pop_size-i-1][1]+nois,sorted_pop[pop_size-i-1][2]]),
                np.array([sorted_pop[pop_size-i-1][0],sorted_pop[pop_size-i-1][1],sorted_pop[pop_size-i-1][2]+nois]),
            ]
        fitnesses=[fitness_one(n[0],n[1],n[2]) for n in neighbors]
        index=np.argmin(fitnesses)
        selected_neigbor=neighbors[index]
        pop[i][0]=selected_neigbor[0]
        pop[i][1]=selected_neigbor[1]
        pop[i][2]=selected_neigbor[2]

    return pop











######################################################################################
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
            if(counter==4):
                nois=True
        parents = select_parents_roulette(pop, fitness, 2 * pop_size)
        pop=crossover(parents,pop_size,seed)
        pop=mutate(pop,seed=seed)
        
        pop=memetic(pop,pop_size,np.min(fitness)*(1/(gen+1)))
        
        if(np.min(fitness)<0.001):
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





solve_3equation_3unknown(3000,2000,42)




#print(fitness_one(4.915641, 4.216009,-0.132744))







