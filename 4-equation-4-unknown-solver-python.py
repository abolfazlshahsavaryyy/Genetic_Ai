import numpy as np

def init(pop_size,seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    z = np.random.uniform(-1e9, 1e9, pop_size)
    t = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y,z,t))

def standard_pop(pop,pop_size):
    params=[10**x for x in range(-8,9)]
    
    for i in range(pop_size):
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0]*j,pop[i][1],pop[i][2],pop[i][3]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][0]=sorted_pop[0][0]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1]*j,pop[i][2],pop[i][3]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][1]=sorted_pop[0][1]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1],pop[i][2]*j,pop[i][3]]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][2]=sorted_pop[0][2]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1],pop[i][2],pop[i][3]*j]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],person[2],person[3]) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][3]=sorted_pop[0][3]

    return pop


        



def equation1(x,y,z,t):
    
    res=(1/15)*x+(-2)*y+(-15)*z+(-4/5)*t-30000000
    return res
def equation2(x,y,z,t):

    res=(-2.5)*x+(-9/4)*y+(12)*z+(-1)*t-170000000
    return res

def equation3(x,y,z,t):

    res = (-13)*x+(0.3)*y+(-6)*z+(-2/5)*t-170000000
    return res


def equation4(x,y,z,t):

    res = (1/2)*x+(2)*y+(7/4)*z+(4/3)*t+90000000
    return res

def fitness_one(x,y,z,t):
    loss=np.abs(equation1(x,y ,z,t))
    loss+=np.abs(equation2(x,y,z,t))
    loss+=np.abs(equation3(x,y,z,t))
    return loss

def fitness_total(pop,pop_size):
    fitness=[fitness_one(person[0],person[1],person[2],person[3]) for person in pop]
    return np.array(fitness)

def select_parents_roulette(pop, fitness, num_parents):
    score = (1 / (1 + fitness)) 
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
            np.array([p1[0], p1[1],p1[2],p1[3]]),
            np.array([p2[0], p1[1],p1[2],p1[3]]),
            np.array([p1[0], p2[1],p1[2],p1[3]]),
            np.array([p1[0], p1[1],p2[2],p1[3]]),
            np.array([p2[0], p2[1],p1[2],p1[3]]),
            np.array([p1[0], p2[1],p2[2],p1[3]]),
            np.array([p2[0], p1[1],p2[2],p1[3]]),
            np.array([p2[0], p2[1],p2[2],p1[3]]),
            np.array([p1[0], p1[1],p1[2],p2[3]]),
            np.array([p2[0], p1[1],p1[2],p2[3]]),
            np.array([p1[0], p2[1],p1[2],p2[3]]),
            np.array([p1[0], p1[1],p2[2],p2[3]]),
            np.array([p2[0], p2[1],p1[2],p2[3]]),
            np.array([p1[0], p2[1],p2[2],p2[3]]),
            np.array([p2[0], p1[1],p2[2],p2[3]]),
            np.array([p2[0], p2[1],p2[2],p2[3]]),
            np.array([(p2[0]+p1[0])/2, (p1[1]+p2[1])/2,(p2[2]+p1[2])/2,(p1[3]+p2[3])/2]),
        ]
        for _ in range(50):
            alpha = np.random.rand()
            child = np.array([
                alpha * p1[0] + (1 - alpha) * p2[0],
                alpha * p1[1] + (1 - alpha) * p2[1],
                alpha * p1[2] + (1 - alpha) * p2[2],
                alpha * p1[3] + (1 - alpha) * p2[3],
            ])
            children.append(child)
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

######################################################################################
def solve_4equation_4unknown(pop_size,generation,seed):
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
        print(f"  ➤ Min Fitness: {min_fit:.11f}")
        print(f"  ➤ Max Fitness: {np.max(fitness):.11f}")
        if(np.abs(min_fit-fitness_before)>min_fit/200):
            fitness_before=min_fit
            counter=0
        else:
            counter+=1
                
        parents = select_parents_roulette(pop, fitness, 2 * pop_size)
        pop=crossover(parents,pop_size,seed)
        pop=mutate(pop,seed=seed,mutation_strength=min_fit/5,mutation_rate=0.2)
        if counter == 3:
            print("⚠️  Stagnation detected. Injecting noise.")
            pop = mutate(pop, mutation_rate=1, mutation_strength=min_fit/10)
            counter = 0
        if(np.min(fitness)<1e-10):
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


solve_4equation_4unknown(1000,2000,3)
