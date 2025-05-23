import numpy as np

def init(pop_size,seed=42):
    np.random.seed(seed)
    x = np.random.uniform(-1e9, 1e9, pop_size)
    y = np.random.uniform(-1e9, 1e9, pop_size)
    return np.column_stack((x, y))

def standard_pop(pop,pop_size,a1,b1,c1,a2,b2,c2):
    params=[10**x for x in range(-8,9)]
    
    for i in range(pop_size):
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0]*j,pop[i][1]]))
        
        
        fitnesses = np.array([fitness_one(person[0],person[1],a1,b1,c1,a2,b2,c2) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][0]=sorted_pop[0][0]
        neighbor=[]
        for j in params:
            neighbor.append(np.array([pop[i][0],pop[i][1]*j]))
        
        fitnesses = np.array([fitness_one(person[0],person[1],a1,b1,c1,a2,b2,c2) for person in neighbor])
        sorted_indices = np.argsort(fitnesses)

        sorted_pop = [neighbor[idx] for idx in sorted_indices]
        pop[i][1]=sorted_pop[0][1]
        

    return pop

        
def equation1(x,y,a,b,c):
    
    res=a*x + b*y +c
    return res
def equation2(x,y,a,b,c):

    res=a*x + b*y +c
    return res
def fitness_one(x,y,a1,b1,c1,a2,b2,c2):
    loss=np.abs(equation1(x,y,a1,b1,c1))
    loss+=np.abs(equation2(x,y,a2,b2,c2))
    
    return loss

def fitness_total(pop,pop_size,a1,b1,c1,a2,b2,c2):
    fitness=[fitness_one(person[0],person[1],a1,b1,c1,a2,b2,c2) for person in pop]
    return np.array(fitness)

def select_parents_roulette(pop, fitness, num_parents):
    score = (1 / (1 + fitness))
    probs = score / np.sum(score)
    indices = np.random.choice(len(pop), size=num_parents, p=probs)
    return pop[indices]



def crossover(parents, pop_size, seed,a1,b1,c1,a2,b2,c2):
    np.random.seed(seed)
    new_pop = []
    num_parents = len(parents)

    for _ in range(pop_size):
        i1, i2 = np.random.choice(num_parents, 2, replace=False)
        p1, p2 = parents[i1], parents[i2]

        
        children = [
            np.array([p1[0], p1[1]]),
            np.array([p2[0], p1[1]]),
            np.array([p1[0], p2[1]]),
            np.array([p2[0], p2[1]]),
            np.array([p2[0]*0.5+p1[0]*0.5, p2[1]*0.5+p1[1]*0.5])
        ]

        
        for _ in range(5):
            alpha = np.random.rand()
            child = np.array([
                alpha * p1[0] + (1 - alpha) * p2[0],
                alpha * p1[1] + (1 - alpha) * p2[1]
            ])
            children.append(child)

        
        fitness_vals = [fitness_one(c[0], c[1],a1,b1,c1,a2,b2,c2) for c in children]
        best_child = children[np.argmin(fitness_vals)]
        new_pop.append(best_child)

    return np.array(new_pop)


def mutate(pop, mutation_rate=0.1, mutation_strength=10.0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_mutations = int(len(pop) * mutation_rate)
    indices = np.random.choice(len(pop), size=num_mutations, replace=False)
    noise = np.random.normal(0, mutation_strength, size=(num_mutations, 2))
    pop[indices] += noise

    return pop




######################################################################################
def solve_3equation_3unknown(pop_size,generation,seed,a1,b1,c1,a2,b2,c2):
    pop=init(pop_size,seed)
    pop=standard_pop(pop,pop_size,a1,b1,c1,a2,b2,c2)
    counter=0
    fitness_before=0
    min_fit=0
    for gen in range(generation):
        
        fitness=fitness_total(pop,pop_size,a1,b1,c1,a2,b2,c2)
        fitness_before=min_fit
        min_fit=np.min(fitness)
        if(gen%10==0):
            print(f"Generation {gen+1}:")
            print(f"  ➤ Min Fitness: {min_fit:.16f}")
            print(f"  ➤ Max Fitness: {np.max(fitness):.16f}")
        if(np.abs(min_fit-fitness_before)!=0):
            fitness_before=min_fit
            counter=0
        else:
            counter+=1
                
        parents = select_parents_roulette(pop, fitness, 2 * pop_size)
        pop=crossover(parents,pop_size,seed,a1,b1,c1,a2,b2,c2)
        pop=mutate(pop,seed=seed,mutation_rate=0.2,mutation_strength=min_fit/5)
        if counter == 3:
            print("⚠️  Stagnation detected. Injecting noise.")
            pop = mutate(pop, mutation_rate=1, mutation_strength=min_fit/5)
            counter = 0

        
        if(np.min(fitness)<1e-18):
            break

    final_fitness = fitness_total(pop,pop_size,a1,b1,c1,a2,b2,c2)
    best_idx = np.argmin(final_fitness)
    best_solution = pop[best_idx]
    best_fit = final_fitness[best_idx]

    print("\n✅ Best Approximate Solution:")
    print(f"  x = {best_solution[0]:.6f}")
    print(f"  y = {best_solution[1]:.6f}")
    print(f"last gen: {gen+1}")
    print(f"  Fitness = {best_fit:.6f}")

print("___________________________")
print("a1*x+b1*y+c1=0")
print("a2*x+b2*y+c2=0")
print("___________________________")
a1=int(input("Enter a1 :"))
b1=int(input("Enter b1 :"))
c1=int(input("Enter c1 :"))
a2=int(input("Enter a2 :"))
b2=int(input("Enter b2 :"))
c2=int(input("Enter c2 :"))
solve_3equation_3unknown(500,10000,43,a1,b1,c1,a2,b2,c2)



