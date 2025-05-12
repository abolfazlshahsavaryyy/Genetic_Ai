import numpy as np
import pandas as pd

def init_random_population(population_size,seed=42):
    """
    this function return 2*10000 sample population
    """
    np.random.seed(seed)
    x=np.random.uniform(-1000000,1000000,population_size)
    y=np.random.uniform(-1000000,1000000,population_size)
    population=np.column_stack((x,y))
    return population


def standard_scaled(population:np.ndarray,equation1,equation2):
    

    for i in range(population.shape[0]):
        best_fitness=200000000000
        select_person=np.array([1000,10000])
        for j in range(11):
            value=(int)(np.random.rand()*(10**j))
            person1=np.array([population[i][0]+value,population[i][1]+value])
            person2=np.array([population[i][0]-value,population[i][1]-value])
            person3=np.array([population[i][0]-value,population[i][1]+value])
            person4=np.array([population[i][0]+value,population[i][1]-value])
            fit1=fitness_one(person1,equation1,equation2)
            fit2=fitness_one(person2,equation1,equation2)
            fit3=fitness_one(person3,equation1,equation2)
            fit4=fitness_one(person4,equation1,equation2)
            if(fit1<best_fitness):
                best_fitness=fit1
                select_person[0]=person1[0]
                select_person[1]=person1[1]
            if(fit2<best_fitness):
                best_fitness=fit2
                select_person[0]=person2[0]
                select_person[1]=person2[1]
            if(fit3<best_fitness):
                best_fitness=fit3
                select_person[0]=person3[0]
                select_person[1]=person3[1]
            if(fit4<best_fitness):
                best_fitness=fit4
                select_person[0]=person4[0]
                select_person[1]=person4[1]
        population[i]=np.array([select_person[0],select_person[1]])
    
    return population


def fitness_total(data,equation1:dict,equation2:dict,population_size):
    
    x=data[:,0]
    y=data[:,1]
    

    loss1=np.abs(equation1['a1']*x+equation1['b1']*y+equation1['c1'])
    loss2=np.abs(equation2['a2']*x+equation2['b2']*y+equation2['c2'])
    return loss1+loss2

def fitness_one(person:np.ndarray,equation1:dict,equation2:dict):
    
    loss1=np.abs(equation1['a1']*person[0]+equation1['b1']*person[1]+equation1['c1'])
    loss2=np.abs(equation2['a2']*person[0]+equation2['b2']*person[1]+equation2['c2'])
    return loss1+loss2

def select_parents_roulette(
        population:np.ndarray,
        fitness:np.ndarray,
        num_parent:int
):
    """
    population is (n,2) np array that it the representation of the populatino
    fitness: is (n,) np array that is represent the fitness for each people
    num_parent detect the number of parent that we use
    """
    score=(1/(1+fitness))**1.5
    p=score/np.sum(score)
    
    index_choose=np.random.choice(len(population),size=num_parent,p=p)
    return population[index_choose]


def crossover(parents:np.ndarray,
              population_size:int
              ,seed,equation1,equation2):
    
    """
    input parent : selected parent that has good fitenss 
    population_size: the size of the population
    seed:

    return : new population with Elitism 
    """
    np.random.seed(seed)
    new_population=[]
    num_parents=len(parents)
    for _ in range(population_size):
        index1,index2=np.random.choice(num_parents,2,replace=False)
        parents1=parents[index1]
        parents2=parents[index2]
        children1=np.array([parents1[0],parents2[1]])
        children2=np.array([parents2[0],parents1[1]])
        children3=np.array([parents2[0],parents2[1]])
        children4=np.array([parents1[0],parents1[1]])
        childrens=[children1,children2,children3,children4]
        fitness=[fitness_one(children1,equation1,equation2),
                 fitness_one(children2,equation1,equation2),
                 fitness_one(children3,equation1,equation2),
                 fitness_one(children4,equation1,equation2)]
        index=np.argmin(fitness)
        new_population.append(childrens[index])
    return np.array(new_population)

def mutate(population: np.ndarray, mutation_rate=0.1, mutation_strength=10.0, seed=None):
    """
    Applies random mutations to a portion of the population.

    Parameters:
        population: np.ndarray, shape (n, 2)
        mutation_rate: float, probability of mutating each individual
        mutation_strength: float, the standard deviation of the mutation noise
        seed: int or None, for reproducibility

    Returns:
        Mutated population (np.ndarray)
    """
    if seed is not None:
        np.random.seed(seed)

    # Number of individuals to mutate
    num_to_mutate = int(len(population) * mutation_rate)

    # Randomly select indices to mutate
    mutation_indices = np.random.choice(len(population), size=num_to_mutate, replace=False)

    # Apply mutation: add Gaussian noise
    noise = np.random.normal(0, mutation_strength, size=(num_to_mutate, 2))
    population[mutation_indices] += noise

    return population

def memetic(population,equation1,equation2,population_size,power):
    fitnesses = np.array([fitness_one(person, equation1, equation2) for person in population])

    # Get the sorted indices (lowest fitness first)
    sorted_indices = np.argsort(fitnesses)

    # Sort the population accordingly
    sorted_population = population[sorted_indices]

    sorted_population=sorted_population[::-1]
    for i in range(100):
        noise = np.random.uniform(0,power)

        neighbor1=np.array([population[population_size-i-1][0]+noise,population[population_size-i-1][1]+noise])
        neighbor2=np.array([population[population_size-i-1][0]-noise,population[population_size-i-1][1]-noise])
        neighbor3=np.array([population[population_size-i-1][0]-noise,population[population_size-i-1][1]+noise])
        neighbor4=np.array([population[population_size-i-1][0]+noise,population[population_size-i-1][1]-noise])
        neighbors=[neighbor1,neighbor2,neighbor3,neighbor4]
        fitness=[fitness_one(neighbor1,equation1,equation2),
                 fitness_one(neighbor2,equation1,equation2),
                 fitness_one(neighbor3,equation1,equation2),
                 fitness_one(neighbor4,equation1,equation2)]
        index=np.argmin(fitness)
        selected_neighbor=neighbors[index]
        population[i][0]=selected_neighbor[0]
        population[i][1]=selected_neighbor[1]


    return population
    pass

def solve_2_quation_2_umknown(a1,b1,c1,a2,b2,c2):
    """
    this function get 2 equation like this:
    a1x+b1y+c1=0 , a2x+b2y+c2=0
    


    """
    
    equation1={
        'a1':a1,'b1':b1,'c1':c1
    }
    equation2={
        'a2':a2,'b2':b2,'c2':c2
    }
    population_size=3000
    population=init_random_population(population_size,seed=42) # create random 2*10000 population
    #population=standard_scaled(population,equation1,equation2)
    for i in range(2000):
        fitness=fitness_total(population,equation1,equation2,population_size) # compute the fitness of all population
        print(f"Min Fitness {i+1}: {np.min(fitness)}")
        print(f"Max Fitness {i+1}: {np.max(fitness)}")
        selected_parents=select_parents_roulette(population,fitness,2*population_size) # chose random base probability parent for the next generation
        population=crossover(selected_parents,population_size,42,equation1,equation2)
        population = mutate(population, mutation_rate=0.1, mutation_strength=50, seed=i)
        population=memetic(population,equation1,equation2,population_size,100*(1/(i+1)))
       
    return fitness,population 

    
    




solve_2_quation_2_umknown(30,40,10,10,10,0)