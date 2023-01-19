import random
import math
from numpy.random import choice

# global variables.
tournament_selection_size = 4 # The number of random chromosomes to compete to select 2 parents from them.
steady_state_selection_rate = 0.6 # The precentage of entire population
generations_number = 50 # The number of generations to run the genetic algorithm.

# Tournament Selection.
def tournamentSelection(population):
    global tournament_selection_size
    parent_chromosome1 = sorted( # First parent.
                    random.choices(population, k=tournament_selection_size)
                )[0] # Selects k random paths, sorts them, and choose the shortest one.
    parent_chromosome2 = sorted( # Second parent.
                    random.choices(population, k=tournament_selection_size)
                )[0] # Selects k random paths, sorts them, and choose the shortest one.
    return parent_chromosome1, parent_chromosome2


# Rank Selection.
def rankSelection(population):
    sorted_population = sorted(population)
    ranked_population = []
    
    # Add rank to each chromosome.
    # The fittest gets the highest rank.
    # That because will make its probability the highest.
    sum_of_probablities = 0
    rank = len(population)
    for i in range(len(population)):
        ranked_population.append([rank, sorted_population[i]])
        sum_of_probablities += rank
        rank -= 1
    
    # Computes for each chromosome the probability.
    chromosome_probabilities = []
    chromosomes = list(range(len(population))) # A list of the chromosomes' indexes.
    for i in range(len(population)):
        chromosome_probability = ranked_population[i][0]/sum_of_probablities # Calculate each chromosome's probablity.
        chromosome_probabilities.append(chromosome_probability)
    
    # Selects two chromosomes based on the computed probabilities.
    # This NumPy's "choice" function that supports probability distributions.
    # choice(list_of_candidates, number_of_items_to_pick, replace=False, p=probability_distribution)
    # "replace=False" to change the behavior so that drawn items are not replaced,
    # Default is True, meaning that a value of "a" can be selected multiple times.
    chromosome1_index, chromosome2_index = choice(chromosomes, 2, replace=False, p=chromosome_probabilities)
    parent_chromosome1 = sorted_population[int(chromosome1_index)]
    parent_chromosome2 = sorted_population[int(chromosome2_index)]
    
    return parent_chromosome1, parent_chromosome2 


# Inversion Mutation.
def inversionMutation(child_chromosome):
    point = random.randint(1, len(child_chromosome)) # Select random index.
    child_chromosome[1:point] = reversed(child_chromosome[1:point]) # Reverse the targets from the beginning of the chromosome to the selected index.
    child_chromosome[point:len(child_chromosome)] = reversed(child_chromosome[point:len(child_chromosome)]) # Reverse the targets from the selected index to the end of chromosome.
    return child_chromosome 


# Scramble Mutation.
def scrambleMutation(child_chromosome):
    point1 = random.randint(1, len(child_chromosome)) # Select first random index.
    point2 = random.randint(1, len(child_chromosome)) # Select second random index.
    
    if(point1 > point2):
        temp = point1
        point1 = point2
        point2 = temp
        
    random.shuffle(child_chromosome[point1:point2]) # Mix the targets within the two selected index.
    return child_chromosome


# Swap Mutation.
def swapMutation(child_chromosome, len_cities):
    point1 = random.randint(1, len_cities - 1)
    point2 = random.randint(1, len_cities - 1)
    child_chromosome[point1], child_chromosome[point2] = ( # Selects 2 random genes and exchanges them.
        child_chromosome[point2],
        child_chromosome[point1],
    )
    return child_chromosome


# Calculating distance of the cities.
def calcDistance(cities):
    total_sum = 0
    for i in range(len(cities) - 1):
        cityA = cities[i]
        cityB = cities[i + 1]

        d = math.sqrt(
            math.pow(cityB[0] - cityA[0], 2) + math.pow(cityB[1] - cityA[1], 2)
        )
        total_sum += d

    return total_sum


# Selecting the population.
def selectPopulation(cities, size):
    population = []
    
    for i in range(size): # size = number of possible paths.
        c = cities.copy() # Copy in order to not change the original order of targets.
        cities_without_first = c[1:] # Remove the first target.     
        random.shuffle(cities_without_first) # Get a random path between the targets.
        cities_first_city_fixed = cities_without_first
        cities_first_city_fixed.insert(0, c[0]) # Add the first target back to always be at the beginning of the chromosome.
        
        distance = calcDistance(cities_first_city_fixed) # Calculate the fitness value (= total distance between the targets).
        population.append([distance, cities_first_city_fixed]) # Adds the path (= chromosome) and its total distance to the papulation.
    fittest = sorted(population)[0] # Takes the fittest (= shortest path).

    return population, fittest # Returns the current population and the shortest path.


# The Genetic Algorithm.
def geneticAlgorithm(
    population,
    len_cities, # The number of targets.
    selection_function, # The selection function.
    mutation_function, # The mutation function.
    mutation_rate, # The probability to perform a mutation operator.
    crossover_rate, # The probability to perform a crossover operator.
):
    global generations_number
    gen_number = 0 # The generation index.
    
    for i in range(generations_number): # The number of generations.
        new_population = []
        
        for i in range(int(len(population) / 2)): # Divided by two because we select two parents in each generation.

            if random.random() < crossover_rate: # random.random() Returns a random number between 0.0 - 1.0.
                # SELECTION
                parent_chromosome1, parent_chromosome2 = selection_function(population)
                
                # CROSSOVER (Order Crossover Operator)
                point = random.randint(1, len_cities - 1) # Selects a random index.
                # First child.
                child_chromosome1 = parent_chromosome1[1][0:point] # Selects a sub-path (from its beginning - to the "point" index).
                for cluster_index in parent_chromosome2[1]: # Adds the missing targets from the second parent.
                    if (cluster_index in child_chromosome1) == False:
                        child_chromosome1.append(cluster_index)
                # Second child.
                child_chromosome2 = parent_chromosome2[1][0:point]
                for cluster_index in parent_chromosome1[1]:
                    if (cluster_index in child_chromosome2) == False:
                        child_chromosome2.append(cluster_index)

            # If crossover not happen
            else: # Choose two random paths.
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]
            
            # MUTATION
            if random.random() < mutation_rate: # random.random() Returns a random number between 0.0 - 1.0.
                child_chromosome1 = mutation_function(child_chromosome1)
                child_chromosome2 = mutation_function(child_chromosome2)
            
            # Adding the two children to the new population.
            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])
        
        population = new_population

        gen_number += 1
        print(gen_number, sorted(population)[0][0])

    answer = sorted(population)[0] # Prints shortest path found.

    return answer, gen_number

# The Genetic Algorithm version to steady state selection.
def geneticAlgorithmSteadyStateSelection(
    population,
    len_cities, # The number of targets.
    mutation_function, # The mutation function.
    mutation_rate, # The probability to perform a mutation operator.
    crossover_rate, # The probability to perform a crossover operator.
):
    global steady_state_selection_rate
    global generations_number
    gen_number = 0
    size_of_fittest_population = int(len(population)*steady_state_selection_rate)

    for i in range(generations_number):
        new_population = []
        sorted_population = sorted(population)
        # The fittest individuals are copied to the new population.
        for i in range(size_of_fittest_population):
            new_population.append(sorted_population[i])
        num_of_least_fittest = len(population)-size_of_fittest_population # Size of the population to be replaced
        for i in range(int(num_of_least_fittest / 2)):
            # SELECTION (Steady State)
            # Two fittest individuals are selected randomally to produce offspring
            fittest_parent1_index = random.randint(0, size_of_fittest_population)
            fittest_parent2_index = random.randint(0, size_of_fittest_population)

            parent_chromosome1 = population[fittest_parent1_index]
            parent_chromosome2 = population[fittest_parent2_index]
                
            # CROSSOVER (Order Crossover Operator)
            if random.random() < crossover_rate: # random.random() Returns a random number between 0.0 - 1.0.
                point = random.randint(0, len_cities - 1) # Selects a random index.
                # First child.
                child_chromosome1 = parent_chromosome1[1][0:point] # Selects a sub-path (from its beginning - to the "point" index).
                for j in parent_chromosome2[1]: # Adds the missing targets from the second parent.
                    if (j in child_chromosome1) == False:
                        child_chromosome1.append(j)
                # Second child.
                child_chromosome2 = parent_chromosome2[1][0:point]
                for j in parent_chromosome1[1]:
                    if (j in child_chromosome2) == False:
                        child_chromosome2.append(j)

            # If crossover not happen
            else: # Choose two random paths.
                child_chromosome1 = random.choices(population)[0][1]
                child_chromosome2 = random.choices(population)[0][1]
                
            # MUTATION
            if random.random() < mutation_rate: # random.random() Returns a random number between 0.0 - 1.0.
                child_chromosome1 = mutation_function(child_chromosome1, len_cities)
                child_chromosome2 = mutation_function(child_chromosome2, len_cities)

            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])
        
        population = new_population
        
        gen_number += 1

        print(gen_number, sorted(population)[0][0])

    answer = sorted(population)[0] # Prints shortest path found.

    return answer, gen_number