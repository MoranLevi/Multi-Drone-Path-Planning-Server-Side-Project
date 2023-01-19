import random
import math
import matplotlib.pyplot as plt
from numpy.random import choice
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Get cities info.
def getCity(fileName):
    cities = []
    f = open("targetsFiles/" + fileName)
    for i in f.readlines():
        node_city_val = i.split()
        cities.append(
            [float(node_city_val[0]), float(node_city_val[1])]
        )

    return cities


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

# Tournament Selection.
def tournamentSelection(population, TOURNAMENT_SELECTION_SIZE):
    parent_chromosome1 = sorted( # First parent.
                    random.choices(population, k=TOURNAMENT_SELECTION_SIZE)
                )[0] # Selects k random paths, sorts them, and choose the shortest one.
    parent_chromosome2 = sorted( # Second parent.
                    random.choices(population, k=TOURNAMENT_SELECTION_SIZE)
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

# Swap Mutation.
def swapMutation(child_chromosome, lenCities):
    point1 = random.randint(1, lenCities - 1)
    point2 = random.randint(1, lenCities - 1)
    child_chromosome[point1], child_chromosome[point2] = ( # Selects 2 random genes and exchanges them.
        child_chromosome[point2],
        child_chromosome[point1],
    )
    return child_chromosome


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

# The Genetic Algorithm.
def geneticAlgorithm(
    population,
    lenCities, # The number of targets.
    TOURNAMENT_SELECTION_SIZE, # The number of random chromosomes to compete to select 2 parents from them.
    MUTATION_RATE, # The probability to perform a mutation operator.
    CROSSOVER_RATE, # The probability to perform a crossover operator.
):
    gen_number = 0 # The generation index.
    
    for i in range(5): # The number of generations.
        new_population = []
        
        for i in range(int(len(population) / 2)): # Divided by two because we select two parents in each generation.
            # SELECTION
            if random.random() < CROSSOVER_RATE: # random.random() Returns a random number between 0.0 - 1.0.
                #parent_chromosome1, parent_chromosome2 = tournamentSelection(population, TOURNAMENT_SELECTION_SIZE)
                parent_chromosome1, parent_chromosome2 = rankSelection(population)
                
             # CROSSOVER (Order Crossover Operator)
                point = random.randint(1, lenCities - 1) # Selects a random index.
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
            if random.random() < MUTATION_RATE: # random.random() Returns a random number between 0.0 - 1.0.
                #Swap Mutation
                #child_chromosome1 = swapMutation(child_chromosome1, lenCities)
                #child_chromosome2 = swapMutation(child_chromosome2, lenCities)
                
                #Inversion Mutation
                child_chromosome1 = inversionMutation(child_chromosome1)
                child_chromosome2 = inversionMutation(child_chromosome2)
                
                #Scramble Mutation
                #child_chromosome1 = scrambleMutation(child_chromosome1)
                #child_chromosome2 = scrambleMutation(child_chromosome2)
            
            # Adding the two children to the new population.
            new_population.append([calcDistance(child_chromosome1), child_chromosome1])
            new_population.append([calcDistance(child_chromosome2), child_chromosome2])
            
        # REPLACEMENT
        # Selecting two of the best options we have (elitism).
        #sortedPopOld = sorted(population)
        #new_population.append(sortedPopOld[0])
        #new_population.append(sortedPopOld[1])
        
        population = new_population

        gen_number += 1
        #if gen_number % 10 == 0: # Prints shortest path every 10 rounds.
        print(gen_number, sorted(population)[0][0])

    answer = sorted(population)[0] # Prints shortest path found.

    return answer, gen_number

# Draw cities and answer map.
def drawMap(city, answer, color):
    city_index = 1
    for cluster_index in city: # Draws the targets.
        plt.plot(cluster_index[0], cluster_index[1], "ro") # "ro" = red marking for each target.
        plt.annotate(city_index, (cluster_index[0], cluster_index[1])) # Adds the index for each target: cluster_index[0] = index, cluster_index[1] = x, cluster_index[2] = y.
        city_index += 1

    for i in range(len(answer[1])): # Draws the line between the targets.
        try:
            first = answer[1][i]
            second = answer[1][i + 1]
            plt.plot([first[0], second[0]], [first[1], second[1]], color)
        except: # In case there is an out of range exception (because of i+1).
            continue

    # Draws the line between the first and the last targets.
    #first = answer[1][0]
    #second = answer[1][-1]
    plt.plot([first[0], second[0]], [first[1], second[1]], color)

# Gets cluster of cities (targets) with the same label given by KMeans.
def getCluster(cities, labels, label_index):
    cluster = []
    cluster_index = np.where(labels == label_index)
    for index in cluster_index[0]:
        cluster.append(cities[index])
    return cluster
       
def start_algorithm(fileName, numberOfDrones):
    # Initial values.
    POPULATION_SIZE = 2000 # The number of possible paths.
    TOURNAMENT_SELECTION_SIZE = 4 # The number of random chromosomes to compete to select 2 parents from them.
    MUTATION_RATE = 0.1 # The probability to perform a mutation operator.
    CROSSOVER_RATE = 0.9 # The probability to perform a crossover operator.
    K = int(numberOfDrones) # The number groups to divide the targets.
    results = []
    # color = ""
    
    cities = getCity(fileName) # Read targets from file.
            
    # Create a list to store the silhouette scores for each number of clusters
    scores = []

    # Create a list to store the paths to send to front-end
    return_list = []

    c = cities.copy()
    cities_without_first = c[1:] # Remove the first target in order to add it again to all of the clusters.
        
    if K == -1: 
        # Loop over a range of values for n_clusters
        for n_clusters in range(2, 11):
            # Create a KMeans model with the current value of n_clusters
            kmeans = KMeans(n_clusters=n_clusters, n_init=1)
            
            kmeans.fit(cities_without_first) # Fill the clusters with the targets.
            labels = kmeans.labels_ # Get the labels of the targets.
                
            # Compute the avg silhouette score for the current model
            score = silhouette_score(cities_without_first, labels)
    
            # Append the score to the list of scores
            scores.append([n_clusters, score])
             
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        K = sorted_scores[0][0]
    
    # Clustering the targets using KMeans
    kmeans = KMeans(n_clusters = K, n_init=1) # Create new instance to be filled K clusters.
    
    kmeans.fit(cities_without_first) # Fill the clusters with the targets.
    labels = kmeans.labels_ # Get the labels of the targets.
        
    # Create dictionary of the clusters (key = label, value = group of chromosomes).
    clusters = {i: getCluster(cities_without_first, labels, i) for i in range(kmeans.n_clusters)}
    
    sum_clusters = 0 # A veriable to sum total distance of all clusters.
    cluster_index = 0 # A veriable to decide each cluster's color.
        
    # Calculate shortest path for each cluster and sum their distances.
    for clusterOfCities in clusters.values():
        try:
            clusterOfCities.remove(c[0])
        except:
            pass
        clusterOfCities.insert(0, c[0]) # Add the initial target to each cluster.
            
        firstPopulation, firstFittest = selectPopulation(clusterOfCities, int(POPULATION_SIZE/K)) # Select initial population.
        answer, genNumber = geneticAlgorithm( # answer = shortest path that was found, genNumber = generation number.
            firstPopulation,
            len(clusterOfCities),
            TOURNAMENT_SELECTION_SIZE,
            MUTATION_RATE,
            CROSSOVER_RATE,
        )
        results.append(answer)
        print("\n----------------------------------------------------------------")
        print("Generation: " + str(genNumber))
        print("Fittest chromosome distance before training: " + str(firstFittest[0]))
        print("Fittest chromosome distance after training: " + str(answer[0]))
        print("----------------------------------------------------------------\n")
            
        # if cluster_index == 0:
        #     color = "blue"
        # elif cluster_index == 1:
        #     color = "green"
        # elif cluster_index == 2:
        #     color = "purple"
        # elif cluster_index == 3:
        #     color = "gray"
        # elif cluster_index == 4:
        #     color = "olive"
        # else:
        #     color = "brown"
        # cluster_index += 1
        # drawMap(cities, answer, color)
        sum_clusters += answer[0]
        targets_list = answer[1]
        targets_list_with_indexes = []
        for target in targets_list:
            targets_list_with_indexes.append(cities.index(target) + 1)
        return_list.append(targets_list_with_indexes)

    # plt.title('Total Shortest Distance = ' + str(round(sum_clusters, 2)))
    # plt.show()
    # results.append(sum_clusters)
    
    # plt.plot(results, 'bo-')
    # run = 0
    # for res in results:
    #     plt.annotate(str(round(res, 2)), (run, res))
    #     run += 1
    # plt.xlabel('Iteration')
    # plt.ylabel('Distance')
    # plt.xticks(np.arange(len(results)), np.arange(1, len(results)+1))
    # plt.show()

    return return_list

# main()
