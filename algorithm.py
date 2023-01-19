import matplotlib.pyplot as plt
import numpy as np
import geneticAlgorithm
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
    plt.plot([first[0], second[0]], [first[1], second[1]], color)


# Gets cluster of cities (targets) with the same label given by KMeans.
def getCluster(cities, labels, label_index):
    cluster = []
    cluster_index = np.where(labels == label_index)
    for index in cluster_index[0]:
        cluster.append(cities[index])
    return cluster


# Start the algorithm that calculate clusters and paths- using KMeans and Genetic Algorithm.
# Returns the paths for each cluster.
def start_algorithm(fileName, numberOfDrones):
    # Initial values.
    population_size = 2000 # The number of possible paths.
    mutation_rate = 0.1 # The probability to perform a mutation operator.
    crossover_rate = 0.9 # The probability to perform a crossover operator.
    K = int(numberOfDrones) # The number groups to divide the targets.
    results = []
    
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
        
    # Calculate shortest path for each cluster and sum their distances.
    for cluster_of_cities in clusters.values():
        try:
            cluster_of_cities.remove(c[0])
        except:
            pass
        cluster_of_cities.insert(0, c[0]) # Add the initial target to each cluster.
            
        firstPopulation, firstFittest = geneticAlgorithm.selectPopulation(cluster_of_cities, int(population_size/K)) # Select initial population.
        answer, genNumber = geneticAlgorithm.geneticAlgorithm( # answer = shortest path that was found, genNumber = generation number.
            firstPopulation,
            len(cluster_of_cities),
            geneticAlgorithm.rankSelection,
            geneticAlgorithm.inversionMutation,
            mutation_rate,
            crossover_rate,
        )

        results.append(answer)
        print("\n----------------------------------------------------------------")
        print("Generation: " + str(genNumber))
        print("Fittest chromosome distance before training: " + str(firstFittest[0]))
        print("Fittest chromosome distance after training: " + str(answer[0]))
        print("----------------------------------------------------------------\n")
            
        # drawMap(cities, answer, color = "blue")
        sum_clusters += answer[0]
        targets_list = answer[1]
        targets_list_with_indexes = []
        for target in targets_list:
            targets_list_with_indexes.append(cities.index(target) + 1)
        return_list.append(targets_list_with_indexes)

    return return_list

