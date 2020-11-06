import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


# Read in the color data file
# Input: string with file name
# Output: the number of colours (integer), and a list numpy arrays with all the colours
def read_data(fname):
    cols = np.loadtxt(fname, skiprows=4)  # The first 4 lines have text information, and are ignored
    ncols = len(cols)  # Total number of colours and list of colours
    return ncols, cols


# Display the colors as a strip of color bars
# Input: list of colors, order of colors, and height/ratio
def plot_colors(col_list, col_order, ratio=40):
    assert len(col_list) == len(col_order)
    img = np.zeros((ratio, len(col_list), 3))
    for i in range(0, len(col_list)):
        img[:, i, :] = col_list[col_order[i]]
    fig, axes = plt.subplots(1, figsize=(10, 6))  # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


def euclid(v, u):
    return np.linalg.norm(v - u)


# Evaluation function.  Measures the quality of a given solution (ordering of colours)
# The function computes the sum of the distances between all consecutive colours in the ordering
# Input: cols: list of colours
#        ordc: ordering of colours
# Output: real number with the sumf of pair-wise differences in the colour ordering
def evaluate(cols, ordc):
    adjacentColPairs = [[cols[ordc[i]], cols[ordc[i - 1]]] for i in range(1, len(ordc))]
    return sum([euclid(i[1], i[0]) for i in adjacentColPairs])


# K means clustering algorithm
# Input: file and k number of clusters
# Output: number of colours, original colour list, sorted colour list
def k_means_clustering(file, k):
    ncolors, col = read_data(file)

    # Create a new dataframe to hold the same info as txt file
    df = pd.DataFrame({'R': col[:, 0], 'G': col[:, 1], 'B': col[:, 2]})

    # Provide the number of clusters wanted
    number_of_clusters = k

    # Create the model, setting the number of clusters, fixing the random seed
    kmeans = KMeans(n_clusters=number_of_clusters)
    # Fit the model
    clusters = kmeans.fit_predict(df)

    # Add the cluster info to the original dataset as a new column
    df['Cluster'] = clusters  # New column called cluster

    # Sort the values by their cluster number
    df.sort_values(['Cluster'], inplace=True)

    # Remove the cluster identifier column
    df.drop(['Cluster'], axis=1, inplace=True)

    # Change from dataframe to numpy array
    final_sorted_colors = df.to_numpy()

    return ncolors, col, final_sorted_colors


# Best iteration
# Input: file name, number of clusters, number of iterations
# Output: number of colours, list of colours of best solution
def best_solution(file, clusters, iterations):
    ncolors, orig_colors = read_data(file)
    print(ncolors)

    # Print / display original colored list for visual comparison
    order1 = list(range(ncolors))
    plot_colors(orig_colors, order1)

    best_solution = orig_colors
    best_solution_distance = 10000

    order = list(range(ncolors))

    for i in range(iterations):
        print("Iteration: ", i+1)
        ncolors, orig_colors, sorted_colors = k_means_clustering(file, clusters)
        current_solution_distance = evaluate(sorted_colors, order)

        if current_solution_distance < best_solution_distance:
            best_solution = sorted_colors
            best_solution_distance = current_solution_distance

    return ncolors, best_solution, best_solution_distance


# Elbow method
# To determine the optimal value of k in KMeans
# Input: file name
def elbow_method(file):
    ncolors, col = read_data(file)

    # Distortion is calculated as the average of the squared distances from the cluster centers of
    # the respective clusters. Typically, the Euclidean distance metric is used.
    distortions = []
    # Inertias is the sum of squared distances of samples to their closest cluster center.
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, (int(ncolors/2)))

    # Create a new dataframe to hold the same info as txt file
    df = pd.DataFrame({'R': col[:, 0], 'G': col[:, 1], 'B': col[:, 2]})

    #  Iterate the values of k from 1 to half the number of colours in the file and calculate the
    #  values of distortions for each value of k and calculate the distortion and inertia for
    #  each value of k in the given range.
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(df)
        kmeanModel.fit(df)

        distortions.append(sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(df, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
        mapping2[k] = kmeanModel.inertia_

    # Plotting the different values of distortion
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method using Distortion')
    plt.show()

    # Plotting the different values of inertias
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Inertia')
    plt.title('The Elbow Method using Inertia')
    plt.show()


# ---- main ----

# File names that hold the colour data
FILE1 = "col100.txt"
FILE2 = "col500.txt"

# Call elbow_method to run the 'elbow method' on each dataset to visually determine the optimal number of clusters
elbow_method(FILE1)
elbow_method(FILE2)

# Set the number of clusters and iterations for each dataset
NUMBER_CLUSTERS_FILE1 = 25
NUMBER_ITERATIONS_FILE1 = 200
NUMBER_CLUSTERS_FILE2 = 50
NUMBER_ITERATIONS_FILE2 = 300


# Call best_iteration function with the variables above
ncols, best_sorted_colors, best_distance1 = best_solution(FILE1, NUMBER_CLUSTERS_FILE1, NUMBER_ITERATIONS_FILE1)
# Plot the best_sorted_color list as a visual
order = list(range(ncols))
plot_colors(best_sorted_colors, order)
# Print to terminal the variables used for the current solution


# Call best_iteration function with the variables above
ncols, best_sorted_colors, best_distance2 = best_solution(FILE2, NUMBER_CLUSTERS_FILE2, NUMBER_ITERATIONS_FILE2)
# Plot the best_sorted_color list as a visual
order = list(range(ncols))
plot_colors(best_sorted_colors, order)


# Print to terminal the variables used for the current solution
print("Final ordered list variables: ")
print(*['File:', FILE1, '| Number of clusters:', NUMBER_CLUSTERS_FILE1, '| Number of iterations:',
        NUMBER_ITERATIONS_FILE1, '| Evaluation: ', best_distance1])
print(*['File:', FILE2, '| Number of clusters:', NUMBER_CLUSTERS_FILE2, '| Number of iterations:',
        NUMBER_ITERATIONS_FILE2, '| Evaluation: ', best_distance2])


