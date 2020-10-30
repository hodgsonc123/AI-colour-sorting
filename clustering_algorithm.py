import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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
    #print(df)

    # Provide the number of clusters wanted
    number_of_clusters = k

    # Create the model, setting the number of clusters, fixing the random seed
    kmeans = KMeans(n_clusters=number_of_clusters)
    # Fit the model
    clusters = kmeans.fit_predict(df)

    #print(clusters)  # Print the results, cluster membership of each state

    # Add the cluster info to the original dataset as a new column
    df['Cluster'] = clusters  # New column called cluster
    #print(df)  # Now the data set has a new column

    # Can see how many items are in each cluster
    #print(df.groupby('Cluster').size())

    # print("now")
    # print(df.iloc[0:1, 0:3])

    # Sort the values by their cluster number
    df.sort_values(['Cluster'], inplace=True)
    #print(df)

    # Remove the cluster identifier column
    df.drop(['Cluster'], axis=1, inplace=True)

    # Change from dataframe to numpy array
    final_sorted_colors = df.to_numpy()
    #print(final_sorted_colors)

    return ncolors, col, final_sorted_colors


# Best iteration
# Input: file name, number of clusters, number of iterations
# Output: number of colours, list of colours of best solution
def best_iteration(file, clusters, iterations):
    ncolors, orig_colors = read_data(file)
    print(ncolors)

    # Print / display original colored list for visual comparison
    order1 = list(range(ncolors))
    plot_colors(orig_colors, order1)

    best_solution = orig_colors
    best_solution_distance = 10000

    order = list(range(ncolors))
    best_solution_distance = evaluate(best_solution, order)

    for i in range(iterations):
        print("Iteration: ", i+1)
        ncolors, orig_colors, sorted_colors = k_means_clustering(file, clusters)
        current_solution_distance = evaluate(sorted_colors, order)

        if current_solution_distance < best_solution_distance:
                    best_solution = sorted_colors
                    best_solution_distance = current_solution_distance

    return ncolors, best_solution


# ---- main ----

# Testing variables
FILE = "col500.txt"
NUMBER_CLUSTERS = 7
NUMBER_ITERATIONS = 50

# Call best_iteration function with the variables above
ncols, best_sorted_colors = best_iteration(FILE, NUMBER_CLUSTERS, NUMBER_ITERATIONS)
# Plot the best_sorted_color list as a visual
order = list(range(ncols))
plot_colors(best_sorted_colors, order)
# Print to terminal the variables used for the current solution
print("Final ordered list: ")
print(*['File:', FILE, '| Number of clusters:', NUMBER_CLUSTERS, '| Number of iterations:', NUMBER_ITERATIONS])