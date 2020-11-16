import numpy as np  # Numerical library, used keeing the list of colours and computing the Euclidean distance
import matplotlib.pyplot as plt
import random as rnd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read in the color data file
# Input: string with file name
# Output: the number of colours (integer), and a list numpy arrays with all the colours
from numpy.core._multiarray_umath import ndarray


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


# Function to generate a random solution (random ordering)  - we can generate a random ordering of the list by using
# the shuffle function from the random library
def random_sol(length):
    sol = list(range(length))  # list of consecutive numbers from 0 to ncolors
    # Shuffle the elements in the list randomly. Shuffles in place and does not return a value
    rnd.shuffle(sol)
    return sol


# inverse function, inverts values between two positions in given array of colour order
# input: solution, ordering of colours
# output: inverse_solution, inverted values between two points in array
def random_inverse(solution):
    # take copy of the solution passed in
    inverse_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(inverse_solution) - 1)
    ran_position2 = rnd.randint(0, len(inverse_solution) - 1)

    # If random positions are the same then change ran_position2
    while ran_position1 == ran_position2:
        ran_position2 = rnd.randint(0, len(inverse_solution) - 1)

    # Order the random positions so ran_position1 is smaller than ran_position2
    if ran_position1 > ran_position2:
        placeholder = ran_position1
        ran_position1 = ran_position2
        ran_position2 = placeholder

    # Take the section that we want to inverse
    inverse_selection = inverse_solution[ran_position1:ran_position2]
    # Reverse all the indexes in the section we want to inverse
    inverse_selection.reverse()

    # Put solution back together
    # Get the original section at the start of the solution
    section_before_rnd1 = inverse_solution[0:ran_position1]
    # Get the section at the end of the solution
    section_after_rnd2 = inverse_solution[ran_position2:len(inverse_solution)]
    # Add the start section to the solution
    inverse_solution = section_before_rnd1
    # Add the reversed section to the solution
    inverse_solution.extend(inverse_selection)
    # Finally add the end section to the solution
    inverse_solution.extend(section_after_rnd2)

    return inverse_solution

# hill_climbing function. generates random solution, performs a random swap of two elements in that solution
# compared the euclidean distance between both solution and stores the best with the lowest distance
# Input: hc_iterations (the number of iterations to run the random swap check) and permutation method e.g. 'swap', 'inversion', 'scramble'
# Output: best_solution, the best solution during hill climbing process
#         improvement_trace, storing the distance at every point an improvement has been made
def hill_climbing(length, sol, hc_iterations, method_choice):
    hc_improvement_trace = []  # stores distance improvements

    # generate a random solution using random_sol
    hc_best_solution = random_sol(length)

    for i in range(hc_iterations):
        best_solution_distance = evaluate(sol, hc_best_solution)

        if method_choice == "inverse":
            random_inverse(hc_best_solution)

            ran_inverse_solution = random_inverse(hc_best_solution)
            ran_inverse_solution_distance = evaluate(colors, ran_inverse_solution)

            if ran_inverse_solution_distance < best_solution_distance:
                hc_best_solution = ran_inverse_solution[:]
                hc_improvement_trace.append(ran_inverse_solution_distance)

        else:
            print("invalid algorithm")

    return hc_best_solution, hc_improvement_trace


def multi_hill_climbing(length, sol, mhc_iterations, hc_iterations, method_choice):
    # initialise arrays for solutions
    # initialise variables to store distances
    mhc_best_solution_distance = 1000  # number larger than any possible distance
    mhc_improvement_trace = []

    for i in range(mhc_iterations):
        current_solution, hc_improve_trace = hill_climbing(length, sol, hc_iterations, method_choice)

        current_solution_distance = evaluate(sol, current_solution)

        if current_solution_distance < mhc_best_solution_distance:
            mhc_best_solution = current_solution
            mhc_best_solution_distance = current_solution_distance
            mhc_improvement_trace.append(current_solution)

    return mhc_best_solution, mhc_best_solution_distance, mhc_improvement_trace

# ***************************************************************************************************************


# ------------------------------------------------------------------------------------
#
#
#                           ABOVE IS MULTISTART HILLCLIMBING
#
#
#                              BELOW IS KMEANS CLUSTERING
#
#
# ------------------------------------------------------------------------------------




# K means clustering algorithm
# Input: file and k number of clusters
# Output: number of colours, original colour list, sorted colour list
def k_means_clustering(i, iterations, ncol, bestsol, k):
    ncolors = ncol
    col = bestsol

    # Create a new dataframe to hold the same info as txt file
    df = pd.DataFrame({'R': col[:, 0], 'G': col[:, 1], 'B': col[:, 2]})
    # print(df)

    # Provide the number of clusters wanted
    number_of_clusters = k

    # Create the model, setting the number of clusters, fixing the random seed
    kmeans = KMeans(n_clusters=number_of_clusters)
    # Fit the model
    clusters = kmeans.fit_predict(df)

    # print(clusters)  # Print the results, cluster membership of each state

    # Add the cluster info to the original dataset as a new column
    df['Cluster'] = clusters  # New column called cluster
    # print(df)  # Now the data set has a new column

    # Can see how many items are in each cluster
    # print(df.groupby('Cluster').size())

    # print("now")
    # print(df.iloc[0:1, 0:3])

    # Sort the values by their cluster number
    df.sort_values(['Cluster'], inplace=True)
    # print(df)

    intermittent_colors = df.to_numpy()
    new_color_list = np.empty((0, 3))

    if i == iterations - 1:
        for j in range(number_of_clusters):
            indiv_cluster = np.empty((0, 4))
            ind_clu_len = len(indiv_cluster)

            for z in range(ncolors):
                if intermittent_colors[z, 3] == j:

                    insert = np.array(intermittent_colors[z])
                    indiv_cluster = np.append(indiv_cluster, [insert], axis=0)

            ic = pd.DataFrame(indiv_cluster)
            ic.drop([3], axis=1, inplace=True)
            indiv_cluster2 = ic.to_numpy()
            length = len(indiv_cluster2)

            best_indcluster_mhc, best_sol_mhc_distance, mhc_imp_trace = multi_hill_climbing(length, indiv_cluster2, 1, 20000,
                                                                                            "inverse")  # Include either "swap", "inversion" or "scramble"

            new_ordered_list = np.empty((0, 3))

            for e in range(length):
                for e1 in range(len(best_indcluster_mhc)):
                    if best_indcluster_mhc[e1] == e:
                        addition = np.array(indiv_cluster2[e1])
                        new_ordered_list = np.append(new_ordered_list, [addition], axis=0)

            new_color_list = np.append(new_color_list, new_ordered_list, axis=0)

    else:
        # Remove the cluster identifier column
        df.drop(['Cluster'], axis=1, inplace=True)

        # Change from dataframe to numpy array
        new_color_list = df.to_numpy()
        # print(final_sorted_colors)

    return ncolors, col, new_color_list


# Best iteration
# Input: file name, number of clusters, number of iterations
# Output: number of colours, list of colours of best solution
def best_solution(ncol, col, clusters, iterations):
    orig_colors = col
    ncolors = ncol
    print(ncolors)

    # Print / display original colored list for visual comparison
    order1 = list(range(ncol))
    plot_colors(orig_colors, order1)

    best_solution = orig_colors
    best_solution_distance = 10000

    for i in range(iterations):
        print("Iteration: ", i + 1)
        ncolors, orig_colors, sorted_colors = k_means_clustering(i, iterations, ncolors, best_solution, clusters)
        order = list(range(ncolors))
        current_solution_distance = evaluate(sorted_colors, order)

        if current_solution_distance < best_solution_distance:
            best_solution = sorted_colors
            best_solution_distance = current_solution_distance

    return ncolors, best_solution, best_solution_distance


# ---- main ----

# Testing variables
FILE1 = "col100.txt"
NUMBER_CLUSTERS_FILE1 = 25
NUMBER_ITERATIONS_FILE1 = 200

FILE2 = "col500.txt"
NUMBER_CLUSTERS_FILE2 = 100
NUMBER_ITERATIONS_FILE2 = 500

# ------------------------------------------------------------------------------------
ncolors, colors = read_data("col100.txt")  # pass in file to reading function

# Call best_iteration function with the variables above
ncols, best_sorted_colors, best_distance2 = best_solution(ncolors, colors, NUMBER_CLUSTERS_FILE1,
                                                          NUMBER_ITERATIONS_FILE1)
# Plot the best_sorted_color list as a visual
order = list(range(ncols))
plot_colors(best_sorted_colors, order)

print("Final ordered list variables: ")
print(*['File:', FILE1, '| Number of clusters:', NUMBER_CLUSTERS_FILE1, '| Number of iterations:',
        NUMBER_ITERATIONS_FILE1, '| Evaluation: ', best_distance2])
