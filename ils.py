import copy

import numpy as np  # Numerical library, used keeing the list of colours and computing the Euclidean distance
import matplotlib.pyplot as plt
import random as rnd

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

# Function to generate a random solution (random ordering)  - we can generate a random ordering of the list by using
# the shuffle function from the random library
def random_sol(ncolors):
    sol = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
    # Shuffle the elements in the list randomly. Shuffles in place and does not return a value
    rnd.shuffle(sol)
    return sol

# calculates the distance between two colours
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

# random swap function. swaps two random positions in the given array (ordering of colours)
# input: solution: solution, ordering of colours
# output: swap_solution, ordering of colours with two random positions swapped

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
def hill_climbing(hc_iterations, sols):
    hc_improvement_trace = []  # stores distance improvements
    # generate a random solution using random_sol
    hc_best_solution = sols[:]

    for i in range(hc_iterations):
        best_solution_eval = evaluate(sols, hc_best_solution)
        ran_inverse_solution = random_inverse(hc_best_solution)
        ran_inverse_solution_eval = evaluate(sols, ran_inverse_solution)

        if ran_inverse_solution_eval < best_solution_eval:
            hc_best_solution = ran_inverse_solution[:]
            hc_improvement_trace.append(ran_inverse_solution_eval)

    return hc_best_solution, hc_improvement_trace


def ils(ils_iterations, hc_iterations, cols):
    best_solution = random_sol(len(cols))

    best_solution, trace = hill_climbing(hc_iterations, best_solution)
    best_eval = evaluate(cols, best_solution)

    for i in range(ils_iterations):
        new_sol, new_trace = hill_climbing(hc_iterations, random_inverse(best_solution))
        new_eval = evaluate(cols, new_sol)
        if new_eval < best_eval:
            best_solution = new_sol[:]
            best_eval = new_eval
    return best_solution

# ***************************************************************************************************************
ncolors100, colors100 = read_data("col100.txt")  # pass in 100 colour file to reading function
ncolors500, colors500 = read_data("col500.txt")  # pass in 500 colour file file to reading function

print(f'Number of colours 100: {ncolors100}')
print("First 5 colours of 100:")
print(colors100[0:5, :])  # prints rgb values for first five colours

# Plot all the colors in the order they are listed in the file
order100 = list(range(ncolors100))  # list of consecutive numbers from 0 to ncolors
print('\nUnordered solution 100...')
plot_colors(colors100, order100, 20)  # You will notice that colors are not ordered in the file
unordered_100_eval = evaluate(colors100, order100)
print(f'Evaluation of unordered 100: {unordered_100_eval}')  # Displaying all decimals
print(f'Evaluation of unordered 100: {np.round(unordered_100_eval, 4)}')  # rounding to display only 4 decimals. This is better for display

print('\nGenerating hill climb 100 solution...')
ils_best100 = ils(20000, 3, colors100) #
plot_colors(colors100, ils_best100, 20)
hc_100_eval = evaluate(colors100, ils_best100) # evaluate the solution
print(f'Evaluation of order hc 100: {hc_100_eval}')  # Displaying all decimals
print(f'Evaluation of order hc 100: {np.round(hc_100_eval, 4)}')  # rounding to display only 4 decimals. This is better for display

#######################################################################################################################

# order500 = list(range(ncolors500))  # list of consecutive numbers from 0 to ncolors
# print('\nUnordered solution 500...')
# plot_colors(colors500, order500, 40)  # You will notice that colors are not ordered in the file
# unordered_500_eval = evaluate(colors500, order500)# evaluate the solution
# print(f'Evaluation of unordered 500: {unordered_500_eval}')  # Displaying all decimals
# print(f'Evaluation of unordered 500: {np.round(unordered_500_eval, 4)}')  # rounding to display only 4 decimals. This is better for display
#
# print('\nGenerating hill climb 500 solution...')
# ils_best500 = hill_climbing(20, 200,colors500 ) # Include either "swap", "inversion" or "scramble"
# plot_colors(colors500, ils_best500, 40)
# hc_500_eval = evaluate(colors500, ils_best500)# evaluate the solution
# print(f'Evaluation of order hc 500: {hc_500_eval}')  # Displaying all decimals
# print(f'Evaluation of order hc 500: {np.round(hc_500_eval, 4)}')  # rounding to display only 4 decimals. This is better for display

##########################################################################################################################

# plot the trace of one run of hillclimb for each file
# plt.figure()
# plt.suptitle('HC Improvement trace 100')
# plt.plot(imp_trace100)
# plt.ylabel("Evaluation Value")
# plt.xlabel("Improvement No.")
# plt.show()

# plt.figure()
# plt.suptitle('HC Improvement trace 500')
# plt.plot(imp_trace500)
# plt.ylabel("Evaluation Value")
# plt.xlabel("Improvement No.")
# plt.show()
