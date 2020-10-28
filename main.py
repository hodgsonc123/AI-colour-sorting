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
def random_sol():
    sol = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
    # Shuffle the elements in the list randomly. Shuffles in place and does not return a value
    rnd.shuffle(sol)
    return sol


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


def random_swap(solution):

    # initialise swap array and variables
    neighbour_solution = []
    swap_val1 = 0
    swap_val2 = 0
    ran_position1 = 0
    ran_position2 = 0

    # take copy of the solution passed in
    neighbour_swap_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(solution) - 1)
    ran_position2 = rnd.randint(0, len(solution) - 1)

    # notes for team: should we put a check in to make sure that the ran_positions dont give the same value

    # switch the values of these two positions
    swap_val1 = neighbour_swap_solution[ran_position1]
    swap_val2 = neighbour_swap_solution[ran_position2]

    neighbour_swap_solution[ran_position1] = swap_val2
    neighbour_swap_solution[ran_position2] = swap_val1

    return neighbour_swap_solution # return the random switched array


def hill_climbing(hc_iterations):

    improvement_trace = []

    # generate a random solution using random_sol
    best_solution = random_sol()

    for i in range(hc_iterations):
        best_solution_distance = evaluate(colors, best_solution)

        ran_swap_solution = random_swap(best_solution)
        ran_swap_solution_distance = evaluate(colors, ran_swap_solution)

        if(ran_swap_solution_distance < best_solution_distance):
            best_solution = ran_swap_solution[:]
            improvement_trace.append(ran_swap_solution_distance)

    return best_solution, improvement_trace


def multi_hill_climbing(mhcIterations):
    # initialise arrays for solutions
    # initialise variables to store distances

    # for loop for mhcIterations
        # call hillclimbing algorithm

        # evaluate current solution

        # if the current solutions evaluation is better than the best soltuion evaluation then
            # make this the best solution

    # return the best solution
    return


# ***************************************************************************************************************
ncolors, colors = read_data("col100.txt")  # pass in file to reading function

print(f'Number of colours: {ncolors}')
print("First 5 colours:")
print(colors[0:5, :])  # prints rgb values for first five colours

# Plot all the colors in the order they are listed in the file
order1 = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
plot_colors(colors, order1)  # You will notice that colors are not ordered in the file

order2 = random_sol()
print("Another random solution: ", order2)
plot_colors(colors, order2)  # the colors are not ordered, but this is a different order

e1 = evaluate(colors, order1)
print(f'Evaluation of order1: {e1}')  # Displaying all decimals
print(f'Evaluation of order1: {np.round(e1, 4)}')  # rounding to display only 4 decimals. This is better for display

e2 = evaluate(colors, order2)
print(f'Evaluation of order2: {e2}')  # Displaying all decimals
print(f'Evaluation of order2: {np.round(e2, 4)}')  # rounding to display only 4 decimals. This is better for display
# NEW COMMENT BY WILL, hi there
#solution = random_sol()
#print(solution)

#neighbour = generate_random_swap(solution)
#print(neighbour)
best_sol_hc, imp_trace = hill_climbing(10000)
plot_colors(colors, best_sol_hc, 40)
e3 = evaluate(colors, best_sol_hc)
print(f'Evaluation of order hc: {e3}')  # Displaying all decimals
print(f'Evaluation of order hc: {np.round(e3, 4)}')  # rounding to display only 4 decimals. This is better for display

