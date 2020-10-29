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


# random swap function. swaps two random positions in the given array (ordering of colours)
# input: solution: solution, ordering of colours
# output: swap_solution, ordering of colours with two random positions swapped
def random_swap(solution):
    # initialise swap array and variables
    neighbour_solution = []
    swap_val1 = 0
    swap_val2 = 0
    ran_position1 = 0
    ran_position2 = 0

    # take copy of the solution passed in
    swap_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(swap_solution) - 1)
    ran_position2 = rnd.randint(0, len(swap_solution) - 1)

    # notes for team: should we put a check in to make sure that the ran_positions dont give the same value
    # If random positions are the same then change ran_position2
    while ran_position1 == ran_position2:
        ran_position2 = rnd.randint(0, len(swap_solution) - 1)

    # store positions being swapped in temp variables
    swap_val1 = swap_solution[ran_position1]
    swap_val2 = swap_solution[ran_position2]

    # complete swap by swapping values at the random positions
    swap_solution[ran_position1] = swap_val2
    swap_solution[ran_position2] = swap_val1

    return swap_solution  # return the random swap solution


# inverse function
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


# scramble function
def random_scramble(solution):
    # take copy of the solution passed in
    scramble_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(scramble_solution) - 1)
    ran_position2 = rnd.randint(0, len(scramble_solution) - 1)

    # If random positions are the same then change ran_position2
    while ran_position1 == ran_position2:
        ran_position2 = rnd.randint(0, len(scramble_solution) - 1)

    # Order the random positions so ran_position1 is smaller than ran_position2
    if ran_position1 > ran_position2:
        placeholder = ran_position1
        ran_position1 = ran_position2
        ran_position2 = placeholder

    # Take the section that we want to scramble
    scramble_section = scramble_solution[ran_position1:ran_position2]
    # Shuffle all the indexes in the section we want to scramble
    rnd.shuffle(scramble_section)

    # Put solution back together
    # Get the unscrambled section at the start of the solution
    section_before_rnd1 = scramble_solution[0:ran_position1]
    # Get the unscrambled section at the end of the solution
    section_after_rnd2 = scramble_solution[ran_position2:len(scramble_solution)]
    # Add the start section to the solution
    scramble_solution = section_before_rnd1
    # Add the scrambled section to the solution
    scramble_solution.extend(scramble_section)
    # Finally add the end section to the solution
    scramble_solution.extend(section_after_rnd2)

    return scramble_solution


# hill_climbing function. generates random solution, performs a random swap of two elements in that solution
# compared the euclidean distance between both solution and stores the best with the lowest distance
# Input: hc_iterations (the number of iterations to run the random swap check) and permutation method e.g. 'swap', 'inversion', 'scramble'
# Output: best_solution, the best solution during hill climbing process
#         improvement_trace, storing the distance at every point an improvement has been made
def hill_climbing(hc_iterations, method_choice):
    hc_improvement_trace = []  # stores distance improvements

    # generate a random solution using random_sol
    hc_best_solution = random_sol()

    for i in range(hc_iterations):
        best_solution_distance = evaluate(colors, hc_best_solution)

        if method_choice == "swap":
            random_swap(hc_best_solution)

            ran_swap_solution = random_swap(hc_best_solution)
            ran_swap_solution_distance = evaluate(colors, ran_swap_solution)

            if ran_swap_solution_distance < best_solution_distance:
                hc_best_solution = ran_swap_solution[:]
                hc_improvement_trace.append(ran_swap_solution_distance)

        if method_choice == "inversion":
            random_inverse(hc_best_solution)

            ran_inverse_solution = random_scramble(hc_best_solution)
            ran_inverse_solution_distance = evaluate(colors, ran_inverse_solution)

            if ran_inverse_solution_distance < best_solution_distance:
                hc_best_solution = ran_inverse_solution[:]
                hc_improvement_trace.append(ran_inverse_solution_distance)

        if method_choice == "scramble":
            random_scramble(hc_best_solution)

            ran_scramble_solution = random_scramble(hc_best_solution)
            ran_scramble_solution_distance = evaluate(colors, ran_scramble_solution)

            if ran_scramble_solution_distance < best_solution_distance:
                hc_best_solution = ran_scramble_solution[:]
                hc_improvement_trace.append(ran_scramble_solution_distance)

    return hc_best_solution, hc_improvement_trace


def multi_hill_climbing(mhc_iterations, hc_iterations):
    # initialise arrays for solutions
    # initialise variables to store distances
    mhc_best_solution_distance = 1000  # number larger than any possible distance
    mhc_improvement_trace = []

    for i in range(mhc_iterations):
        current_solution, hc_improve_trace = hill_climbing(hc_iterations)

        current_solution_distance = evaluate(colors, current_solution)

        if current_solution_distance < mhc_best_solution_distance:
            mhc_best_solution = current_solution
            mhc_best_solution_distance = current_solution_distance
            mhc_improvement_trace.append(current_solution)

    return mhc_best_solution, mhc_best_solution_distance, mhc_improvement_trace


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

best_sol_hc, imp_trace = hill_climbing(1000, "inversion") # Include either "swap", "inversion" or "scramble"
plot_colors(colors, best_sol_hc, 40)
e3 = evaluate(colors, best_sol_hc)
print(f'Evaluation of order hc: {e3}')  # Displaying all decimals
print(f'Evaluation of order hc: {np.round(e3, 4)}')  # rounding to display only 4 decimals. This is better for display

#best_sol_mhc, best_sol_mhc_distance, mhc_imp_trace = multi_hill_climbing(3, 10000)
#plot_colors(colors, best_sol_mhc, 40)
#e4 = evaluate(colors, best_sol_hc)
#print(f'Evaluation of order mhc: {e4}')  # Displaying all decimals
#print(f'Evaluation of order mhc: {np.round(e4, 4)}')  # rounding to display only 4 decimals. This is better for display


