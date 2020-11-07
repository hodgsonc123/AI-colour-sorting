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
def random_sol():
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
def random_swap(solution):
    # take copy of the solution passed in
    swap_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(swap_solution) - 1)
    ran_position2 = rnd.randint(0, len(swap_solution) - 1)

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


# scramble function, randomly rearranges values between two random points in given array
# input: solution, array ordering of colours
# output: scramble_solution, array with scrambled values between two points
def random_scramble(solution):
    # take copy of the solution passed in
    scramble_solution = solution[:]

    # generate two random positions in the array
    ran_position1 = rnd.randint(0, len(scramble_solution) - 1)
    ran_position2 = rnd.randint(0, len(scramble_solution) - 1)

    # while/if random positions are the same then change ran_position2
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

        elif method_choice == "inversion":
            random_inverse(hc_best_solution)

            ran_inverse_solution = random_scramble(hc_best_solution)
            ran_inverse_solution_distance = evaluate(colors, ran_inverse_solution)

            if ran_inverse_solution_distance < best_solution_distance:
                hc_best_solution = ran_inverse_solution[:]
                hc_improvement_trace.append(ran_inverse_solution_distance)

        elif method_choice == "scramble":
            random_scramble(hc_best_solution)

            ran_scramble_solution = random_scramble(hc_best_solution)
            ran_scramble_solution_distance = evaluate(colors, ran_scramble_solution)

            if ran_scramble_solution_distance < best_solution_distance:
                hc_best_solution = ran_scramble_solution[:]
                hc_improvement_trace.append(ran_scramble_solution_distance)
        else:
            print("invalid algorithm")
        print(i)

    return hc_best_solution, hc_improvement_trace


def multi_hill_climbing(mhc_iterations, hc_iterations, method_choice):

    mhc_best_solution_distance = 1000  # number larger than any possible distance
    #mhc_improvement_trace = [] # initialise trace to store the evaluation each time improvement is made

    for i in range(mhc_iterations): # for mhc repetitions do...

        current_solution, hc_improve_trace = hill_climbing(hc_iterations, method_choice) #call hill climbing function for given iterations and method
        current_solution_distance = evaluate(colors, current_solution) # evaluate the given solution

        if current_solution_distance < mhc_best_solution_distance: # if new solution is better than the currnet best solution then
            mhc_best_solution = current_solution # store the current solution as the best solution
            mhc_best_solution_distance = current_solution_distance # store the current evaluation value as the best value
            #mhc_improvement_trace.append(current_solution) # add

    return mhc_best_solution, mhc_best_solution_distance,# mhc_improvement_trace

# evaluate bext method function
# tests each hill climb 'mutation operator' swap, inversion and scramble
# with iterations 5000, 10000, 15000, 20000, 25000, 30000 to find optimal number.
# produces a plot for each method and iteration combination
def evaluate_best_method():

    test_iterations = 0 # initialise variable to store number of iterations
    swap_trace = [] # initialise a trace for each method
    inversion_trace = []
    scramble_trace = []

    for i in range(6): # for 6 times do ... (6 chosen to test a reasonable range of iterations 5000-30000)
        test_iterations += 5000 # increase number of iterations by 5000
        best_sol_hc_swap, imp_trace_swap = hill_climbing(test_iterations, "swap") # run swap method
        swap_evaluation = evaluate(colors, best_sol_hc_swap) # evaluate best solution returned
        swap_trace.append(swap_evaluation) # add the evaluation value to the trace

        # same as swap method but for inversion
        best_sol_hc_inversion, imp_trace_inversion = hill_climbing(test_iterations, "inversion")
        inversion_evaluation = evaluate(colors, best_sol_hc_inversion)
        inversion_trace.append(inversion_evaluation)

        # same as swap method but for scramble
        best_sol_hc_scramble, imp_trace_scramble = hill_climbing(test_iterations, "scramble")
        scramble_evaluation = evaluate(colors, best_sol_hc_scramble)
        scramble_trace.append(scramble_evaluation)

    # plot trace for swap method
    plt.figure()
    plt.suptitle('HC Testing Iterations and swap')
    plt.plot(swap_trace)
    plt.ylabel("Distance Value")
    plt.xlabel("Iteration")
    plt.show()
    print(f'best swap: {evaluate(colors, best_sol_hc_swap)}')  # Display best swap inversion evaluation value

    # plot trace for inversion method
    plt.figure()
    plt.suptitle('HC Testing Iterations and inversion')
    plt.plot(inversion_trace)
    plt.ylabel("Distance Value")
    plt.xlabel("Iteration")
    plt.show()
    print(f'best inversion: {evaluate(colors, best_sol_hc_inversion)}')  # display best inversion evaluation value

    # plot trace for scramble method
    plt.figure()
    plt.suptitle('HC Testing Iterations and scramble')
    plt.plot(scramble_trace)
    plt.ylabel("Distance Value")
    plt.xlabel("Iteration")
    plt.show()
    print(f'best scramble solution: {evaluate(colors, best_sol_hc_scramble)}')  # Displaying  best scramble evaluation value

# ***************************************************************************************************************
ncolors, colors = read_data("col100.txt")  # pass in file to reading function

print(f'Number of colours: {ncolors}')
print("First 5 colours:")
print(colors[0:5, :])  # prints rgb values for first five colours

# Plot all the colors in the order they are listed in the file
order1 = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
plot_colors(colors, order1)  # You will notice that colors are not ordered in the file

e1 = evaluate(colors, order1)
print(f'Evaluation of unordered: {e1}')  # Displaying all decimals
print(f'Evaluation of unordered: {np.round(e1, 4)}')  # rounding to display only 4 decimals. This is better for display

best_sol_hc, imp_trace = hill_climbing(1000, "swap") # Include either "swap", "inversion" or "scramble"
plot_colors(colors, best_sol_hc, 40)
e3 = evaluate(colors, best_sol_hc)
print(f'Evaluation of order hc: {e3}')  # Displaying all decimals
print(f'Evaluation of order hc: {np.round(e3, 4)}')  # rounding to display only 4 decimals. This is better for display

best_sol_mhc, best_sol_mhc_distance= multi_hill_climbing(3, 2000, "swap") # Include either "swap", "inversion" or "scramble"
plot_colors(colors, best_sol_mhc, 40)
print('mhc sol', best_sol_mhc)
e4 = evaluate(colors, best_sol_mhc)
print(f'Evaluation of order mhc: {e4}')  # Displaying all decimals
print(f'Evaluation of order mhc: {np.round(e4, 4)}')  # rounding to display only 4 decimals. This is better for display

# plot the trace of one run of hillclimb
plt.figure()
plt.suptitle('HC Improvement trace')
plt.plot(imp_trace)
plt.ylabel("Distance Value")
plt.xlabel("Improvement No.")
plt.show()

#evaluate_best_method() # WARNING: TAKES ABOUT AN HOUR TO RUN. run method to evaluate best method
