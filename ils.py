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


# hill_climbing function. generates random solution, performs a random swap of two elements in that solution
# compared the euclidean distance between both solution and stores the best with the lowest distance
# Input: hc_iterations (the number of iterations to run the random swap check) and permutation method e.g. 'swap', 'inversion', 'scramble'
# Output: best_solution, the best solution during hill climbing process
#         improvement_trace, storing the distance at every point an improvement has been made
def hill_climbing(hc_iterations, method_choice, cols, ran_sol):
    hc_improvement_trace = []  # stores distance improvements
    hc_best_solution = ran_sol

    if method_choice == "swap":
        for i in range(hc_iterations):
            best_solution_eval = evaluate(cols, hc_best_solution)
            ran_swap_solution = random_swap(hc_best_solution)
            ran_swap_solution_eval = evaluate(cols, ran_swap_solution)

            if ran_swap_solution_eval < best_solution_eval:
                hc_best_solution = ran_swap_solution[:]
            hc_improvement_trace.append(ran_swap_solution_eval)

    elif method_choice == "inversion":
        for i in range(hc_iterations):
            best_solution_eval = evaluate(cols, hc_best_solution)
            ran_inverse_solution = random_inverse(hc_best_solution)
            ran_inverse_solution_eval = evaluate(cols, ran_inverse_solution)

            if ran_inverse_solution_eval < best_solution_eval:
                hc_best_solution = ran_inverse_solution[:]
                hc_improvement_trace.append(ran_inverse_solution_eval)

    elif method_choice == "scramble":
        for i in range(hc_iterations):
            best_solution_eval = evaluate(cols, hc_best_solution)
            ran_scramble_solution = random_scramble(hc_best_solution)
            ran_scramble_solution_eval = evaluate(cols, ran_scramble_solution)

            if ran_scramble_solution_eval < best_solution_eval:
                hc_best_solution = ran_scramble_solution[:]
                hc_improvement_trace.append(ran_scramble_solution_eval)
    else:
        print("invalid algorithm")
    return hc_best_solution, hc_improvement_trace

def reorder(order, orig_colour_list):

    new_order = []
    for i in range (len(order)):
        new_order[i].append(orig_colour_list[order[i]])

    return new_order # re orderes colour list RGB values

# perturbation function. A perturb is a large mutation intended to kickstart hill climb out of a local minima.
# input: number, the number of swaps/how sever a perturb you would like
#        sol, a solution/colour ordering
# output: the pertubed solution/ordering
def perturb(sol, number):
    perturb_sol = sol[:] # make copy of solution

    for i in range(number): # for the number passed in do
        perturb_sol = random_swap(perturb_sol) # perform a swap of two random positions

    return perturb_sol

# iterated local search function. runs hill climbing for passed in interations but applies a perturbation which is a larger change so the normal mutation to attempt
# to kick start a solution out of local minima
# Inputs: ils_iterations, the number of iterations to repeat the perturbation process
#        hc iterations, the number of iterations to run in the hill climb method. aka the number of different colour orderes to try
#        method_choice, the mutation method being swap, inversion or scramble
#        cols, the list of colours from the selected file
# Output: mhc_best_solution, the colours ordering with the best evaluation value
def ils(ils_iterations, hc_iterations, method_choice, cols):
    ils_trace = []
    ran_sol = random_sol(len(cols))

    ils_best_solution, hc_improve_trace = hill_climbing(hc_iterations, method_choice, cols, ran_sol)  # call hill climbing function for given iterations and method
    mhc_best_solution_eval = evaluate(cols, ils_best_solution)  # evaluate the given solution
    ils_trace.append(mhc_best_solution_eval)
    for i in range(ils_iterations):  # for mhc repetitions do...

        perturb_order = perturb(ils_best_solution, 3)

        current_solution, hc_improve_trace = hill_climbing(hc_iterations, method_choice, cols, perturb_order)  # call hill climbing function for given iterations and method
        current_solution_eval = evaluate(cols, current_solution)  # evaluate the given solution

        if current_solution_eval < mhc_best_solution_eval:  # if new solution is better than the currnet best solution then
            ils_best_solution = current_solution  # store the current solution as the best solution
            mhc_best_solution_eval = current_solution_eval  # store the current evaluation value as the best value
        ils_trace.append(current_solution_eval)
    return ils_best_solution, ils_trace

# evaluate bext method function

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


print('\nGenerating ils 100 solution...')
ils_best100, ils_trace= ils(5, 20000, "inversion", colors100) # Run multi hill climb on 100. Include either "swap", "inversion" or "scramble"
plot_colors(colors100, ils_best100, 20)
print('\nmhc_best100:', ils_best100)
ils_100_eval = evaluate(colors100, ils_best100)# evaluate the solution
print(f'Evaluation of order ils 100: {ils_100_eval}')  # Displaying all decimals
print(f'Evaluation of order ils 100: {np.round(ils_100_eval, 4)}')  # rounding to display only 4 decimals. This is better for display

plt.figure()
plt.suptitle('HC trace 100')
plt.plot(ils_trace)
plt.ylabel("Evaluation Value")
plt.xlabel("ils iteration")
plt.show()

my_best100 = [1, 62, 27, 71, 41, 11, 95, 42, 34, 77, 16, 54, 31, 0, 24, 72, 18, 46, 51, 35, 44, 81, 93, 19, 66, 39, 55, 23, 49, 74,
              48, 22, 94, 84, 78, 50, 37, 28, 43, 60, 80, 6, 63, 89, 64, 73, 3, 15, 82, 2, 8, 79, 58, 75, 86, 69, 47, 52, 67, 29, 68,
              76, 40, 70, 10, 91, 25, 56, 5, 17, 32, 92, 83, 59, 57, 9, 4, 61, 85, 21, 38, 26, 88, 65, 87, 90, 30, 45, 7, 13, 20, 53, 36, 14, 33, 12] # 15.9079, 3 20000

#######################################################################################################################

order500 = list(range(ncolors500))  # list of consecutive numbers from 0 to ncolors
print('\nUnordered solution 500...')
plot_colors(colors500, order500, 40)  # You will notice that colors are not ordered in the file
unordered_500_eval = evaluate(colors500, order500)# evaluate the solution
print(f'Evaluation of unordered 500: {unordered_500_eval}')  # Displaying all decimals
print(f'Evaluation of unordered 500: {np.round(unordered_500_eval, 4)}')  # rounding to display only 4 decimals. This is better for display


print('\nGenerating multi hill climb 500 solution...')
ils_best500, ils_500trace= ils(3, 100000, "inversion", colors500) # Include either "swap", "inversion" or "scramble"
plot_colors(colors500, ils_best500, 40)
print('\nmhc_best500:', ils_best500)
ils_500_eval= evaluate(colors500, ils_best500)# evaluate the solution
print(f'Evaluation of order ils 500: {ils_500_eval}')  # Displaying all decimals
print(f'Evaluation of order ils 500: {np.round(ils_500_eval, 4)}')  # rounding to display only 4 decimals. This is better for display

my_best500 = [91, 389, 222, 56, 416, 220, 258, 70, 231, 345, 436, 133, 353, 139, 75, 58, 297, 457, 164, 447, 427, 491, 196, 426, 406, 79, 271, 157, 27, 106, 162, 299, 108, 298, 42, 185, 312, 34, 302, 399, 404, 344, 1, 437, 138, 377, 219, 189, 235, 301, 122, 421, 432, 476, 349, 262, 484, 159, 25, 462, 155, 53, 171, 36, 33, 12, 314, 291, 357, 430, 5, 343, 453, 338, 456, 83, 374, 477, 126, 347, 224, 466, 253, 319, 365, 208, 270, 148, 102, 17, 351, 392, 479, 167, 206, 336, 92, 32, 268, 264, 435, 460, 195, 388, 156, 147, 242, 333, 471, 158, 325, 217, 239, 444, 376, 24, 72, 18, 358, 46, 255, 442, 44, 245, 410, 35, 137, 51, 209, 141, 332, 172, 8, 197, 370, 490, 276, 115, 286, 81, 411, 103, 93, 19, 384, 251, 478, 480, 396, 2, 469, 82, 247, 15, 318, 327, 230, 134, 136, 109, 452, 391, 292, 232, 127, 371, 236, 326, 191, 78, 373, 101, 207, 160, 263, 313, 9, 369, 61, 4, 244, 295, 57, 240, 320, 454, 211, 3, 468, 153, 308, 246, 331, 218, 73, 315, 203, 441, 64, 407, 89, 324, 380, 274, 346, 362, 179, 63, 266, 403, 168, 60, 80, 354, 6, 486, 305, 422, 459, 383, 43, 114, 45, 272, 281, 364, 194, 125, 378, 188, 29, 132, 129, 322, 174, 420, 428, 165, 449, 96, 14, 348, 76, 352, 337, 360, 152, 68, 342, 256, 340, 267, 67, 321, 52, 38, 228, 238, 21, 259, 85, 335, 210, 121, 59, 287, 439, 334, 213, 306, 204, 192, 117, 66, 11, 100, 401, 104, 446, 41, 71, 450, 402, 144, 425, 48, 387, 472, 74, 180, 49, 372, 285, 463, 199, 355, 438, 233, 193, 356, 323, 394, 22, 151, 23, 55, 39, 260, 166, 250, 111, 465, 212, 309, 163, 128, 330, 461, 359, 382, 303, 495, 311, 202, 488, 367, 84, 50, 37, 227, 448, 150, 492, 386, 30, 198, 280, 28, 451, 223, 339, 475, 146, 408, 225, 184, 413, 110, 375, 97, 261, 118, 90, 296, 279, 87, 107, 120, 130, 398, 481, 65, 169, 145, 149, 229, 88, 131, 283, 405, 143, 273, 409, 241, 140, 390, 186, 417, 483, 214, 381, 269, 366, 176, 385, 226, 94, 494, 181, 293, 62, 350, 445, 154, 113, 289, 424, 47, 124, 414, 278, 201, 429, 317, 77, 175, 16, 254, 290, 252, 288, 237, 310, 170, 434, 328, 173, 99, 397, 412, 473, 54, 205, 0, 423, 294, 363, 248, 257, 275, 300, 443, 95, 482, 243, 31, 98, 282, 395, 249, 418, 489, 474, 234, 368, 329, 190, 467, 105, 433, 183, 10, 464, 455, 470, 187, 119, 116, 400, 440, 304, 431, 69, 86, 142, 265, 415, 182, 221, 112, 26, 493, 178, 135, 379, 277, 487, 123, 40, 307, 200, 284, 7, 215, 419, 393, 161, 341, 20, 13, 485, 216, 316, 177, 458, 361]
# 54.2256 3, 100000 meaning it is better than multi hill climb as the same iterations but evaluation improvement of 10.
#########################################################################################################################

plt.figure()
plt.suptitle('HC trace 500')
plt.plot(ils_500trace)
plt.ylabel("Evaluation Value")
plt.xlabel("ils iteration")
plt.show()