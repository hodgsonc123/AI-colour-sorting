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

# euclidean distance function. calculates the distance between two colours
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

# greedy constructive heuristic function. takes in a list of colours, selects a random point for the beginning of a new order array
# then searches the list of colours to find the closest colour match (shortest euclidean distance) to that random colour.
# this colour is then entered in the new array of the new greedy ordering and removed from the unordered list. the process repeats until we are left with the best order
# input: original_colour_values_array, array of colours stored as RGB values
#        current_pos, number of colours in the file
# output: best_greedy_ordering, the best greedy ordering

def greedy(original_colour_values_array, ncols, current_pos):

    copy_colour_values_array = original_colour_values_array[:]
    greedy_ordering = []

    # rnd.randint(0, len(original_colour_values_array)-1)
    current_colour = original_colour_values_array[current_pos]

    greedy_ordering.append(current_pos)

    copy_colour_values_array = np.delete(copy_colour_values_array, current_pos, 0)

    closest_colour = (0.0, 0.0, 0.0)

    while len(copy_colour_values_array) != 0:
        distance_to_closest_colour = 1000

        pos = 0

        for i in range(len(copy_colour_values_array)):
            distance_to_current_colour = euclid(current_colour, copy_colour_values_array[i])
            if distance_to_current_colour <= distance_to_closest_colour:
                closest_colour = current_colour
                distance_to_closest_colour = distance_to_current_colour
                pos = i

        for x in range(ncols):
            orig = original_colour_values_array[x]
            copy = copy_colour_values_array[pos]
            if (orig == copy).all():
                greedy_ordering.append(x)
                break

        copy_colour_values_array = np.delete(copy_colour_values_array, pos, 0)
        current_colour = closest_colour
    return greedy_ordering

# multi greedy function. Repeats greedy function for each position in the array as the starting point and returns the best possible greedy ordering.
# input: ncol, number of colours in the file
#        col, array of colours stored as RGB values
# output: best_greedy_ordering, the best greedy ordering
def multi_greedy(ncol, col):
    best_greedy_ordering = []  # array to store the best ordering
    best_eval = evaluate(col, random_sol())  # evaluate a random solution as a starting value for the best evaluation

    for i in range(ncol):  # repeat for the number of colours
        greedy_ordering = greedy(col, ncol, i)  # return greedy ordering for position i as start point
        new_eval = evaluate(colors, greedy_ordering)  # evaluate this greedy ordering
        if new_eval < best_eval:  # if the current evaluation is better than the best
            best_eval = new_eval  # set this as the new best
            best_greedy_ordering = greedy_ordering  # set the current ordering as the best ordering

    return best_greedy_ordering
# ***************************************************************************************************************
ncolors, colors = read_data("col500.txt")  # pass in file to reading function

print(f'Number of colours: {ncolors}')
print("First 5 colours:")
print(colors[0:5, :])  # prints rgb values for first five colours

# Plot all the colors in the order they are listed in the file
order1 = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
print('\nUnordered colours from file...')
plot_colors(colors, order1, 20)  # You will notice that colors are not ordered in the file
e1 = evaluate(colors, order1)
print(f'Evaluation of unordered: {e1}')  # Displaying all decimals
print(f'Evaluation of unordered: {np.round(e1, 4)}')  # rounding to display only 4 decimals. This is better for display

print('\nGenerating greedy solution...')
greedy_sol = greedy(colors, ncolors, rnd.randint(0, ncolors-1))
print('Greedy sol', greedy_sol)
plot_colors(colors, greedy_sol, 20)
e5 = evaluate(colors, greedy_sol)
print(f'Evaluation of order, greedy: {e5}')  # Displaying all decimals
print(f'Evaluation of order, greedy: {np.round(e5, 4)}')  # rounding to display only 4 decimals. This is better for display

print('\nGenerating multi greedy solution...')
multi_greedy_sol = multi_greedy(ncolors, colors)
print('Multi greedy sol', multi_greedy_sol)
plot_colors(colors, multi_greedy_sol, 20)
multi_greedy_evaluation = evaluate(colors, multi_greedy_sol)
print(f'Evaluation of order, multi greedy: {multi_greedy_evaluation}')  # Displaying all decimals
print(f'Evaluation of order, multi greedy: {np.round(multi_greedy_evaluation, 4)}')  # rounding to display only 4 decimals. This is better for display


