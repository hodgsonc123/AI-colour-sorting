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
# this colour is then entered in the new array of the new greedy ordering. the process repeats until we are left with the best order

def greedy1(colors):

    closest_colour = (0.0,0.0,0.0)
    original_colours = colors[:]
    greedy_ordering = []

    start_position = rnd.randint(0, len(colors)-1)
    current_colour = original_colours[start_position]
    print(start_position)

    greedy_ordering.append(start_position)

    original_colours = np.delete(original_colours, start_position, 0)

    while (len(original_colours) != 0):
        shortest_distance = 600
        #print(len(original_colours))
        for colour in original_colours:
            distance = euclid(current_colour, colour)
            #print('distance', distance)
            if distance <= shortest_distance:
                #print('shortest', shortest_distance)
                closest_colour = colour
                shortest_distance = distance

        index = np.where(original_colours == closest_colour)
        print('next: ', closest_colour)
        print('index', index)
        int_index = int(index[0])
        greedy_ordering.append(index)
        print('index 0', index)

        original_colours = np.delete(original_colours, int_index)
        current_colour = closest_colour
    return greedy_ordering

def greedy(original_colour_values_array):

    copy_colour_values_array = original_colour_values_array[:]
    greedy_ordering = []

    current_position = rnd.randint(0, len(original_colour_values_array)-1)
    current_colour = original_colour_values_array[current_position]

    greedy_ordering.append(current_position)

    copy_colour_values_array = np.delete(copy_colour_values_array, current_position, 0)

    closest_colour = (0.0,0.0,0.0)

    while len(copy_colour_values_array) != 0:
        distance_to_closest_colour = 10000

        for i in range(len(copy_colour_values_array)):
            distance_to_current_colour = euclid(current_colour, copy_colour_values_array[i])
            if distance_to_current_colour < distance_to_closest_colour:
                closest_colour = current_colour
                distance_to_closest_colour = distance_to_current_colour

        index = np.where(original_colour_values_array == closest_colour) #this is where is goes wrong
        int_index = int(index[0])

        greedy_ordering.append(int_index)

        copy_colour_values_array = np.delete(copy_colour_values_array, int_index, 0)
        current_colour = closest_colour
    return greedy_ordering
# ***************************************************************************************************************
ncolors, colors = read_data("col10.txt")  # pass in file to reading function

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

greedy_sol = greedy(colors)
print('greedy sol',greedy_sol)
plot_colors(colors, greedy_sol, 40)
e5 = evaluate(colors, greedy_sol)
print(f'Evaluation of order greedy: {e5}')  # Displaying all decimals
print(f'Evaluation of order greedy: {np.round(e5, 4)}')  # rounding to display only 4 decimals. This is better for display
