# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
# changeeeee

import numpy as np     # Numerical library, used keeing the list of colours and computing the Euclidean distance
import matplotlib.pyplot as plt
import random as rnd

# Read in the color data file
# Input: string with file name
# Oputput: the number of colours (integer), and a list numpy arrays with all the colours
def read_data(fname):
    cols = np.loadtxt(fname, skiprows = 4) # The first 4 lines have text information, and are ignored
    ncols = len(cols)     # Total number of colours and list of colours
    return ncols,cols

ncolors, colors = read_data("col500.txt")

print(f'Number of colours: {ncolors}')
print("First 5 colours:")
print(colors[0:5,  :])


# Dsiplay the colors as a strip of color bars
# Input: list of colors, order of colors, and height/ratio

def plot_colors(col_list, col_order, ratio = 10):
    assert len(col_list) == len(col_order)
    img = np.zeros((ratio, len(col_list), 3))
    for i in range(0, len(col_list)):
        img[:, i, :] = col_list[col_order[i]]
    fig, axes = plt.subplots(1, figsize=(10,6)) # figsize=(width,height) handles window dimensions
    axes.imshow(img, interpolation='nearest')
    axes.axis('off')
    plt.show()


# Plot all the colors in the order they are listd in the file
order1 = list(range(ncolors))   # list of consequtive numbers from 0 to ncolors
plot_colors(colors, order1)    #  You will notice that colors are not ordered in the file

# Function to generate a random solution (random ordering)  - we can generate a random ordering of the list by using
# the shuffle function from the random library
def random_sol():
    sol = list(range(ncolors))   # list of consequtive numbers from 0 to ncolors
    # Shuffle the elements in the list randomly. Shuffles in place and doesn’t retunr a value
    rnd.shuffle(sol)
    return sol

order2 = random_sol()
print("Another random solution: ", order2)
plot_colors(colors, order2)  # the colors are not ordered, but this is a different order

# You can test different ratios of the hight/width of the lines in the plot
print("Same ordering of colurs with a larger ratio")
plot_colors(colors, order2, 20)



def euclid(v, u):
    return np.linalg.norm(v - u)

# Evaluation function.  Measures the quality of a given solution (ordering of colours)
# The function computes the sum of the distances between all consecutive colours in the ordering
# Input: cols: list of colours
#        ordc: ordering of colours
# Output: real number with the sumf of pair-wise differences in the colour ordering

def evaluate(cols, ordc):
    adjacentColPairs = [[cols[ordc[i]],cols[ordc[i-1]]] for i in range(1,len(ordc))]
    return sum([euclid(i[1], i[0]) for i in adjacentColPairs])


e1 = evaluate(colors, order1)
print(f'Evaluation of order1: {e1}') # Displaying all decimals
print(f'Evaluation of order1: {np.round(e1,4)}') # rounding to display only 4 decimals. This is better for display

e2 = evaluate(colors, order2)
print(f'Evaluation of order1: {e2}') # Displaying all decimals
print(f'Evaluation of order1: {np.round(e2,4)}') # rounding to display only 4 decimals. This is better for display