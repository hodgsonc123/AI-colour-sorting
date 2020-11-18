import numpy as np      # Numerical library, used keeing the list of colours and computing the Euclidean distance

# Read in the color data file
# Input: string with file name
# Oputput: the number of colours (integer), and a list numpy arrays with all the colours
def read_data(fname):
    cols = np.loadtxt(fname, skiprows = 4) # The first 4 lines have text information, and are ignored
    ncols = len(cols)     # Total number of colours and list of colours
    return ncols,cols

import matplotlib.pyplot as plt

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

import random as rnd


# Function to generate a random solution (random ordering)  - we can generate a random ordering of the list by using
# the shuffle function from the random library
def random_sol(ncolors):
    sol = list(range(ncolors))  # list of consequtive numbers from 0 to ncolors
    # Shuffle the elements in the list randomly. Shuffles in place and doesn’t retunr a value
    rnd.shuffle(sol)
    return sol

# This is an auxiliary function. It calculate the Euclidean distance between two individual colours
# Input: v and u as to be numpy arrays, vectors of real numbers with the RGB coordinates.
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


# greedy constructive heuristic function. takes in a list of colours, selects a random point for the beginning of a greedy ordering array
# then searches the list of colours to find the closest colour match (shortest euclidean distance) to that colour, checking it is not already in the sorted array
# This colours index is appended in the greedy_ordering array, the process repeats until we are left with the greedy ordering
# input: original_colour_values_array, array of colours stored as RGB values
# output: greedy_ordering, the greedy ordering of colours
def greedy(original_colour_values_array):

    greedy_ordering = [] # initialise empty array to store greedy ordering
    greedy_ordering.append(rnd.randint(0, len(original_colour_values_array)-1)) # select a random start point in the array

    for i in range(len(original_colour_values_array) - 1): # for the length of the colour array -1 times do...
        current_colour_index = greedy_ordering[i] # set the current colour index to the current position in the greedy ordering array
        dist_to_closest_col = 1000 # distance greater than any possible distance
        closest_colour_index = 0 # init variable to store the closest colour index

        for j in range(len(original_colour_values_array)): # for the length of the colours array do...
            current_colour = original_colour_values_array[current_colour_index] # get the RGB values of the current colour

            if j != current_colour_index and j not in greedy_ordering: # if j is not  the current colour index and has not been sorted already do
                next_colour = original_colour_values_array[j] # set the next colour to be the colour as position j in the colour array
            else: # otherwise, continue the for loop
                continue

            dist_to_next_col = euclid(current_colour, next_colour) # get the distance between the current colour and the next colour

            if dist_to_next_col < dist_to_closest_col: # if this distance is smaller that the current smallest
                closest_colour_index = j # store position j as the closest colour index
                dist_to_closest_col = dist_to_next_col # and store this distance as the new closest distance

        greedy_ordering.append(closest_colour_index) # append the closest colour that was found. (process will be repeated on this colour)

    return greedy_ordering

ncolors100, colors100 = read_data("col10.txt")  # pass in 100 file to reading function
ncolors500, colors500 = read_data("col500.txt")  # pass in 500 file to reading function

# Plot all the colors in the order they are listed in the file
order100 = list(range(ncolors100))  # list of consecutive numbers from 0 to ncolors
print('\nUnordered colours from file 100...')
plot_colors(colors100, order100, 20)  # You will notice that colors are not ordered in the file
unordered_evaluation = evaluate(colors100, order100)
print(f'Evaluation of unordered 100: {unordered_evaluation}')  # Displaying all decimals
print(f'Evaluation of unordered 100: {np.round(unordered_evaluation, 4)}')  # rounding to display only 4 decimals. This is better for display

print('\nGenerating greedy solution 100...')
greedy_sol = greedy(colors100)
print('Greedy sol 100', greedy_sol)
plot_colors(colors100, greedy_sol, 20)
greedy_evaluation = evaluate(colors100, greedy_sol)
print(f'Evaluation of order 100, greedy: {greedy_evaluation}')  # Displaying all decimals
print(f'Evaluation of order 100, greedy: {np.round(greedy_evaluation, 4)}')  # rounding to display only 4 decimals. This is better for display

my_best100 = [11, 41, 71, 27, 62, 94, 78, 50, 37, 28, 84, 87, 90, 30, 45, 7, 13, 20, 53, 36, 25, 56, 5, 17, 32, 92, 70, 10, 58, 75, 86, 69, 47, 52, 67, 29, 68, 76, 40, 14, 33, 12, 91, 54, 31, 24, 72, 18, 46, 51, 44, 35, 81, 93, 19, 66, 23, 55, 39, 22, 48, 49, 74, 1, 88, 26, 38, 21, 85, 4, 9, 57, 59, 83, 3, 64, 89, 63, 80, 60, 43, 6, 73, 15, 82, 2, 8, 79, 95, 42, 34, 77, 16, 0, 61, 65]
# 17.49

########################################################################################################################################################################################################################################

print('\nGenerating greedy solution 500...')
greedy_sol = greedy(colors500)
print('Greedy sol 500', greedy_sol)
plot_colors(colors500, greedy_sol, 40)
greedy_evaluation = evaluate(colors500, greedy_sol)
print(f'Evaluation of order 500, greedy: {greedy_evaluation}')  # Displaying all decimals
print(f'Evaluation of order 500, greedy: {np.round(greedy_evaluation, 4)}')  # rounding to display only 4 decimals. This is better for display

my_best500 = [316, 485, 216, 215, 13, 341, 20, 161, 393, 419, 7, 284, 200, 307, 40, 123, 487, 277, 361, 431, 304, 69, 86, 436, 133, 353, 75, 58, 297, 457, 306, 213, 240, 295, 57, 160, 207, 101, 292, 391, 236, 371, 232, 127, 134, 230, 327, 318, 15, 82, 247, 233, 438, 355, 199, 48, 144, 425, 402, 55, 23, 151, 472, 387, 22, 111, 39, 401, 100, 11, 446, 41, 71, 265, 142, 112, 26, 149, 145, 140, 390, 84, 367, 488, 273, 409, 241, 169, 481, 65, 229, 296, 90, 118, 87, 279, 223, 339, 85, 259, 335, 379, 21, 178, 493, 135, 4, 9, 313, 263, 121, 59, 439, 334, 231, 345, 433, 183, 10, 325, 158, 329, 368, 474, 234, 282, 395, 185, 42, 312, 289, 113, 415, 182, 221, 283, 131, 88, 405, 143, 202, 359, 303, 311, 382, 495, 398, 130, 120, 107, 448, 227, 150, 492, 198, 475, 146, 408, 451, 37, 50, 28, 369, 373, 78, 191, 326, 176, 366, 269, 381, 214, 494, 181, 157, 450, 271, 104, 27, 62, 293, 350, 445, 154, 1, 437, 344, 404, 399, 302, 138, 377, 175, 77, 219, 189, 235, 301, 421, 122, 429, 317, 424, 47, 124, 476, 432, 52, 321, 67, 267, 256, 342, 68, 152, 360, 337, 76, 348, 165, 449, 349, 262, 484, 159, 291, 314, 12, 33, 14, 96, 177, 458, 119, 187, 470, 464, 455, 190, 467, 105, 217, 239, 471, 156, 388, 141, 209, 46, 18, 72, 24, 294, 31, 423, 0, 363, 243, 98, 54, 473, 205, 99, 328, 173, 237, 310, 170, 434, 201, 278, 414, 91, 389, 222, 416, 220, 258, 462, 25, 155, 53, 36, 171, 56, 357, 430, 5, 453, 343, 338, 148, 102, 17, 32, 92, 336, 147, 206, 167, 351, 392, 479, 70, 456, 319, 365, 208, 270, 253, 466, 83, 374, 287, 324, 380, 362, 346, 274, 179, 89, 407, 64, 315, 441, 203, 347, 126, 477, 3, 153, 308, 468, 331, 246, 218, 73, 266, 168, 63, 452, 109, 136, 403, 6, 486, 43, 114, 383, 459, 184, 30, 386, 261, 97, 375, 110, 340, 238, 228, 38, 417, 483, 186, 61, 244, 204, 192, 426, 197, 370, 490, 276, 115, 480, 478, 384, 19, 93, 103, 81, 411, 286, 117, 396, 2, 491, 196, 427, 447, 320, 454, 211, 469, 193, 323, 356, 394, 463, 285, 372, 49, 180, 74, 128, 330, 309, 212, 385, 465, 226, 94, 163, 106, 461, 162, 299, 108, 298, 34, 397, 412, 288, 254, 290, 252, 16, 400, 440, 116, 435, 264, 460, 195, 376, 444, 255, 300, 95, 248, 482, 257, 275, 137, 51, 410, 245, 44, 442, 332, 172, 8, 333, 242, 406, 164, 79, 443, 418, 489, 249, 139, 210, 45, 272, 364, 281, 194, 378, 188, 29, 132, 125, 352, 413, 280, 422, 60, 80, 354, 305, 225, 250, 166, 260, 66, 251, 35, 358, 268, 224, 428, 420, 174, 322, 129]
#51...