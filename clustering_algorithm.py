import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


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


# K means clustering algorithm
# Input: list of colours and k number of clusters
# Output: sorted colour list
def k_means_clustering(col, k):
    # Create a new dataframe to hold the same info as txt file
    df = pd.DataFrame({'R': col[:, 0], 'G': col[:, 1], 'B': col[:, 2]})
    print(df)

    # Provide the number of clusters wanted
    number_of_clusters = k

    # Create the model, setting the number of clusters, fixing the random seed
    kmeans = KMeans(n_clusters=number_of_clusters)
    # Fit the model
    clusters = kmeans.fit_predict(df)

    print(clusters)  # Print the results, cluster membership of each state

    # Add the cluster info to the original dataset as a new column
    df['Cluster'] = clusters  # New column called cluster
    print(df)  # Now the data set has a new column

    # Can see how many items are in each cluster
    print(df.groupby('Cluster').size())

    # print("now")
    # print(df.iloc[0:1, 0:3])

    # Sort the values by their cluster number
    df.sort_values(['Cluster'], inplace=True)
    print(df)

    # Remove the cluster identifier column
    df.drop(['Cluster'], axis=1, inplace=True)

    # Change from dataframe to numpy array
    final_sorted_colors = df.to_numpy()
    print(final_sorted_colors)

    return final_sorted_colors


ncolors, colors = read_data("col500.txt")
print(colors)

sorted_colors = k_means_clustering(colors, 5)

# Printing original order as a colour bars
order1 = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
plot_colors(colors, order1)  # You will notice that colors are not ordered in the file

# Print the new order as a colour chart
order2 = list(range(ncolors))  # list of consecutive numbers from 0 to ncolors
plot_colors(sorted_colors, order2)  # You will notice that colors are not ordered in the file
