
# function plots heat-map graph
# X, Y - two parameter of graph
def map(X, Y):
    import seaborn as sns;
    sns.set(color_codes=True)
    import matplotlib.pyplot as plt

    species = Y
    lut = dict(zip(species.unique(), ['red', 'green', 'blue', 'yellow', 'cyan']))
    print(lut)
    col_colors = species.map(lut)
    g = sns.clustermap(X.T, col_colors=col_colors)
    plt.show()
    return 0