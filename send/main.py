#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import os
from numpy import flatnonzero

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

import open_Data
#import Designing_Machine_Learning
import Examples_scikit_learn
import Examples_XGBoost
import Examples_Keras
# import plot_kmeans_digits
import heat_map_clus



# from Designing_Machine_Learning import intro


if __name__ == "__main__":

    # number_of_sample = 200
    using_of_test_data = True

    short_ans = input("Do you wish to use real big data ? (y/n) ")
    if short_ans == "y": using_of_test_data = False

    number_of_sample = "null"
    while not number_of_sample.isdigit():
        number_of_sample = input("Enter how many Instance you want to use (10-801) : ")

    try:
        int(number_of_sample)
        number_of_sample = int(number_of_sample)
    except ValueError:
        print("'Try again")
        sys.exit("aa! errors! Enter integer number next time.")


    if using_of_test_data:
        file_of_data = 'data_Test.csv'
        file_of_labels = 'labels_Test.csv'
    else:
        file_of_data = 'data.csv'
        file_of_labels = 'labels.csv'
    X, Y, Z = open_Data.open(os.path.join(os.path.dirname(__file__), 'data', file_of_data),
                             os.path.join(os.path.dirname(__file__), 'data', file_of_labels),
                             number_of_sample)
    # Supprime zéros et normalization
    X = open_Data.clearing_of_data(X);
    # print(np.mean(X, axis=0))
    # print(np.std(X, axis=0))

    # print(X)
    # print(Y)
    # print(Z)

    short_ans = input("Do you wish to use regression to train the model? (y/n) ")
    if short_ans == "y": Examples_scikit_learn.intro(X, Y)

    short_ans = input("Do you wish to calculate the cross-validation of this model? (y/n) ")
    if short_ans == "y":
        cross_validation = Examples_scikit_learn.cross_validation(X, Y, 10)

    short_ans = input("Do you wish to make regularization of this model? (y/n) ")
    if short_ans == "y":
        regularisation = Examples_scikit_learn.regularisation(X, Y, 20)

    short_ans = input("Do you wish to plot learning curve of this model? (y/n) ")
    if short_ans == "y":
        Examples_scikit_learn.learning_curve(X, Y, 20)

    short_ans = input("Do you wish to search the grid of this model? (y/n) ")
    if short_ans == "y":
        Examples_scikit_learn.grid_search(X, Y, 2)

    short_ans = input("Do you wish to make a clusterizaton of this data? (y/n) ")
    if short_ans == "y":
        Examples_scikit_learn.clusterisation(X, Y, 5, 40)
        # print(X.shape[0], X.shape[1])

    short_ans = input("Do you wish to test PCA on this data? (y/n) ")
    if short_ans == "y":
        X_reduced = Examples_scikit_learn.PCA(X, number_of_sample)
        print(X_reduced.shape)
        # print(X_reduced)
        Examples_scikit_learn.intro(X_reduced, Y)

    short_ans = input("Do you wish to build decision tree? (y/n) ")
    if short_ans == "y":
        tree_dessicion = Examples_XGBoost.intro(X, Y)
        print(tree_dessicion)

    short_ans = input("Do you really like Artificial neural networks? (y/n) ")
    if short_ans == "y":
        Examples_Keras.intro(X, Y)

    short_ans = input("Last execution. Do you wish to plot heat-map of data ? (y/n) ")
    if short_ans == "y":
        mask = Y == 'BRCA'
        pos = flatnonzero(mask)
        X_BRCA = X.iloc[pos]
        Y_BRCA = Y.iloc[pos]
        print(X_BRCA.shape[0], X_BRCA.shape[1])
        heat_map_clus.map(X_BRCA, Y_BRCA)
        heat_map_clus.map(X, Y)


