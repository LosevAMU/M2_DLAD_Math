# import pandas as pd
from pandas import crosstab
# from numpy import *

# function fits and tests model of logistic regression
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer
# function returns cross-tabulation and accuracy of model
def intro(file_of_data, file_of_labels):
    #import numpy as np
    # import pandas as pd
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score
    from sklearn import metrics

    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(file_of_data, file_of_labels, test_size=0.3, random_state=42)
    # X_train.rows = range(0, 41, 1)

    # Train the model
    logreg.fit(X_train, y_train)
    Z = logreg.predict(X_test)
    # print((logreg.coef_).max(axis=0))

    # Evaluate the model
    # print(pd.crosstab(y_test, Z))
    print(crosstab(y_test, Z))
    #print(type(y_test))
    #print(type(Z))
    print(accuracy_score(y_test, Z))

    # y = y_test
    # pred = Z
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, Z, pos_label=2)
    # print(metrics.auc(fpr, tpr))

# function makes cross-validation of data and model
# function devides training set in 10 folds and train model
# using 9 folds with test on 10-th, repeats 10 times for
# every fold to test.
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer,
# n_folds - number of folds to make the cross-validation
# function returns file with results of cross-validation
def cross_validation(file_of_data, file_of_labels, n_folds):
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score

    X_train_gross, X_test_gross, y_train_gross, y_test_gross = train_test_split(file_of_data,
                                                                                file_of_labels,
                                                                                test_size=0.3,
                                                                                random_state=42)
    # print(len(file_of_data.index))
    X_train_gross.index = range(0, len(X_train_gross.index), 1)
    y_train_gross.index = range(0, len(y_train_gross.index), 1)

    len_of_fragment = len(X_train_gross.index)//n_folds
    # print(len_of_fragment)
    list_of_train = []
    my_reponse = []
    for x in range(n_folds-1):
        list_tmp = range((x * len_of_fragment), ((x+1) * (len_of_fragment)), 1)
        list_of_train.append(list_tmp)
    list_of_train.append(range(((n_folds-1) * len_of_fragment), len(X_train_gross.index), 1))

    f = open('output/cross_validation.txt', 'w')
    f.write('#_of_fold' + '\t' + 'score_of_fold' + '\t' + 'score_of_test' + '\n')
    for x in range(n_folds):
        # print(list_of_train[x])
        X_test = X_train_gross.iloc[(list_of_train[x]), :]
        # print(X_train_gross)
        X_train = X_train_gross.drop(list_of_train[x], axis=0)
        y_test = y_train_gross.iloc[(list_of_train[x])]
        y_train = y_train_gross.drop(list_of_train[x], axis=0)
        #
        logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

        logreg.fit(X_train, y_train)
        Z = logreg.predict(X_test)
        # print(Z[0:20])
        # print(y_test[0:20])
        f.write(str(x+1) + '\t' +
                str(accuracy_score(y_test, Z)) + '\t' +
                str(accuracy_score(y_test_gross, logreg.predict(X_test_gross)))+'\n')

        my_reponse.append([x, accuracy_score(y_test, Z), accuracy_score(y_test_gross, logreg.predict(X_test_gross))])
        print(x, accuracy_score(y_test, Z), accuracy_score(y_test_gross, logreg.predict(X_test_gross)))

    f.close()

    return my_reponse

# function calculates regularisation of model
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer,
# dispersion - dispersion of regularisation
# script runs through interval (-dispersion, +dispersion)
# function returns file with results of regularisation
def regularisation(file_of_data, file_of_labels, dispersion):
    from sklearn.model_selection import train_test_split
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(file_of_data, file_of_labels,
                                                        test_size=0.3,
                                                        random_state=42)
    my_reponse = []
    f = open('output/regularisation.txt', 'w')
    f.write('exp_C' + '\t' + 'score_of_test' + '\n')
    for i in range(-dispersion, dispersion + 1, 1):
        logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial', C=(10**i))

        # Train the model
        logreg.fit(X_train, y_train)
        Z = logreg.predict(X_test)

        # Evaluate the model
        # print(pd.crosstab(y_test, Z))
        f.write(str(i) + '\t' +
                str(accuracy_score(y_test, Z)) + '\n')
        print(i, accuracy_score(y_test, Z))
        my_reponse.append([i, accuracy_score(y_test, Z)])

    f.close()
    return my_reponse

# function plots graph of learning curve
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer,
# pas - number of variables of diagram by Ox
# script runs through interval (-dispersion, +dispersion)
# function returns file with data to plot the graph
def learning_curve(file_of_data, file_of_labels, pas):

    from sklearn import linear_model
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')

    X_train, X_test, y_train, y_test = train_test_split(file_of_data,
                                                        file_of_labels,
                                                        test_size=0.3,
                                                        random_state=42)
    fuse = len(X_train.index)
    # print(fuse)
    len_of_pas = max(1, fuse // pas)
    # X_train.rows = range(0, 41, 1)

    # Train the model
    # logreg.fit(X_train, y_train)
    # Z = logreg.predict(X_test)

    x = []
    train_ac = []
    test_ac = []
    f = open('output/curve.txt', 'w')
    # f = open('output/regularisation.txt', 'w')
    f.write('#_of_samples' + '\t' + 'score_of_train' + '\t' + 'score_of_test' + '\n')

    for i in range(5, len(X_train.index), len_of_pas):
        if i > fuse:
            break
        # print(i)

        logreg.fit(X_train[1:i], y_train[1:i])
        training_accuracy = accuracy_score(logreg.predict(X_train), y_train)
        testing_accuracy = accuracy_score(logreg.predict(X_test), y_test)
        x.append(i)
        train_ac.append(training_accuracy)
        test_ac.append(testing_accuracy)
        f.write(str(i) + '\t' +
                str(training_accuracy) + '\t' +
                str(testing_accuracy) + '\n')
        print(training_accuracy, testing_accuracy)

    import matplotlib.pyplot as plt
    plt.plot(x, train_ac, 'r')  # plotting t, a separately
    plt.plot(x, test_ac, 'b')  # plotting t, b separately
    plt.show()

    f.close()

    return 0

# function calculates one of parameter of model
# here - coefficient of regularisation
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer,
# dispersion - dispersion of regularisation
# script runs through interval (-dispersion, +dispersion)
# function displays the best value of parameter
# and accuracy of model with this parameter
def grid_search(file_of_data, file_of_labels, dispersion):
    from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='multinomial')
    X_train, X_test, y_train, y_test = train_test_split(file_of_data, file_of_labels, test_size=0.3, random_state=42)

    from sklearn.model_selection import GridSearchCV
    parameters = {'C': [10**(-dispersion), 10**dispersion]}
    grid_search = GridSearchCV(estimator=logreg,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10)
    grid_search = grid_search.fit(X_train, y_train)
    print(accuracy_score(grid_search.predict(X_test), y_test))
    print(grid_search.best_params_)

    return 0

# function determines the clusters of data
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer,
# n_group - number of clusters
# r_seed - random seed
# function displays the cross-tabulation of clusterization
def clusterisation(file_of_data, file_of_labels, n_group, r_seed):
    #import matplotlib.pyplot as plt
    #import matplotlib
    import numpy as np
    from sklearn.cluster import KMeans
    np.random.seed(r_seed)

    kmeans = KMeans(n_clusters=n_group)
    kmeans.fit(file_of_data)

    # print(kmeans.cluster_centers_)
    resolution = kmeans.labels_

    u, indices = np.unique(resolution, return_inverse=True)

    print(crosstab(file_of_labels, resolution))

    return 0

# function reduces data using PCA
# file_of_data - data with gene expression,
# survie - number of features after reducing
# function returns reduced data
def PCA(file_of_data, survie):
    from sklearn.decomposition import PCA
    reduced_data = PCA(survie).fit_transform(file_of_data)

    return reduced_data