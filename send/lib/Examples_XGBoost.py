import xgboost
import pandas as pd

# function constructs decision tree
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer
# function returns cross-tabulation of probability and decision
# function displays roc-curve and AUC
# function plots and saves graph of relative importance of gene
# and finally function create the file of decision tree with feature map
def intro(file_of_data, file_of_labels):
    from sklearn.model_selection import train_test_split

    file_of_labels = (file_of_labels.iloc[:] == 'BRCA')
    X_train, X_test, y_train, y_test = train_test_split(file_of_data, file_of_labels, test_size=0.3, random_state=42)
    #

    # dmat = xgboost.DMatrix(data=file_of_data.values, label=file_of_labels.values)
    dtrain = xgboost.DMatrix(data=X_train, label=y_train)
    dtest = xgboost.DMatrix(data=X_test)
    #
    params = {'objective': 'binary:logistic', 'silent': True}
    bst = xgboost.train(params=params, dtrain=dtrain)
    preds = bst.predict(data=dtest)
    answer = pd.crosstab(y_test, preds)
    # print(pd.crosstab(y_test, preds))
    # print(answer)
    bst.dump_model('output/dump.raw.txt')

    import matplotlib.pyplot as plt

    xgboost.plot_importance(bst)
    # xgboost.plot_tree(bst, num_trees=2)
    plt.rcParams['figure.figsize'] = [50, 10]
    plt.show()

    feature_importance = file_of_data.from_records(list(bst.get_score(importance_type="gain").items()))
    # print(feature_importance)
    # print(type(feature_importance))
    feature_importance.to_csv('output/feature_importance.csv')
    auc(y_test, preds)
    return answer


# function constructs roc-curve
# and calculates AUG
# y_test - test data
# Z - predicted data
def auc(y_test, Z):
    from sklearn.metrics import accuracy_score
    from sklearn import metrics

    y_test_reverce = []
    for i in y_test:
        if i:
            y_test_reverce.append(1)
        else:
            y_test_reverce.append(0)

    Z_reverce = []
    for i in Z:
        if i > 0.5:
            Z_reverce.append(1)
        else:
            Z_reverce.append(0)


    # print(type(y_test_reverce ))
    y_test_reverce = pd.Series(y_test_reverce)
    Z_reverce = pd.Series(Z_reverce)

    y = y_test_reverce
    pred = Z_reverce

    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    print(fpr)
    print(tpr)
    print(metrics.auc(fpr, tpr))

    return 0