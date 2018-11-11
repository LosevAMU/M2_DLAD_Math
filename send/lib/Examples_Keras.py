# import xgboost
import pandas as pd
# import keras

# function constructs ANN, learn it and test it
# file_of_data - data with gene expression,
# file_of_labels - data with kind of cancer
# function displays cross-tabulation of classification
# and accuracy on test data
def intro(file_of_data, file_of_labels):
    # from keras.utils import to_categorical
    # from sklearn import linear_model
    from sklearn.model_selection import train_test_split
    from keras.models import Model
    from keras.layers import Dense, Input
    from sklearn.metrics import accuracy_score

    Y_encoded = list()
    # print(file_of_labels)
    for i in file_of_labels:
        if i == 'BRCA':
            Y_encoded.append(0)
        else:
            Y_encoded.append(1)
    # print(Y_encoded)
    from keras.utils import to_categorical
    Y_bis = to_categorical(Y_encoded)
    # print(Y_bis)

    # from sklearn import linear_model

    # logreg = linear_model.LogisticRegression(solver='lbfgs', multi_class='auto')

    X_train, X_test, y_train, y_test = train_test_split(file_of_data, Y_bis, test_size=0.3, random_state=42)

    init = 'random_uniform'
    input_layer = Input(shape=(len(file_of_data.columns),))
    mid_layer = Dense(15, activation='relu', kernel_initializer=init)(input_layer)
    mid_layer_2 = Dense(8, activation='relu', kernel_initializer=init)(mid_layer)
    output_layer = Dense(2, activation='softmax', kernel_initializer=init)(mid_layer_2)

    model = Model(input=input_layer, output=output_layer)
    model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=1)
    Z = model.predict(X_test)

    y_test_reverce = []
    for i in y_test:
        y_test_reverce.append(i[0])
    Z_reverce = []
    for i in Z:
        if i[0] > i[1]:
            Z_reverce.append(1)
        else:
            Z_reverce.append(0)
    # print(type(y_test_reverce ))
    y_test_reverce = pd.Series(y_test_reverce)
    Z_reverce = pd.Series(Z_reverce)

    print(pd.crosstab(y_test_reverce, Z_reverce))
    print(accuracy_score(y_test_reverce, Z_reverce))

    return 0
