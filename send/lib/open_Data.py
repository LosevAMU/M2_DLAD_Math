
# function opens the files and devides them into three
# variables: X - data, Y - class, Z - index of lines
# file_of_data - file with gene expression,
# file_of_labels - file with kind of cancer
# number_of_sample - number of Instances which we wish to analyze
# function returns three variables : X - data, Y - class, Z - index of lines.
def open(file_of_data, file_of_labels, number_of_sample):
    import pandas as pd

    X_df = pd.read_csv(file_of_data,  nrows=number_of_sample)
    Y_df = pd.read_csv(file_of_labels,  nrows=number_of_sample)

    X = X_df.iloc[:, 1:]
    Y = Y_df.iloc[:, 1]
    Z = Y_df.iloc[:, 0]
    return X, Y, Z

# function removes zero-columns and
# normalize data to have mean = 0 and std = 1
# X_in - table to clean and normalize
# function returns clearned and normalized table
def clearing_of_data(X_in):
    import sklearn
    import pandas as pd

    X_in = X_in.loc[:, (X_in != 0).any(axis=0)]
    # print(X_in)
    X_out = sklearn.preprocessing.scale(X_in, axis=0)
    # print(X_out)
    X_out = pd.DataFrame(X_out)
    X_out.columns = X_in.columns
    # print(X_out)
    return X_out
