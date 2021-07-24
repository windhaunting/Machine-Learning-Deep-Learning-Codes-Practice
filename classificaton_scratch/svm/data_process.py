# -*- coding: utf-8 -*-



#
import pandas as pd



def read_file_features():
    # for binary classifier, so we 
    # read filefeature
    
    input_file = 'Iris.csv'

    df = pd.read_csv(input_file)
    df = df.drop(['Id'],axis=1)   # drop the first id to get feature only
    
    target = df['Species']   # target as label
    s = set()
    for val in target:
        s.add(val)
    s = list(s)
    
    # drop Iris-virginica class for binary classification
    rows = list(range(100,150))
    df = df.drop(df.index[rows])      # drop index
    
    return df
    
    
if __name__ == "__main__":

    read_file_features()