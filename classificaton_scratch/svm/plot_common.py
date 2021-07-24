#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 17:04:20 2020

@author: fubao
"""



# plot visualization for exploratory and display result

import pandas as pd
import matplotlib.pyplot as plt
from data_process import read_file_features

def plot_feature_label(df):
    
    # Id,SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
    print ("df: ", df)
    #x = df['SepalLengthCm']
    #y = df['PetalLengthCm']
    
    # extract SepalLengthCm values when class is Iris-setosa
    setosa_x = df.loc[df['Species'] == 'Iris-setosa', 'SepalLengthCm']

    # extract PetalLengthCm values when class is Iris-setosa
    setosa_y = df.loc[df['Species'] == 'Iris-setosa', 'PetalLengthCm']
    
    # extract SepalLengthCm values when class is Iris-versicolor
    versicolor_x = df.loc[df['Species'] == 'Iris-versicolor', 'SepalLengthCm']
    
    # extract PetalLengthCm values when class is Iris-versicolor
    versicolor_y = df.loc[df['Species'] == 'Iris-versicolor', 'PetalLengthCm']
    
    plt.figure(figsize=(8,6))
    plt.scatter(setosa_x,setosa_y,marker='+',color='green')
    plt.scatter(versicolor_x,versicolor_y,marker='_',color='red')
    plt.show()
    
    
    

if __name__ == "__main__":

    df = read_file_features()
    plot_feature_label(df)