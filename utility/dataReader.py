from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras import Input
import tensorflow as tf
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import math

import os
# used to shuffle dataset and create folds
def datasetToFolds(data):
    # shuffle data
    data = data.sample(n=len(data),axis=0,ignore_index=True)
    #splice data to 5-fold
    fnum = int(input("Number of Folds : "))
    ffold = []
    batchlen = math.floor(len(data)/fnum)
    rem = len(data) - batchlen * fnum
    for i in range(fnum):
        if(i==fnum-1):
            try :
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index().drop(axis=1,labels=["index"]))
            except:
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen+rem][:].reset_index())
        else:
            try :
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index().drop(axis=1,labels=["index"]))
            except:
                ffold.append(data.iloc[i*batchlen:(i+1)*batchlen][:].reset_index())
    return ffold
# used to read data from csv
def data_reader():
    # read csv
    data = pd.read_csv("F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_A\\utility\\dataset.csv", delimiter=";",low_memory = False)
    class_namesDict = {"sitting" :1, "walking" :2, "standing":3, "standingup":4, "sittingdown":5}
    data.replace(class_namesDict,inplace=True)
    #data.replace({"sitting" :np.array([0,0,0,0,1]), "walking" :np.array([0,0,0,1,0]), "standing":np.array([0,0,1,0,0]), "standingup":np.array([0,1,0,0,0]), "sittingdown":np.array([1,0,0,0,0])},inplace=True)
    # data preprocess
    # x,y,z in [-617, 533]
    j=0
    data_coordinates = []
    for i in range(math.floor(len(data.index))):
        try :
            data_coordinates.append(data.loc[i][["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"]].astype('int64'))
            j+=1
        except :
            data.drop(axis=0,index=j,inplace=True)
    data_coordinates = pd.DataFrame(data=data_coordinates,columns = ["x1","y1","z1","x2","y2","z2","x3","y3","z3","x4","y4","z4"] )
    data_coordinates = (data_coordinates + 617)/(617+533)#data in [0,1]
    data_coordinates = pd.concat([data_coordinates , data.iloc[:math.floor(len(data.index))]["class"]] , axis=1 , join="outer")
    # save to csv file
    data_coordinates.to_csv(path_or_buf="F:\\5oEtos\\EarinoEksamhno\\YpologistikhNohmosunh\\Project_B\\utility\\processedDataset.csv", sep=';')

    return data_coordinates
