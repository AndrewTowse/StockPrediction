import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp

csvPath = r"C:\Users\andre\PycharmProjects\StockPrediction\Prediction\AAPL.csv"
ohlcArray, technicIndicatorsNorm, nextDayOpenNorm, nextDayOpen, openNormaliser = mp.createDataSet(csvPath)

print("OHLC SHAPE: ", ohlcArray.shape)
print("Indicator SHAPE: ", technicIndicatorsNorm.shape)
print("NextDayOpenNormalised SHAPE: ", nextDayOpenNorm.shape)
print("NextDayOpenOriginal SHAPE: ", nextDayOpen.shape)


#seperates the data into train data and test data
testSize = .10
trainSize = 1 - testSize
trainSize = int(ohlcArray.shape[0] * trainSize)


#seperating the data into respective sizes
#Training Data
ohlcTrain = ohlcArray[:trainSize]
technicTrain = technicIndicatorsNorm[:trainSize]
nextNormTrain = nextDayOpenNorm[:trainSize]
nextOpenTrain = nextDayOpen[:trainSize]

#Test Data
ohlcTest = ohlcArray[trainSize:]
technicTest = technicIndicatorsNorm[trainSize:]
nextNormTest = nextDayOpenNorm[trainSize:]
nextOpenTest = nextDayOpen[trainSize:]

#Sequential Model
#NOT DOING THIS NOW

#MODEL
print("Ohlc SHAPE: ", ohlcTrain.shape)
print("Indicator SHAPE: ", technicTrain.shape)
print("NextDayOpenNormalised SHAPE: ", nextNormTrain.shape)
print("NextDayOpenOriginal SHAPE: ", nextOpenTrain.shape)

#input layers
#this input is the shape for the 5 values for the amount of days specified
inputOHLC = layers.Input(input_shape=(ohlcTrain.shape[1], 5))
inputIndicator = layers.Input(input_shape=1)





