import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp

csvPath = r"C:\Users\andre\PycharmProjects\StockPrediction\Prediction\AAPL.csv"
ohlcArray, technicIndicatorsNorm, nextDayOpenNorm, nextDayOpen, openNormaliser = mp.createDataSet(csvPath)
'''
print("OHLC SHAPE: ", ohlcArray.shape)
print("Indicator SHAPE: ", technicIndicatorsNorm.shape)
print("NextDayOpenNormalised SHAPE: ", nextDayOpenNorm.shape)
print("NextDayOpenOriginal SHAPE: ", nextDayOpen.shape)
'''

#prints the beginning of the datasets
#print("OHLC data: ", ohlcArray)
#print("Technical Indicator Data: ", technicIndicatorsNorm[:2])
#print("Next Day Open Nomralised Data: ", nextDayOpenNorm)
#print("Next Day open Actual data: ", nextDayOpen[:2])

#seperates the data into train data and test data
testSize = .005
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


#prints shape of all the data
print("Ohlc SHAPE: ", ohlcTrain.shape)
print("Indicator SHAPE: ", technicTrain.shape)
print("NextDayOpenNormalised SHAPE: ", nextNormTrain.shape)
print("NextDayOpenOriginal SHAPE: ", nextOpenTrain.shape)


#Making two seperate models will allow us to have ohlc data not affect the techIndicators until the final model
#Model that will use the LSTM
ohlcInput = layers.Input(shape=(ohlcTrain.shape[1], 5))
#LSTM layer
ohlcLSTMLayer = layers.LSTM(50)(ohlcInput)
#creates dropout to prevent overfitting and add it to be the next layer in the model
ohlcLSTMLayer = layers.Dropout(rate=0.2)(ohlcLSTMLayer)
#creates first model to be concatenated
ohlcModel = keras.Model(inputs=ohlcInput, outputs=ohlcLSTMLayer)


#Model for the technical indicators
technicInput = layers.Input(shape= (technicTrain.shape[1],))
#dense layer for technical side
technicLayer = layers.Dense(20)(technicInput)
#
technicLayer = layers.Activation("relu")(technicLayer)
#technicLayer = layers.Activation("relu")(technicLayer)
#dropout layer to prevent overfitting
technicLayer = layers.Dropout(rate=0.2)(technicLayer)
#creates second model for the technicIndicator
technicModel = keras.Model(inputs=technicInput, outputs=technicLayer)


mergedModel = layers.concatenate([ohlcModel.output, technicModel.output])
#uses sigmoid activation to make sure the outputs are in between 0 and 1
mergedModel = layers.Dense(64, activation= 'sigmoid')(mergedModel)
mergedModel = layers.Activation('relu')(mergedModel)
#mergedModel = layers.Dropout(rate=0.2)(mergedModel)
#mergedModel = layers.Dense(64, activation= 'sigmoid')(mergedModel)
mergedModel = layers.Dense(1, activation= 'linear')(mergedModel)

finalModel = keras.Model(inputs=[ohlcInput, technicInput], outputs=mergedModel)

adam = keras.optimizers.Adam(lr=0.0005)
finalModel.compile(optimizer= adam, loss="mse")
finalModel.fit([ohlcTrain, technicTrain], nextNormTrain, epochs=100, batch_size=32, shuffle= True, verbose=1)





finalModel.save(f"finalSavedModel.h5")

predicted = finalModel.predict([ohlcTest, technicTest])

#print(predicted)
predicted = openNormaliser.inverse_transform(predicted)
#print(predicted)
#assert nextDayOpen.shape == predicted.shape, "They be different shapes"


realMeanSquaredError = np.mean(np.square(nextOpenTest - predicted))
scaledMeanSquaredError = realMeanSquaredError / (np.max(nextOpenTest) - np.min(nextOpenTest)) * 100
print(scaledMeanSquaredError)



import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(nextOpenTest[start:end],'o', label='real')
pred = plt.plot(predicted[start:end],"o", label='predicted')

'''
# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')
'''

plt.legend(['Real', 'Predicted'])
plt.show()


