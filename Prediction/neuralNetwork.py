import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp
import joblib


csvPath = "APPL.csv"
ohlcArray, smaIndicators, macdIndicators, emaIndicators, nextDayOpenNorm, nextDayOpen, openNormaliser, lastClose = mp.createDataSet(csvPath, 25)

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
smaTrain = smaIndicators[:trainSize]
emaTrain = emaIndicators[:trainSize]
macdTrain = macdIndicators[:trainSize]
nextNormTrain = nextDayOpenNorm[:trainSize]
nextOpenTrain = nextDayOpen[:trainSize]
lastCloseTrain = lastClose[trainSize:]

#Test Data
ohlcTest = ohlcArray[trainSize:]
smaTest = smaIndicators[trainSize:]
emaTest = emaIndicators[trainSize:]
macdTest = macdIndicators[trainSize:]
nextNormTest = nextDayOpenNorm[trainSize:]
nextOpenTest = nextDayOpen[trainSize:]
lastCloseTest = lastClose[trainSize:]

#prints shape of all the data
print("Ohlc SHAPE: ", ohlcTrain.shape)
print("SMA Indicator SHAPE: ", smaTrain.shape)
print("EMA Indicator SHAPE: ", emaTrain.shape)
print("MACD Indicator SHAPE: ", macdTrain.shape)
print("NextDayOpenNormalised SHAPE: ", nextNormTrain.shape)
print("NextDayOpenOriginal SHAPE: ", nextOpenTrain.shape)







'''
#Making several seperate models will allow us to have ohlc data not affect the techIndicators until the final model
#Model that will use the LSTM
ohlcInput = layers.Input(shape=(ohlcTrain.shape[1], 5))
#LSTM layer
ohlcLSTMLayer = layers.LSTM(50)(ohlcInput)
#creates dropout to prevent overfitting and add it to be the next layer in the model
ohlcLSTMLayer = layers.Dropout(rate=0.2)(ohlcLSTMLayer)
#creates first model to be concatenated
ohlcModel = keras.Model(inputs=ohlcInput, outputs=ohlcLSTMLayer)

################## SMA ###################
#Model for the sma indicators
smaInput = layers.Input(shape= (smaTrain.shape[1],))
#dense layer for technical side
smaLayer = layers.Dense(20)(smaInput)
#
smaLayer = layers.Activation("relu")(smaLayer)
#technicLayer = layers.Activation("relu")(technicLayer)
#dropout layer to prevent overfitting
smaLayer = layers.Dropout(rate=0.2)(smaLayer)
#creates second model for the technicIndicator
smaModel = keras.Model(inputs=smaInput, outputs=smaLayer)
########################################

################## EMA ###################
#Model for the sma indicators
emaInput = layers.Input(shape= (emaTrain.shape[1],))
#dense layer for technical side
emaLayer = layers.Dense(20)(emaInput)
#
emaLayer = layers.Activation("relu")(emaLayer)
#technicLayer = layers.Activation("relu")(technicLayer)
#dropout layer to prevent overfitting
emaLayer = layers.Dropout(rate=0.2)(emaLayer)
#creates second model for the technicIndicator
emaModel = keras.Model(inputs=emaInput, outputs=emaLayer)
########################################

################## MACD ###################
#Model for the sma indicators
macdInput = layers.Input(shape= (macdTrain.shape[1],))
#dense layer for technical side
macdLayer = layers.Dense(20)(macdInput)
#
macdLayer = layers.Activation("relu")(macdLayer)
#technicLayer = layers.Activation("relu")(technicLayer)
#dropout layer to prevent overfitting
macdLayer = layers.Dropout(rate=0.2)(macdLayer)
#creates second model for the technicIndicator
macdModel = keras.Model(inputs=macdInput, outputs=macdLayer)
########################################

mergedModel = layers.concatenate([ohlcModel.output, smaModel.output, emaModel.output, macdModel.input])
#uses sigmoid activation to make sure the outputs are in between 0 and 1
mergedModel = layers.Dense(128, activation= 'sigmoid')(mergedModel)
mergedModel = layers.Activation('relu')(mergedModel)
#mergedModel = layers.Dropout(rate=0.2)(mergedModel)
#mergedModel = layers.Dense(64, activation= 'sigmoid')(mergedModel)
mergedModel = layers.Dense(1, activation= 'linear')(mergedModel)

finalModel = keras.Model(inputs=[ohlcInput, smaInput, emaInput, macdInput], outputs=mergedModel)

adam = keras.optimizers.Adam(lr=0.0005)
finalModel.compile(optimizer= adam, loss="mse")
finalModel.fit([ohlcTrain, smaTrain, emaTrain, macdTrain], nextNormTrain, epochs=100, batch_size=32, shuffle= True, verbose=1)


finalModel.save(f"newAppleSavedModel.h5")
'''
##############################################################################################

finalModel = keras.models.load_model("appleSavedModel.h5")

predicted = finalModel.predict([ohlcTest, smaTest, emaTest, macdTest])

predicted = openNormaliser.inverse_transform(predicted)
#print(predicted)
#assert nextDayOpen.shape == predicted.shape, "They are different shapes"



realMeanSquaredError = np.mean(np.square(nextOpenTest - predicted))
print("Real mean error: ", realMeanSquaredError)
scaledMeanSquaredError = realMeanSquaredError / (np.max(nextOpenTest) - np.min(nextOpenTest)) * 100
print("Scaled mean error: ", scaledMeanSquaredError)
print(predicted)


import matplotlib.pyplot as plt

plt.gcf().set_size_inches(10, 5, forward=True)

start = 0
end = -1


plt.plot(lastCloseTest[start:end],"o", label='realPoint', color = 'green')
plt.plot(nextOpenTest[start:end],"o", label='realPoint', color = 'black')
plt.plot(predicted[start:end],"o", label='predicted points', color= 'red')

#plt.plot(predicted[start:end], label='predicted line', color = 'green')

'''
# real = plt.plot(unscaled_y[start:end], label='real')
# pred = plt.plot(y_predicted[start:end], label='predicted')
'''

plt.legend(['Last close', 'Real next open', 'Predicted next open'])
plt.show()


