import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp
import matplotlib.pyplot as plt
import bs4
import requests
import joblib
import moneyAccount as ma

def predictNextDay(path, normaliser, modelInputs):
    print("-------------Predicting next day-------------")
    #loads the model used to predict
    loadedModel = keras.models.load_model(path)

    #predicts what the next open will be using the input list
    predicted = loadedModel.predict(modelInputs)

    #changes the 0-1 value into its actual value for the actual price
    predicted = normaliser.inverse_transform(predicted)
    print("Predicted price: ", predicted, "\n")
    return predicted


#method specifically for retrieving only the most recent days data for prediction
def retrieveData(path, dlURL):
    print("-------------Retrieving Data-------------")

    print("Downloading csv file")
    #retrieves csv file from the day
    r = requests.get(dlURL)
    url_content = r.content
    csv_file = open(path, 'wb')
    csv_file.write(url_content)
    csv_file.close()


    print("Manipulating the Dataset")
    #calls manipulate_dataset and splits up the returned data set into what we need to predict the next day
    ohlcArray, smaIndicators, macdIndicators, emaIndicators, nextDayOpenNorm, nextDayOpen, openNormaliser = mp.createDataSet("AAPL.csv", 25)

    #sets the lists to only be the last element but still have the same dimensions
    ohlcArray = ohlcArray[-1:]
    smaIndicators = smaIndicators[-1:]
    emaIndicators = emaIndicators[-1:]
    macdIndicators = macdIndicators[-1:]

    data = pd.read_csv(path)
    # creates just values so no slice error?
    data = data.values

    #changes the values in 0-1 to their actual scaled values and sets lastClose equal to the most recent close price
    lastClose = data[-1][4]

    print("Last Close: ", lastClose, "\n")


    #creates the list of inputs for prediction and to be returned
    inputs = [ohlcArray, smaIndicators, emaIndicators, macdIndicators]

    return inputs, openNormaliser, lastClose


#method that will decide whether to buy, sell, or keep a stock
def determineBuy(predicted, lastClose):
    print("-------------Determining buy-------------")
    buy = False

    tempDiff = predicted - lastClose
    percentIncrease = tempDiff/lastClose

    if lastClose < predicted:
        buy = True

    shares, money = ma.getAccount()



    if(buy):
        ma.buy(money, lastClose)
    else:
        print("Don't Buy.")

    return buy


def morningSell():
    shares, money = ma.getAccount()
    open = getPrice()
    ma.sell(shares, money, open)

    return


def getPrice():
    r = requests.get('https://finance.yahoo.com/quote/AAPL?p=AAPL')
    stockPrice = bs4.BeautifulSoup(r.text, 'html')
    current = stockPrice.find('div', {'class': 'D(ib) Mend(20px)'}).find('span').text

    print(current)

    return float(current)