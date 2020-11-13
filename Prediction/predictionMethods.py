import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp
import matplotlib.pyplot as plt
import bs4
import requests
import re

def predictNextDay(path, ohlc, sma, ema, macd):
    loadedModel = keras.models.load_model(path)
    predicted = loadedModel.predict([ohlc, sma, ema, macd])

    return predicted

def retrieve_ohlcData():
    r = requests.get('https://finance.yahoo.com/quote/AAPL/history?p=AAPL')
    stockPrice = bs4.BeautifulSoup(r.text, 'html')
    current = stockPrice.find('a', {'class' : "Fl(end) Mt(3px) Cur(p)"})

    '''
    curr = []
    counter = 0
    for his in current:
        if counter == 25: break
        if "Dividend" not in his.text:
            temp = his.text[12:]
            print(temp, "\n")
            temp = re.split('\d+', temp, 2)
            print(temp, "\n")
            curr.append(temp)
            counter = counter + 1'''

    return 5