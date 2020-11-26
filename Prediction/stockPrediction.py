import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp
import matplotlib.pyplot as plt
import predictionMethods as pm
from datetime import datetime
from datetime import date

import calendar
#from threading import timer
import moneyAccount as ma
import threading

def run():
    print("- - - - - - Programming running - - - - - -\n")
    repeat()

    return


def repeat():
    print("-------------Repeat Called-------------\n")

    model_path = "appleSavedModel.h5"
    csvLink = 'https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1574716257&period2=1606338657&interval=1d&events=history&includeAdjustedClose=true'
    csvPath = 'AAPL.csv'

    buy = startPredict(model_path, csvPath, csvLink)

    secondsB = calc_wait_buy()

    tb = threading.Timer(secondsB, repeat)
    tb.start()

    if(buy):
        secondsS = calc_wait_sell()

        ts = threading.Timer(secondsS, pm.morningSell())
        ts.start()
    else:
        print("Not buying waiting for next prediction")

    return




def startPredict(modelPath, csvPath, csvLink):

    print("-------------Starting Prediction-------------\n")
    inputs, normaliser, lastClose = pm.retrieveData(csvPath, csvLink)
    nextDayPredicted = pm.predictNextDay(modelPath, normaliser, inputs)

    buy = pm.determineBuy(nextDayPredicted, lastClose)

    return buy


def calc_wait_buy():
    print("-------------Calculating buy wait time-------------")
    #seconds = 0
    today = date.today().strftime("%A")

    x = datetime.today()
    y = datetime.today()

    #checks if weekend for timer to wait past weekend to call repeat again
    if(today == 'Friday'):
        y = y.replace(day=y.day + 3, hour=16, minute=0, second=0, microsecond=0)
    elif(today == 'Saturday'):
        y = y.replace(day=y.day + 2, hour=16, minute=0, second=0, microsecond=0)
    else:
        y = y.replace(day=y.day + 1, hour=16, minute=0, second=0, microsecond=0)


    print("Today is ", today)
    deltaTime = y-x
    print("Time until next prediction: ", deltaTime, "\n")

    #calculates seconds for timer to start
    days = deltaTime.days
    days = days * 86400
    seconds = deltaTime.seconds
    seconds = seconds + days

    #sets seconds equal to 30 for testing
    #seconds = 30

    return seconds


def calc_wait_sell():
    print("-------------Calculating sell wait time-------------")
    #seconds = 0
    today = date.today().strftime("%A")

    x = datetime.today()
    y = datetime.today()

    #checks if weekend for timer to wait past weekend
    if(today == 'Friday'):
        y = y.replace(day=y.day + 3, hour=9, minute=30, second=0, microsecond=0)
    elif(today == 'Saturday'):
        y = y.replace(day=y.day + 2, hour=9, minute=30, second=0, microsecond=0)
    else:
        y = y.replace(day=y.day + 1, hour=9, minute=30, second=0, microsecond=0)


    deltaTime = y-x
    print("Time until sell: ", deltaTime, "\n")

    #calculates seconds for timer to start
    days = deltaTime.days
    days = days * 86400
    seconds = deltaTime.seconds
    seconds = seconds + days

    #sets seconds equal to 10 for testing
    #seconds = 10

    return seconds




'''
model_path = "appleSavedModel.h5"

csvLink = 'https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1573664651&period2=1605287051&interval=1d&events=history&includeAdjustedClose=true'

csvPath = 'AAPL.csv'
'''

run()



