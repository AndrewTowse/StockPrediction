import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import manipulate_data_set as mp
import matplotlib.pyplot as plt
import predictionMethods as pm
from datetime import datetime
from threading import Timer
from decimal import Decimal

def buy(money, price):
    m = open('account.txt', 'r+')
    s = open('shares.txt', 'r+')

    remaining = money%price
    shares = int(money/price)

    remaining = str(remaining)
    shares = str(shares)

    m.write(remaining)
    s.write(shares)

    m.close()
    s.close()

    return


def sell(shares, money, currentPrice):
    m = open('account.txt', 'r+')
    s = open('shares.txt', 'r+')
    h = open('accountHistory.txt', 'r+')
    currentHistory = h.readline()


    total = currentPrice * shares
    money = money + total

    shares = 0

    money = str(money)
    shares = str(shares)


    m.write(money)
    s.write(shares)

    print(currentHistory)
    history =", " + money
    h.write(history)

    m.close()
    s.close()




    return


'''
def hold():

    return
'''


def getAccount():
    print("Getting Account Information")
    m = open('account.txt', 'r+')
    money = m.readline()
    s = open('shares.txt', 'r+')
    shares = s.readline()
    print("Money: ", money)
    print("Shares: ", shares)

    money = float(money)
    shares = int(shares)


    m.truncate(0)
    s.truncate(0)
    m.close()
    s.close()

    return shares, money



