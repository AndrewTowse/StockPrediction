import pandas as pd
from sklearn import preprocessing
import numpy as np

csvPath = r"C:\Users\andre\Downloads\AAPL.csv"

pd.set_option("display.max_rows", None, "display.max_columns", None)
np.set_printoptions(threshold=np.inf)

def createDataSet(csv):
    data = pd.read_csv(csv)
    #drops date from the dataset axis parameter means to delete the column of all the dates
    data = data.drop('Date', axis=1)
    #drops Adj close column same way as above
    data = data.drop('Adj Close', axis=1)

    #puts all of the values into 0-1 (normalizing) I think it drops the labels maybe?
    normalise = preprocessing.MinMaxScaler()
    dataNormalised = normalise.fit_transform(data)
    #print(data)

    #ohlc become a list of numpy arrays that contain the rows next to each other equal to dayRange
    dayRange = 50
    ohlcArray = np.array([dataNormalised[i:i + dayRange].copy() for i in range(len(dataNormalised) - dayRange)])

    #numpy array of the last dayRange(50) opening values
    nextOpenNorm = np.array([dataNormalised[:, 0][i + dayRange].copy() for i in range(len(dataNormalised) - dayRange)])
    #expands dimension by 1, by putting each price in its own array inside the nextOpenVals array becoming an array of arrays
    nextOpenNorm = np.expand_dims(nextOpenNorm, -1)


    nextOpen = np.array([data[:, 0][i + dayRange].copy() for i in range(len(data) - dayRange)])
    print(nextOpen)

    def calculateEMA(ohlcList, timePeriod):
        #finds the simple moving average so the first EMA has a previous moving average to use in the formula
        sma = np.mean(ohlcList[:, 3])
        #stores the simple moving average in emaVals
        emaVals = [sma]

        #calculates the weighted multiplier
        k = 2 / (1 + timePeriod)

        #loop that starts timePeriod from the end of the list and calculates EMA
        for i in range(len(ohlcList) - timePeriod, len(ohlcList)):
            #retrieves the closing price at i
            close = ohlcList[i][3]
            #calculates the Exponential Moving Average based off of the previous EMA
            tempEMA = close * k + emaVals[-1] * (1 - k)
            #appends the new EMA to the end of the list to be used in the next iteration for the EMA
            emaVals.append(tempEMA)

        #returns the last value in the array of EMAs because it is the EMA of all the range of data given
        return emaVals[-1]

    #technical indicator is a pattern based  signal
    technicalIndicators = []
    #loop will store all the EMAs or SMAs of the prices
    for currPriceArr in ohlcArray:
        simpleMA = np.mean(currPriceArr[:, 3])
        exponentialMA = calculateEMA(currPriceArr, 50)
        technicalIndicators.append([simpleMA])
        #technicalIndicators.append([simpleMA, exponentialMA])


    return ohlcArray

data = createDataSet(csvPath)
#print(data[:5])