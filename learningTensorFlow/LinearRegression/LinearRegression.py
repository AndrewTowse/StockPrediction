import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style




data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

#predict is also known as a label
#label is what you are predicting or what you are looking for
predict = "G3"

#data.drop("attribute")gets rid of the G3 attribute from the dataset
x = np.array(data.drop([predict], 1))
y = np.array(data[predict])


#taking all of our attributes and labels and splitting them up into four different arrays
#x_train is going to be a section of x array#y_train is going to be a section of y array
#x_test and y_test  is test data that is used to test the accuracy of the model we are going to create
#if we trained the model off the entire dataset it would simply memorize the pattern
#the 0.1 for test_size is splitting up 10% of our sample data so that the program can train off
# of the data then predict and we can see how it did
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

'''
#This loop trains the program 30 times to get the best model
#because just taking the first model could be poor
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x,y, test_size = 0.1)

    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)

    acc = linear.score(x_test, y_test)

    print(acc)

    if acc > best:
        #this code stores the regression model in a file to be opened because we don't want to have to learn each time
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
        best = acc

'''


pickleRead = open("studentmodel.pickle", "rb")
linear = pickle.load(pickleRead)

print("Coefficients: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range (len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

#makes grid look half decient in matplotlib
style.use("ggplot")

#setup scatterplot
p = "G1"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
