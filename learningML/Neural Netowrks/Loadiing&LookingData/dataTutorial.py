import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#imports data set from keras one of keras given datasets
data = keras.datasets.fashion_mnist


#keras makes it easy by having this method to split up data, however when training your own data you will
#have to make your own for loops an d array to sort the data into the right variables
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#there are 10 different labels 0-9
#we must create a list for each number for what it really represents inthe data
class_names = ["T-Shirt/top", "Trouser", "Pullover", "Dress" "Coat",
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
'''
#prints the actual data of each image
#prints each row of the image and their pixel information
#this numbers outputted will be between 0 and 255
print(train_images[7])
'''

#changes the values to be between 0 and 1
#but the data is now shrunk down to be more easily used later
train_images = train_images/255.0
test_images = train_images/255.0

print(train_images[7])


'''
#show an image using matplotlib
plt.imshow(train_images[7])
plt.show()
'''


#this will show the same image as above but without the purple and green colors
plt.imshow(train_images[7], cmap = plt.cm.binary)
plt.show()

