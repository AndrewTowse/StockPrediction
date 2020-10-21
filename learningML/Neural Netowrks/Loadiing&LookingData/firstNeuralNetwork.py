import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#imports data set from keras one of keras given datasets
from tensorflow.python.keras import Sequential

data = keras.datasets.fashion_mnist


#splits data into different variables
(train_images, train_labels), (test_images, test_labels) = data.load_data()

#sets the class names to array to be translated
class_names = ["T-Shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#divides values of pixels by 255.0 to get them in the range of 0-1
train_images = train_images/255.0
test_images = test_images/255.0

#creates the neural network
model= keras.Sequential([
    #must flatten the 2D array into 1D to pass it to an individual neuron
    keras.layers.Flatten(input_shape=(28,28)),
    #dense means the network will be full connected and we give it 128 neurons
    # the activation function relu is rectify linear unit (fast activation function
    keras.layers.Dense(128, activation='relu'),
    #softmax makes it so the probabilty of all the neurons adds up to 1 so we can
    #see what the network thinks about each class
    keras.layers.Dense(10, activation='softmax'),
])

#tons of different loss functions and optimizers and mettrics
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#epoche randomly picks images and labels
#epoche gives the same images in a different order in this case
#each neuron will see each image 5 times in a random order
#giving epoche doesn't always increase accuracy (tweak and play with it)
model.fit(train_images, train_labels, epochs=5)

#because the networks learns the train data we cannot test its accuracy on the train data
#we must use the test data we set aside to test its accuracy on
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Tested accuaccy: ', test_acc)

prediction = model.predict(test_images)


for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

'''
#np.argmax gets the highest value neuron from a list and give us the index of that neuron
print(class_names[np.argmax(prediction[0])])
'''