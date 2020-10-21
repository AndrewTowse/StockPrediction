import tensorflow as tf
from tensorflow import keras
import numpy as np

#this makes it so the program does not put the values calculated
#from the neural network output that are between 0 and 1 into scientific notation
#if this is not used we will be returned numbers in scientific notation and they will look like they large than 1
np.set_printoptions(suppress=True)


#imports movie database
data = keras.datasets.imdb

#num_words = 10000 this only takes the 10000 most frequent words
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

#prints the first review which is a list of each word in the review
print(train_data[0])

word_index = data.get_word_index()

#k is key (string) v is value(integer) and it will add a bunch of different keys into the dataset
word_index = {k:(v+3) for k, v in word_index.items()}
#we want all the movie lengths to be the same so PAD adds zeroes to the
# #longest review to not add words but be able to comapre the date
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

#this reverses the order of the tuples because right now it is a list of strings that point to an integer
#we want a list of integers that point to strings
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

#this allows you to size all of the review to be the same length
#this trims the long reviews to be shorter lists and expands the short lists with PAD until 250 is reached
#padding is if you want to fill it before or after the initial data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=250)

#reverses the integers with their string to be readable to humans
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


#THIS section of code is what creates the model
#only need to run this until you are happy with the model
#we saved this model in model.h5 so we can just load it in for faster running
'''
#model
model = keras.Sequential()
#initially creates 10,000 word vectors
#16 dimension for our vector
model.add(keras.layers.Embedding(88000, 16))
#takes the 16 dimensions and decreases its dimensions to be put into the dense layers
model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation='relu'))
#here there is 1 output neuron because we want to know if the review is either good or bad
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#binary_crossentropy works well because we have 2 possible outputs we want
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

#batch_size -> is how many movie reviews we are going to do at a time
#
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

#prints the accuracy and loss of our model
results = model.evaluate(test_data, test_labels)
print(results)

#SAVING THE MODEL #h5 is an extension for a saved model in keras and tensorflow
model.save("model.h5")
'''

def review_encode(s):
        encoded = [1]

        for word in s:
                if word.lower() in word_index:
                    encoded.append(word_index[word.lower()])
                else:
                    encoded.append(2)
        return encoded

#loads model you want to use
model = keras.models.load_model("model.h5")

with open("upReview.txt") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding='post', maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
'''
#takes first review from test data
test_review = test_data[0]
#uses the model we created to predict the first review
predict = model.predict([test_review])
print("Review: ")
#prints the review in words so we can read it
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)
'''




