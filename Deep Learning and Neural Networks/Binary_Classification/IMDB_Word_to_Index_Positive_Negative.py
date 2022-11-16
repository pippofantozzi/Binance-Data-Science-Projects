from keras import models  
from keras import layers 

#Loading the IMDB Dataset
from keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(f'Training Data 0: {train_data[0]} Training label 0: {train_labels[0]}')

#One-hot encode your lists to turn them into vectors of 0s, and 1s. This would mean, for instance, turning the sequence [3,5] into a 10,000-dimensional
#vector that would be all 0s except for indices 0 and 5, which would be 1s. Then you could use as the first layer in your networks a Dense Layer capable of handling
#floation-point vector data

import numpy as np
def vectorize_sequences(sequences, dimensions=10000):
    results = np.zeros((len(sequences), dimensions))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results

X_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

Y_train = np.asarray(train_labels).astype('float32')
Y_test = np.asarray(test_labels).astype('float32')

#The model definition
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape = (10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

#Because youre facing a binary classification problem and the output of your network is a probability, its best to use the binary_crossentropy
# loss
#Compiling the model
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['accuracy'])

#configuring the optimizer
from keras import optimizers 
model.compile(optimizers.RMSprop(lr=0.001),
loss='binary_crossentropy',metrics=['accuracy'])

#Using custom losses and metrics
from keras import losses,metrics
model.compile(optimizers.RMSprop(learning_rate=0.001),
loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

#Setting aside a validation set
X_val = X_train[:10000]
partial_x_train = X_train[10000:]
y_val = Y_train[:10000]
partial_y_train = Y_train[10000:]

#training your model
model.compile(optimizer = 'rmsprop',
loss = 'binary_crossentropy', metrics=['accuracy'])

history = model.fit(partial_x_train, partial_y_train, 
epochs=20, batch_size=512, validation_data=(X_val, y_val))

history_dict = history.history
print(history_dict.keys())

#plotting the training and validation loss
import matplotlib.pyplot as plt

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#Plotting the training and validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#As you can see, the trianing loss decreases with every epoch, and the trianing accuracy,
#increases with every epoch. Thats what you would expect when running gradient descent optimization
#the quantity youre trying to minimize should be less with every iteration. But that isnt the case for
#the validatio lowss and accuracy, they seem to peak at fourth epoch. This is an example of over-fitting
#youre overoptimization on the training data, and you end up learning representations that are too specific

#In this case, to prevent overfitting, you could stop training after three epochs.

#Retraining a model from scratch

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, Y_test)
print(f'Loss: {results[0]}')
print(f'Accuracy: {results[1]}')
model.predict(x_test)

