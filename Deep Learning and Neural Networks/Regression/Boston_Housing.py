#Loading the Boston housing dataset

from keras.datasets import boston_housing

(train_data, train_labels),(test_data,test_labels) = boston_housing.load_data()

print(train_data.shape)
print(test_data.shape)

#Normalizing the data

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std 

test_data -= mean
test_data /= std 

#In general, the less training data you have, the worse overfitting will be, and using a small network is one way to mitigate overfitting
#model definition

from keras import models, layers 

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse', metrics=['mae'])
    return model 

#K-Fold validation

import numpy as np

k = 4
num_val_samples = len(train_data) // k 
num_epochs = 500
all_scores = []
all_mae_histories = []

for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_labels[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:1 * num_val_samples],
        train_data[(i+1) * num_val_samples:]],
        axis=0
    )

    partial_train_targets = np.concatenate(
        [train_labels[:1 * num_val_samples],
        train_labels[(i+1) * num_val_samples:]],
        axis=0
    )

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
    epochs=num_epochs, batch_size=1, verbose=0)

    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=1)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)

#Plotting validation scores
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history)+1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('ValidationMAE')
plt.show()
plt.clf()

#Plotting validation scores, excluding the first 10 data points
def smooth_curve(points, factor=0.9):
    smoothed_points=[]
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


#According to this plot, validation MAE stops improving significantly after 80 epochs. Past that point you start overfitting

model = build_model()
model.fit(train_data, train_labels, epochs = 80, batch_size = 16, verbose = 0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_labels)
print(test_mae_score)

#were still off by abou $2,550 according to that score



