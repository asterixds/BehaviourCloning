from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import random
import scipy.misc
from utils import load
from generator import generator


"""Build and compile the model"""
from keras.models import Sequential
from keras.regularizers import l2
from keras.layers import Flatten,Dense,Convolution2D, MaxPooling2D, Cropping2D, AveragePooling2D,BatchNormalization,Dropout
from keras.layers.core import Lambda
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

model = Sequential()
# Crop the sky and bottom pixels, normalise and reduce dimensionality
model.add(Cropping2D(((80,25),(1,1)), input_shape=[160, 320, 3], name="Crop2D"))
model.add(BatchNormalization(axis=1, name="Normalise"))
model.add(AveragePooling2D(pool_size=(1,4), name="Resize", trainable=False))

# Successively learn through multiple convolutions, relu activations and pooling layers,
model.add(Convolution2D(24, 3, 3, subsample=(2,2), name="Conv1", activation="relu"))
model.add(MaxPooling2D(name="MaxPool1"))
model.add(Convolution2D(48, 3, 3, subsample=(1,1), name="Conv2", activation="relu"))
model.add(MaxPooling2D(name="MaxPool2"))
model.add(Convolution2D(72, 3, 3, subsample=(1,1), name="Conv3", activation="relu"))
model.add(MaxPooling2D(name="MaxPool3"))
model.add(Dropout(0.2, name="Dropout1"))

# Learn the steering angles through 3 fully connected layers
model.add(Flatten(name="Flatten"))
#model.add(Dense(1024, activation="relu", name="FC1"))
model.add(Dense(100, activation="relu", name="FC2"))
model.add(Dense(50, activation="relu", name="FC3"))
model.add(Dense(10, activation="relu", name="FC4"))

# Final Output  of steering angles
model.add(Dense(1, name="Steering", activation='linear'))


"""Load image paths and angle records"""
records = []
records = load('./data/driving_log.csv', records)

"""generator functions for training  and validation sets"""
batch_size = 32
S_train, S_val = train_test_split(records, test_size=0.02)
gen_train = generator(S_train, batch_size=batch_size, augment=True)
gen_val = generator(S_val, batch_size=batch_size)

"""Train model"""
nb_epoch = 8
nb_samples_per_epoch = 20000
nb_val_samples = len(S_val)
learning_rate = 1e-4

"""Train and validate the model using generators"""
model.compile(loss='mse', optimizer=Adam(learning_rate))
print(model.summary())
model.fit_generator(gen_train,
                    samples_per_epoch=nb_samples_per_epoch,
                    validation_data=gen_val,
                    nb_val_samples=nb_val_samples, nb_epoch=nb_epoch,
                    callbacks=[])

"""Save model"""
model.save("model3_07.h5")
print("Saved model")
