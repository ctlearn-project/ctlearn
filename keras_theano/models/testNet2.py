
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Nadam, SGD

#model construction
###################

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1,120,120)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#feeding into linear classifier - final layer sigmoid to give log
#probabilities
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#compile model
##############

#optimize with 
#adam gradient descent
#binary crossentropy loss
#accuracy and crossentropy metrics

#sgd = SGD(lr=0.01, clipnorm=1.)
#nadam = Nadam(lr=0.0000002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(optimizer='nadam',loss='binary_crossentropy',metrics=['binary_accuracy'])

#save model
model.save('testNet2[r1].h5')

