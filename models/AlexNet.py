
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Nadam, SGD

#model construction
###################

model = Sequential()

#CONV1
model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(1,120,120)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

#CONV2
model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

#CONV3
model.add(Convolution2D(384, 3, 3, border_mode='same'))
model.add(Activation('relu'))

#CONV4
model.add(Convolution2D(384, 3, 3, border_mode='same'))
model.add(Activation('relu'))

#CONV5
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())

#FC1
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

#FC2
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Activation('relu'))

#FC3
model.add(Dense(1))
model.add(BatchNormalization())
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
model.save('AlexNet[r1].h5')


