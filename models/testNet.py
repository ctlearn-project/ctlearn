
from keras.models import Sequential
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import Nadam, SGD

#model construction
###################

model = Sequential()

# three sets of Convolution layers (32, 32, 64 filters of size 3x3) followed by relu activation filters and
# pooling layers (pooling size 2x2)
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(1,120,120)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#feeding into linear classifier - final layer sigmoid to give log
#probabilities
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
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
model.save('testNet[r4].h5')

#train model
############

#model.fit_generator(training_generator,samples_per_epoch=5000,nb_epoch=50,validation_data=validation_generator,nb_val_samples=1000)

#save weights
#############

#model.save_weights('gammaNNv1RunWeights.h5')

