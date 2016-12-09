
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#load saved model
model = load_model('NNv1.h5')

#data augmentation/preprocessing
################################

#processing training data
training_preprocess = ImageDataGenerator(
        #horizontal_flip=True,
        #vertical_flip=True,
        )

#processing validation data
validation_preprocess = ImageDataGenerator()

#generator for training data
training_generator = training_preprocess.flow_from_directory(
        '/home/bryankim96/Projects/dl_gamma/data/train',
        target_size=(120, 120),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#generator for validation data
validation_generator = validation_preprocess.flow_from_directory(
        '/home/bryankim96/Projects/dl_gamma/data/validation',
        target_size=(120, 120),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        )

#train model
############

logger = CSVLogger('r3.log')
checkpoint = ModelCheckpoint('r3{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')
earlystoploss = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
earlystopacc = EarlyStopping(monitor='val_binary_accuracy', min_delta=0.1, patience=4, verbose=0, mode='auto')
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)

history = model.fit_generator(training_generator,samples_per_epoch=122976,nb_epoch=20,callbacks =[logger,checkpoint], validation_data=validation_generator,nb_val_samples=30720)

# list all data in history
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy[r3].png', bbox_inches='tight')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss[r3].png', bbox_inches='tight')

#save weights
#############

model.save('NNv1[r3].h5')
