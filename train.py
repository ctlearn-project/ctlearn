
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy

from keras.callbacks import CSVLogger

#load saved model
model = load_model('gammaNNv1[nadam,1e-07].h5')

#data augmentation/preprocessing
################################

#processing training data
training_preprocess = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
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
        classes=['gamma','proton'],
        )

#generator for validation data
validation_generator = validation_preprocess.flow_from_directory(
        '/home/bryankim96/Projects/dl_gamma/data/validation',
        target_size=(120, 120),
        color_mode='grayscale',
        batch_size=32,
        class_mode='binary',
        classes=['gamma','proton'],
        )

#train model
############

logger = CSVLogger('gNNv1[nadam,1e-07].log')

history = model.fit_generator(training_generator,samples_per_epoch=10240,nb_epoch=100,callbacks =[logger], validation_data=validation_generator,nb_val_samples=800)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('accuracy[run2].png', bbox_inches='tight')
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.savefig('loss[run2].png', bbox_inches='tight')

#save weights
#############

model.save('gNNv1[nadam,1e-07].h5')


