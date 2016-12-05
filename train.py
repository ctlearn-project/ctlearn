
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

#load saved model
model = load_model('gammaNNv1.h5')

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

history = model.fit_generator(training_generator,samples_per_epoch=1024,nb_epoch=50,validation_data=validation_generator,nb_val_samples=1000)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#save weights
#############

model.save('gammaNNv1.h5')


