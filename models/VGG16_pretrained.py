#Keras VGG16 network with input dimensions (3,120,120) -> color mode = rgb (3 color channels)


from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras import backend as K

# create the base pre-trained model with imagenet weights
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=None, input_shape=(3,120,120))

#input_shape=(1, 120, 120)

x = base_model.output

#pooling + fully connected + binary classification layers
#x = GlobalAveragePooling2D()(x)
#x = Dense(1024, activation='relu')(x)
#predictions = Dense(2, activation='softmax')(x)

# classification layers
x = Flatten(name='flatten')(x)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dense(4096, activation='relu', name='fc2')(x)
predictions = Dense(1, activation='sigmoid', name='predictions')(x)

# combined model
model = Model(input=base_model.input, output=predictions)

# freeze all layers in base VGG16 model, only train classifier on top
#for layer in base_model.layers:
    #layer.trainable = False

model.compile(optimizer='nadam', loss='binary_crossentropy',metrics=['binary_accuracy'])

# save model for training
model.save('VGG16[r2].h5')



