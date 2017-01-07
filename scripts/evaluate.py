
from keras.models import load_model

#load saved model
model = load_model('gammaNNv1.h5')

# evaluate the model
scores = model.evaluate_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
