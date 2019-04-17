import matplotlib

matplotlib.use('TkAgg')
from utilities.gen_util import load_generators_diffuse_point
from utilities.training_util import snapshot_training
from keras.layers import Input, Dense, BatchNormalization, LeakyReLU
from keras.models import Model
from models.SqueezeExciteInceptionV3 import SEInceptionV3

BATCH_SIZE = 256
machine = 'towerino'

# Load the data
train_gn, val_gn, test_gn, energy = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_energy=True, want_log_energy=True,
    include_time=True,
    clean=False)

# %%
# Create the model
input_img = Input(shape=(67, 68, 4), name='m1m2')
dense_out = SEInceptionV3(include_top=False,
                          weights=None,
                          input_tensor=input_img,
                          input_shape=None,
                          pooling='avg'
                          )
x = dense_out.layers[-1].output

x = BatchNormalization()(x)
x = Dense(64)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dense(1, name='energy', kernel_regularizer='l2')(x)

model = Model(inputs=input_img, output=x)

net_name = 'SE-InceptionV3-energy'

# %%
result = snapshot_training(model=model,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.05,
                           epochs=12,
                           snapshot_number=12,
                           task='energy',
                           machine=machine,
                           swa=2
                           )
