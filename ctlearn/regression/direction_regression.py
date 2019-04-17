import matplotlib

matplotlib.use('TkAgg')
from utilities.gen_util import load_generators_diffuse_point
from utilities.training_util import snapshot_training
from keras.layers import Input, Dense
from models.se_DenseNet import SEDenseNetImageNet121

# %%
BATCH_SIZE = 32
machine = 'towerino'
# Load the data
train_gn, val_gn, test_gn, position = load_generators_diffuse_point(
    machine=machine,
    batch_size=BATCH_SIZE,
    want_golden=True,
    want_position=True,
    include_time=True,
    clean=False)

# %% Define the model
print('Loading the Neural Network...')

input_img = Input(shape=(67, 68, 4), name='m1')

denseNet = SEDenseNetImageNet121(input_tensor=input_img, include_top=False, weights=None)

x = denseNet.layers[-1].output
x = Dense(2, name='position', kernel_regularizer='l2')(x)
model = Model(inputs=input_img, output=x)

net_name = 'SEDenseNet121_position_l2'

# %% Train

result = snapshot_training(model=model,
                           machine=machine,
                           train_gn=train_gn, val_gn=val_gn, test_gn=test_gn,
                           net_name=net_name,
                           max_lr=0.0015,
                           epochs=10,
                           snapshot_number=10,
                           task='direction',
                           swa=3
                           )
