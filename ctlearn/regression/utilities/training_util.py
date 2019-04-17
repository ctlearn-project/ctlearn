import pickle
import time

from keras.callbacks import CSVLogger

from utilities.snapshot import SnapshotCallbackBuilder
from utilities.swa import SWA


def snapshot_training(model, train_gn, val_gn, net_name, max_lr=0.01, epochs=10, snapshot_number=5, task='direction',
                      machine='towerino', test_gn=None, swa=1):
    # Compile accordingly
    if task == 'direction':
        model.compile(optimizer='sgd', loss='mse')
    elif task == 'energy':
        model.compile(optimizer='sgd', loss='mse')
    elif task == 'separation':
        model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

    # check the model
    model.summary()

    # Set unique model name based on date-time
    nowstr = time.strftime('%Y-%m-%d_%H-%M-%S')
    net_name_time = f"{net_name}_{nowstr}"

    # Callbacks setup
    snapshot = SnapshotCallbackBuilder(epochs, snapshot_number, max_lr)
    callbacks = snapshot.get_callbacks(model_prefix=net_name_time)

    if swa > -1:
        filename = f'output_data/swa_models/{net_name_time}_SWA.h5'
        swa_callback = SWA(filename, swa)
        callbacks.append(swa_callback)

    logger = CSVLogger(f'output_data/csv_logs/{net_name_time}.csv')
    callbacks.append(logger)

    # Training
    if machine == 'towerino' or machine == 'titanx':
        result = model.fit_generator(generator=train_gn,
                                     validation_data=val_gn,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     use_multiprocessing=False,
                                     workers=8)
        print('Training completed')
    elif machine == '24cores':
        result = model.fit_generator(generator=train_gn,
                                     validation_data=val_gn,
                                     epochs=epochs,
                                     verbose=1,
                                     callbacks=callbacks,
                                     use_multiprocessing=True,
                                     workers=24)

    print('Saving the model...')
    model.save_model(f'/home/emariott/deepmagic/output_data/checkpoints/{net_name_time}.hdf5')
    # Perform Test
    if test_gn is not None:
        print('Predicting test...')
        y_pred_test = model.predict_generator(generator=test_gn,
                                              verbose=1,
                                              use_multiprocessing=False,
                                              workers=8)

        print('Saving predictions...')
        reconstructions_path = f'output_data/reconstructions/{net_name_time}.pkl'
        with open(reconstructions_path, 'wb') as f:
            pickle.dump(y_pred_test, f)
        print('saved')

        # Save the result
        result_path = f'output_data/loss_history/{net_name_time}.pkl'
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)
        print('Result saved')

    return result, y_pred_test
