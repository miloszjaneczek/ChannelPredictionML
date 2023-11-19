import scipy.io
import numpy as np
import tensorflow as tf
import keras
import os
import copy
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from keras import layers
print(tf.config.list_physical_devices('GPU'))
#tf.debugging.set_log_device_placement(True)

def scheduler(epoch, lr):
    print("learning rate:" + str(lr / 1.2))
    return lr / 1.2

with tf.device('/CPU:0'):
    howManyPreviousChannels = 2
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=[35+34*howManyPreviousChannels]))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0001),
                 loss=tf.keras.losses.MeanAbsolutePercentageError(),
                metrics=['mae'])
    mins = np.empty(((35+(howManyPreviousChannels*34))))
    maxs = np.empty(((35+(howManyPreviousChannels*34))))
    isFirst = True
    for fileno in range(7, 640):
        freq_resp = scipy.io.loadmat("trainingData/freq_resp" + str(fileno) + ".mat")
        gyro = scipy.io.loadmat("trainingData/gyro" + str(fileno) + ".mat")
        index_array = np.arange(3276)
        freq_resp_train = np.empty(((35+(howManyPreviousChannels*34)), 0))
        freq_resp_test = np.empty((32, 0))

        for i in range(freq_resp['freq_response'][1,1,1,:].size-(howManyPreviousChannels+1)):
            #x prep
            freq_resp_train_temp = copy.deepcopy(freq_resp['freq_response'][:,:,:,i])
            freq_resp_train_temp = freq_resp_train_temp.reshape(-1, 3276)
            freq_resp_train_temp_amp = np.abs(freq_resp_train_temp)
            freq_resp_train_temp_phase = np.angle(freq_resp_train_temp)
            for n in range(howManyPreviousChannels):
                freq_resp_train_temp = copy.deepcopy(freq_resp['freq_response'][:,:,:,i+n+1])
                freq_resp_train_temp = freq_resp_train_temp.reshape(-1, 3276)
                freq_resp_train_temp_amp = np.vstack((freq_resp_train_temp_amp, np.abs(freq_resp_train_temp)))
                freq_resp_train_temp_phase = np.vstack((freq_resp_train_temp_phase, np.angle(freq_resp_train_temp)))
            freq_resp_train_temp = np.vstack((freq_resp_train_temp_amp,freq_resp_train_temp_phase, index_array))
            for n in range(howManyPreviousChannels+1):
                freq_resp_train_temp = np.vstack((freq_resp_train_temp, np.tile(gyro['gyro'][:,i+n], (3276,1)).T))
            freq_resp_train = np.append(freq_resp_train, freq_resp_train_temp, axis=1)
            #y prep
            freq_resp_test_temp_prev = copy.deepcopy(freq_resp['freq_response'][:,:,:,i+howManyPreviousChannels])
            freq_resp_test_temp_prev = freq_resp_test_temp_prev.reshape(-1, 3276)
            freq_resp_test_temp = copy.deepcopy(freq_resp['freq_response'][:,:,:,i+howManyPreviousChannels+1])
            freq_resp_test_temp = freq_resp_test_temp.reshape(-1, 3276)
            freq_resp_test_temp_amp = np.abs(freq_resp_test_temp)
            freq_resp_test_temp_amp_prev = np.abs(freq_resp_test_temp_prev)
            freq_resp_test_temp_amp = freq_resp_test_temp_amp - freq_resp_test_temp_amp_prev
            freq_resp_test_temp_phase = np.angle(freq_resp_test_temp)
            freq_resp_test_temp_phase_prev = np.angle(freq_resp_test_temp_prev)
            freq_resp_test_temp_phase = freq_resp_test_temp_phase - freq_resp_test_temp_phase_prev
            freq_resp_test_temp = np.vstack((freq_resp_test_temp_amp, freq_resp_test_temp_phase))
            freq_resp_test = np.append(freq_resp_test, freq_resp_test_temp, axis=1)
        freq_resp_train = freq_resp_train.T
        freq_resp_test = freq_resp_test.T
        #normalization
        for i in range(freq_resp_train[0].size):
        #    freq_resp_train[:,i] = (freq_resp_train[:,i] - freq_resp_train[:,i].min()) / (freq_resp_train[:,i].max() - freq_resp_train[:,i].min())
            if isFirst:
                mins[i] = freq_resp_train[:,i].min()
                maxs[i] = freq_resp_train[:,i].max()
            freq_resp_train[:,i] = (freq_resp_train[:,i] - mins[i]) / (maxs[i] - mins[i])
        print("FILENO: " + str(fileno))
        isFirst = False
        #predicted = model.predict(np.array([freq_resp_train[40],]))
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        history = model.fit(freq_resp_train, freq_resp_test, epochs=5, callbacks=[callback], batch_size=256, shuffle=True)
        model.save("delta.keras")