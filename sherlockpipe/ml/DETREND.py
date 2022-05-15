import os
import pathlib

import numpy as np
import keras
import tensorflow as tf
import pandas as pd
from keras import Model
from sklearn.utils import shuffle


class AutoEncoder():
    def __init__(self, input_size=20610):
        self.input_size = input_size

    def build(self):
        # (flux, centroidx, centroidy, motionx, motiony, bck)
        input = keras.Input(shape=(self.input_size, 6))
        autoencoder_layer1_strides = 10
        autoencoder_layer1_filters = 5000
        autoencoder_layer1_ks = 100
        autoencoder_layer2_strides = 5
        autoencoder_layer2_filters = 1250
        autoencoder_layer2_ks = 33
        autoencoder_layer3_strides = 5
        autoencoder_layer3_filters = 420
        autoencoder_layer3_ks = 15
        autoencoder_layer4_strides = 2
        autoencoder_layer4_filters = 128
        autoencoder_layer4_ks = 9
        autoencoder_layer5_strides = 2
        autoencoder_layer5_filters = 64
        autoencoder_layer5_ks = 7
        autoencoder_layer6_strides = 1
        autoencoder_layer6_filters = 32
        autoencoder_layer6_ks = 5
        autoencoder_layer7_strides = 1
        autoencoder_layer7_filters = 16
        autoencoder_layer7_ks = 3
        self.enc_layer1 = keras.layers.SpatialDropout1D(rate=0.2)(input)
        self.enc_layer1_r = keras.layers.Conv1D(filters=autoencoder_layer1_filters, kernel_size=autoencoder_layer1_ks, padding="same", activation="relu")(self.enc_layer1)
        self.enc_layer1 = keras.layers.MaxPooling1D(pool_size=50, strides=autoencoder_layer1_strides, padding="same")(self.enc_layer1_r)
        self.enc_layer1 = keras.layers.Dropout(rate=0.1)(self.enc_layer1)
        self.enc_layer2_r = keras.layers.Conv1D(filters=autoencoder_layer2_filters, kernel_size=autoencoder_layer2_ks, padding="same", activation="relu")(self.enc_layer1)
        self.enc_layer2 = keras.layers.MaxPooling1D(pool_size=20, strides=autoencoder_layer2_strides, padding="same")(self.enc_layer2_r)
        self.enc_layer2 = keras.layers.Dropout(rate=0.1)(self.enc_layer2)
        self.enc_layer3_r = keras.layers.Conv1D(filters=autoencoder_layer3_filters, kernel_size=autoencoder_layer3_ks, padding="same", activation="relu")(self.enc_layer2)
        self.enc_layer3 = keras.layers.MaxPooling1D(pool_size=15, strides=autoencoder_layer3_strides, padding="same")(self.enc_layer3_r)
        self.enc_layer3 = keras.layers.Dropout(rate=0.1)(self.enc_layer3)
        self.enc_layer4_r = keras.layers.Conv1D(filters=autoencoder_layer4_filters, kernel_size=autoencoder_layer4_ks, padding="same", activation="relu")(self.enc_layer3)
        self.enc_layer4 = keras.layers.MaxPooling1D(pool_size=10, strides=autoencoder_layer4_strides, padding="same")(self.enc_layer4_r)
        self.enc_layer4 = keras.layers.Dropout(rate=0.1)(self.enc_layer4)
        self.enc_layer5_r = keras.layers.Conv1D(filters=autoencoder_layer5_filters, kernel_size=autoencoder_layer5_ks, padding="same", activation="relu")(self.enc_layer4)
        self.enc_layer5 = keras.layers.MaxPooling1D(pool_size=4, strides=autoencoder_layer5_strides, padding="same")(self.enc_layer5_r)
        self.enc_layer5 = keras.layers.Dropout(rate=0.1)(self.enc_layer5)
        self.enc_layer6_r = keras.layers.Conv1D(filters=autoencoder_layer6_filters, kernel_size=autoencoder_layer6_ks, padding="same", activation="relu")(self.enc_layer5)
        self.enc_layer6 = keras.layers.MaxPooling1D(pool_size=3, strides=autoencoder_layer6_strides, padding="same")(self.enc_layer6_r)
        self.enc_layer6 = keras.layers.Dropout(rate=0.1)(self.enc_layer6)
        self.enc_layer7_r = keras.layers.Conv1D(filters=autoencoder_layer7_filters, kernel_size=autoencoder_layer7_ks, padding="same", activation="relu")(self.enc_layer6)
        self.enc_layer7 = keras.layers.MaxPooling1D(pool_size=2, strides=autoencoder_layer7_strides, padding="same")(self.enc_layer7_r)
        self.enc_layer7 = keras.layers.Dropout(rate=0.1)(self.enc_layer7)
        self.dec_layer7 = keras.layers.UpSampling1D(autoencoder_layer7_strides)(self.enc_layer7)
        self.dec_layer7 = keras.layers.Conv1DTranspose(filters=autoencoder_layer7_filters, kernel_size=autoencoder_layer7_ks, padding="same")(self.enc_layer7)
        self.dec_layer6 = keras.layers.UpSampling1D(autoencoder_layer6_strides)(self.dec_layer7)
        self.dec_layer6 = keras.layers.Conv1DTranspose(filters=autoencoder_layer6_filters, kernel_size=autoencoder_layer6_ks, padding="same")(self.dec_layer6)
        self.dec_layer6 = keras.layers.Add()([self.enc_layer6_r, self.dec_layer6])
        self.dec_layer5 = keras.layers.UpSampling1D(autoencoder_layer5_strides)(self.dec_layer6)
        self.dec_layer5 = keras.layers.Conv1DTranspose(filters=autoencoder_layer5_filters, kernel_size=autoencoder_layer5_ks, padding="same")(self.dec_layer5)
        self.dec_layer5 = keras.layers.Add()([self.enc_layer5_r, self.dec_layer5])
        self.dec_layer4 = keras.layers.UpSampling1D(autoencoder_layer4_strides)(self.dec_layer5)
        self.dec_layer4 = keras.layers.Cropping1D(cropping=(0, 1))(self.dec_layer4)
        self.dec_layer4 = keras.layers.Conv1DTranspose(filters=autoencoder_layer4_filters, kernel_size=autoencoder_layer4_ks, padding="same")(self.dec_layer4)
        self.dec_layer4 = keras.layers.Add()([self.enc_layer4_r, self.dec_layer4])
        self.dec_layer3 = keras.layers.UpSampling1D(autoencoder_layer3_strides)(self.dec_layer4)
        self.dec_layer3 = keras.layers.Cropping1D(cropping=1)(self.dec_layer3)
        self.dec_layer3 = keras.layers.Conv1DTranspose(filters=autoencoder_layer3_filters, kernel_size=autoencoder_layer3_ks, padding="same")(self.dec_layer3)
        self.dec_layer3 = keras.layers.Add()([self.enc_layer3_r, self.dec_layer3])
        self.dec_layer2 = keras.layers.UpSampling1D(autoencoder_layer2_strides)(self.dec_layer3)
        self.dec_layer2 = keras.layers.Cropping1D(cropping=2)(self.dec_layer2)
        self.dec_layer2 = keras.layers.Conv1DTranspose(filters=autoencoder_layer2_filters, kernel_size=autoencoder_layer2_ks, padding="same")(self.dec_layer2)
        self.dec_layer2 = keras.layers.Add()([self.enc_layer2_r, self.dec_layer2])
        self.dec_layer1 = keras.layers.UpSampling1D(autoencoder_layer1_strides)(self.dec_layer2)
        self.dec_layer1 = keras.layers.Conv1DTranspose(filters=autoencoder_layer1_filters, kernel_size=autoencoder_layer1_ks, padding="same")(self.dec_layer1)
        self.dec_layer1 = keras.layers.Add()([self.enc_layer1_r, self.dec_layer1])
        self.linear_proj = keras.layers.GlobalAveragePooling1D()(self.dec_layer1)
        self.linear_proj = keras.layers.Dense(self.input_size, activation="relu")(self.linear_proj)
        self.linear_proj = keras.layers.Dropout(rate=0.1)(self.linear_proj)
        self.linear_proj = keras.layers.Dense(self.input_size, activation="softmax")(self.linear_proj)
        self.model = Model(inputs=input, outputs=self.linear_proj)
        return self

    def inform(self):
        keras.utils.vis_utils.plot_model(self.model, "detrend_autoencoder_resnet.png", show_shapes=True)
        self.compile("adam", "binary_crossentropy")
        self.model.summary()
        return self

    def compile(self, optimizer, loss):
        self.model.compile(optimizer=optimizer, loss=loss)

    def prepare_training_data(self, training_dir, train_percent=0.8, test_percent=0.2):
        lc_filenames = [str(file) for file in list(pathlib.Path(training_dir).glob('*_lc.csv'))]
        lc_filenames.sort()
        lc_filenames = shuffle(lc_filenames)
        train_last_index = int(self.input_size * train_percent)
        test_last_index = train_last_index + int(self.input_size * test_percent)
        dataset_length = len(lc_filenames)
        test_last_index = test_last_index if test_last_index < dataset_length else dataset_length
        return lc_filenames[0:train_last_index], lc_filenames[train_last_index:test_last_index]

    def train(self, training_dir, batch_size, epochs, dataset_iterations_per_epoch=1, train_percent=0.8,
              test_percent=0.2):
        train_filenames, test_filenames = self.prepare_training_data(training_dir, train_percent, test_percent)
        training_batch_generator = AutoencoderGenerator(train_filenames, batch_size, self.input_size)
        validation_batch_generator = AutoencoderGenerator(test_filenames, batch_size, self.input_size)
        train_dataset_size = len(train_filenames)
        test_dataset_size = len(test_filenames)
        self.model.fit(x=training_batch_generator,
                       steps_per_epoch=int(dataset_iterations_per_epoch * train_dataset_size // batch_size),
                       epochs=epochs, verbose=1,
                       validation_data=validation_batch_generator,
                       validation_steps=int(test_dataset_size // batch_size))


class AutoencoderGenerator(tf.keras.utils.Sequence):
    def __init__(self, lc_filenames, batch_size, input_size):
        self.lc_filenames = lc_filenames
        self.batch_size = batch_size
        self.input_size = input_size

    def __len__(self):
        return (np.ceil(len(self.lc_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_filenames = self.lc_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        filenames_shuffled = shuffle(batch_filenames)
        batch_data_array = np.empty((len(filenames_shuffled), self.input_size, 6))
        batch_data_values = np.empty((len(filenames_shuffled), self.input_size))
        i = 0
        for file in filenames_shuffled:
            input_df = pd.read_csv(file, comments="#", usecols=['flux', 'flux_err', 'centroid_x', 'centroid_y',
                                                                'motion_x', 'motion_y', 'bck_flux'], low_memory=True)
            values_df = pd.read_csv(file, comments="#", usecols=['eb_model', 'bckeb_model', 'planet_model'],
                                    low_memory=True)
            batch_data_array[i] = input_df.to_numpy()
            values_df['model'] = (1 - values_df['eb_model']) + (1 - values_df['bckeb_model']) + (1 - values_df['planet_model'])
            batch_data_values[i] = values_df['model'].to_numpy()
            i = i + 1
        return batch_data_array, batch_data_values

auto_encoder = AutoEncoder().build().inform()
auto_encoder.train("/mnt/DATA-2/ete6/lcs/", 200, 50)

