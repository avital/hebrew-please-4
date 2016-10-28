# Make matplotlib not try to talk with X; librosa always imports
# matplotlib; Once librosa v0.5 is released, this should no longer be
# necessary
import matplotlib
matplotlib.use('Agg')

from model import make_model

import sys
import numpy as np
import numpy.random
from scipy import ndimage
import scipy.interpolate
import random
import socket
import time
import hashlib
import librosa
import matplotlib.pyplot as plt
import frequency_stats

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard

# Work around version mismatch between TensorFlow and Keras.
# See https://github.com/fchollet/keras/issues/3857
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

import os

def listwavfiles(path):
    """
    Returns a list of full paths to .wav files in a given directory
    """
    return [os.path.abspath(os.path.join(path, file))
            for file in os.listdir(path)
            if file.endswith('.wav')]

samples = {
    1: listwavfiles('./data2/speech-english'),
    0: listwavfiles('./data2/speech-hebrew')
}

def add_gaussian_noise(spectrogram, scale):
    """
    'spectrogram' is a real-valued spectrogram (abs of complex STFT).

    Add random Guassian noise on all frequency bins
    """
    return np.abs(spectrogram + np.random.normal(0, scale, spectrogram.shape))

def stretch(spectrogram, factor, num_columns):
    """
    'spectrogram' is a real-valued spectrogram (abs of complex STFT).

    Stretch the spectrogram by 'factor' via linear interpolation,
    then return the first 'num_columns' columns
    """

    stretched = np.zeros((spectrogram.shape[0], num_columns))

    for column in xrange(num_columns):
        fractional_column_in_source = column / factor

        # Find two closest columns and their weight for linear interpolation
        column1_in_source = int(fractional_column_in_source)
        column2_in_source = column1_in_source + 1
        column2_in_source_weight = fractional_column_in_source - column1_in_source
        column1_in_source_weight = 1 - column2_in_source_weight

        # Compute the new column
        stretched[:, column] = (
            spectrogram[:, column1_in_source] * column1_in_source_weight +
            spectrogram[:, column2_in_source] * column2_in_source_weight
        )

    return stretched

def center_unit(spectrogram):
    '''
    Apply a linear transformation on the spectrogram that
    emperically gets each entry to have a distribution with
    mean 0 and standard deviation 1
    '''

    return ((spectrogram.T - frequency_stats.means[0:185]) / frequency_stats.stds[0:185]).T

def normalize_spectrogram(spectrogram):
    spectrogram2 = spectrogram + 0.00001
    return spectrogram2 / np.sum(spectrogram2, 0)

# Given a 2-dimensional array, return an array with smaller
# dimensions. Specifically, skip every (stride1-1) elements in
# dimension 1 and (stride2-1) elements in dimension 2
def reduce_dimensions(arr, stride1, stride2):
    return arr[0:arr.shape[0]:stride1, 0:arr.shape[1]:stride2]

def main():
    def data_generator():
        batch_size = 128
        while True:
            random.seed(time.time())
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                sample = random.choice(samples[label])
                sample_length = librosa.core.get_duration(filename=sample) # XXX precompute

                sample_duration = 3
                offset_start = random.uniform(0, sample_length*0.8-sample_duration)
                sample_segment_data, sr = librosa.core.load(
                    sample, sr=11025, offset=offset_start, duration=sample_duration
                )
                abs_spectrogram = np.absolute(librosa.stft(sample_segment_data, n_fft=512))[0:185, :]

                # Normalize first, so we can pick a good noise distribution
                normalized_abs_spectrogram = normalize_spectrogram(abs_spectrogram)

                # Add noise 40% of the time
                if random.uniform(0, 1) < 0.4:
                    noise_factor = random.uniform(0, 0.01)
                    noisy_abs_spectrogram = add_gaussian_noise(
                        normalized_abs_spectrogram, noise_factor
                    )
                else:
                    noisy_abs_spectrogram = normalized_abs_spectrogram

                stretched_abs_spectrogram = stretch(
                    noisy_abs_spectrogram, random.uniform(0.7, 1.42), 173
                )

                # Then normalize again after adding noise (since sums
                # of columns have changed)
                normalized_abs_spectrogram_2 = normalize_spectrogram(np.absolute(stretched_abs_spectrogram))
                centered_unit_abs_spectrogram = center_unit(normalized_abs_spectrogram_2)

                reduced_spectrogram = reduce_dimensions(centered_unit_abs_spectrogram, 3, 3)
                sample_segment_spectrogram = np.expand_dims(reduced_spectrogram, axis=0)
                batch_data.append(sample_segment_spectrogram)
                batch_labels.append(label)

            yield (np.stack(batch_data), batch_labels)

    def make_validation_data(nb_samples):
        random.seed(1337)
        data = []
        labels = []

        for i in xrange(nb_samples):
            label = random.choice([0, 1])
            sample = random.choice(samples[label])
            sample_length = librosa.core.get_duration(filename=sample)
            offset_start = random.uniform(sample_length*0.8, sample_length-2)
            sample_segment_data, sr = librosa.core.load(sample, sr=11025, offset=offset_start, duration=2)
            sample_segment_spectrogram = np.expand_dims(
                reduce_dimensions(
                    center_unit(
                        normalize_spectrogram(
                            np.absolute(
                                librosa.stft(sample_segment_data, n_fft=512)
                            )[0:185, :]
                        )
                    ), 3, 3
                ), axis=0
            )
            data.append(sample_segment_spectrogram)
            labels.append(label)

        return (np.stack(data), labels)

    # model.load_weights('weights.hdf5')

    nb_val_samples = 512
    val_data = make_validation_data(nb_val_samples)

    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=3000,
        validation_data=val_data,
        callbacks=[
            ModelCheckpoint("weights.hdf5", monitor="val_acc", save_best_only=True),
#            EarlyStopping(monitor="val_acc", patience=8),
            TensorBoard(log_dir='/mnt/nfs/HebrewPleaseSimple1',
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

