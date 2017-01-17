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

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard, Callback

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

yt_samples = {
    1: listwavfiles('./data3/yt-english'),
    0: listwavfiles('./data3/yt-hebrew')
}

# Validation samples from other YouTube videos
yt_val_samples = {
    1: listwavfiles('./data3/yt-english-val'),
    0: listwavfiles('./data3/yt-hebrew-val')
}

# Unlabelled wide-language-index samples for domain adaptation
wli_unlabelled_samples = listwavfiles('./data3/wli-unlabelled')

# Labelled wide-language-index samples for validation
wli_val_samples = {
    1: listwavfiles('./data3/wli-english-val'),
    0: listwavfiles('./data3/wli-hebrew-val')
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

def random_offset(audio_duration, segment_duration, start_decile, end_decile):
    """Split 'audio_duration' seconds into even "parts" of approximate
    minute length.  Return a random initial offset which will be used
    to extract a segment 'segment_duration' seconds long. The segment
    is guaranteed to be in the [start_decile, end_decile) decile of
    the part.

    """
    audio_duration = float(audio_duration)
    num_parts = int(audio_duration / 60)
    part_length = audio_duration / num_parts
    decile_length = part_length / 10
    # Make sure we have enough random choices available. '*2' just a heuristic.
    assert segment_duration*2 <= decile_length, (segment_duration, decile_length)

    chosen_part = random.randint(0, num_parts-1)
    chosen_offset_in_part = random.uniform(
        start_decile*decile_length,
        end_decile*decile_length-segment_duration
    )

    chosen_offset = chosen_part*part_length + chosen_offset_in_part
    assert(chosen_offset >= 0)
    assert(chosen_offset < audio_duration-segment_duration)
    return chosen_offset

def make_validation_data(nb_samples, samples, make_offset, with_domains=True):
    random.seed(1337)
    data = []
    labels = []

    for i in xrange(nb_samples):
        label = random.choice([0, 1])
        sample = random.choice(samples[label])
        audio_duration = librosa.core.get_duration(filename=sample)
        sample_duration = 2
        offset_start = make_offset(audio_duration, sample_duration)
        sample_segment_data, sr = librosa.core.load(
            sample, sr=11025, offset=offset_start, duration=sample_duration
        )
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

    labels = np.array(labels)

    if with_domains:
        return (
            {'input': np.stack(data)},
            {'predicted_label': labels, 'predicted_domain': np.zeros(labels.shape)}, # output
            {'predicted_label': np.ones(labels.shape), 'predicted_domain': np.zeros(labels.shape)} # weights
        )
    else:
        return (np.stack(data), labels)

class AdditionalValidation(Callback):
    def __init__(self, label_model):
        self.label_model = label_model

    def on_train_begin(self, logs={}):
        nb_val_samples = 256

        self.yt_val_data = make_validation_data(
            nb_val_samples,
            yt_val_samples,
            make_offset = lambda audio_duration, sample_duration: random.uniform(
                0, audio_duration-sample_duration
            ),
            with_domains=False
        )
        self.params['metrics'].extend(['yt_val_acc', 'yt_val_loss'])

        self.wli_val_data = make_validation_data(
            nb_val_samples,
            wli_val_samples,
            make_offset = lambda audio_duration, sample_duration: random.uniform(
                0, audio_duration-sample_duration
            ),
            with_domains=False
        )
        self.params['metrics'].extend(['wli_val_acc', 'wli_val_loss'])

    def on_epoch_end(self, batch, logs={}):
        (logs['yt_val_loss'], logs['yt_val_acc']) = self.label_model.evaluate(*self.yt_val_data)
        (logs['wli_val_loss'], logs['wli_val_acc']) = self.label_model.evaluate(*self.wli_val_data)

def main():
    def data_generator():
        batch_size = 128
        while True:
            random.seed(time.time())
            batch_data = []
            batch_labels = []
            batch_labels_w = []
            batch_domains = []
            batch_domains_w = []

            for i in xrange(batch_size):
                yt_or_wli = random.choice(['yt', 'wli'])
                if yt_or_wli == 'yt':
                    domain = 1 # YouTube domain
                    label = random.choice([0, 1])
                    label_weight = 1
                    sample = random.choice(yt_samples[label])
                else:
                    domain = 0 # wide-language-index domain
                    label = 1 # unimportant; weight is 0
                    label_weight = 0
                    sample = random.choice(wli_unlabelled_samples)

                audio_duration = librosa.core.get_duration(filename=sample) # XXX precompute

                sample_duration = 3
                offset_start = random_offset(audio_duration, sample_duration, 0, 8)
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
                batch_labels_w.append(label_weight)
                batch_domains.append(domain)
                batch_domains_w.append(1)

            yield ({'input': np.stack(batch_data)}, # input
                   {'predicted_label': np.array(batch_labels), 'predicted_domain': np.array(batch_domains)}, # output
                   {'predicted_label': np.array(batch_labels_w), 'predicted_domain': np.array(batch_domains_w)} # weights
)

    # model.load_weights('weights.hdf5')

    nb_val_samples = 256
    val_data = make_validation_data(
        nb_val_samples,
        yt_samples,
        make_offset=lambda audio_duration, sample_duration: random_offset(
            audio_duration, sample_duration, 8, 10
        )
    )

    (model, label_model) = make_model()
    json_string = label_model.to_json()
    open('architecture.json', 'w').write(json_string)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=1024,
        nb_epoch=3000,
        validation_data=val_data,
        callbacks=[
            AdditionalValidation(label_model),
            ModelCheckpoint("weights.hdf5", monitor="val_acc", save_best_only=True),
#            EarlyStopping(monitor="val_acc", patience=8),
            TensorBoard(log_dir='/mnt/nfs/HebPlz2017-yt-to-wli-DANN',
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

