# Make matplotlib not try to talk with X; librosa always imports
# matplotlib; Once librosa v0.5 is released, this should no longer be
# necessary
import matplotlib
matplotlib.use('Agg')

from model import make_model

import sys
import numpy
from scipy import ndimage
import random
import socket
import time
import hashlib
import librosa

from keras.callbacks import ProgbarLogger, ModelCheckpoint, EarlyStopping, TensorBoard

import os

def listwavfiles(path):
    """
    Returns a list of full paths to .wav files in a given directory
    """
    return [os.path.abspath(os.path.join(path, file))
            for file in os.listdir(path)
            if file.endswith('.wav')]

def smallhash(str):
    """
    Hashes str into a number in the range [0, 8)
    """
    bighash = hashlib.sha224(str).hexdigest()
    return int(bighash[0], 16) % 8

samples = {
    1: [file for file in listwavfiles('./data/speech') if smallhash(file) != 1],
    0: [file for file in listwavfiles('./data/non-speech') if smallhash(file) != 1]
}

val_samples = {
    1: [file for file in listwavfiles('./data/speech') if smallhash(file) == 1],
    0: [file for file in listwavfiles('./data/non-speech') if smallhash(file) == 1]
}

def normalize_spectrogram(spectrogram):
    spectrogram2 = spectrogram + 0.00001
    return spectrogram2 / numpy.sum(spectrogram2, 0)

def main():
    nb_val_samples = 512

    def data_generator():
        batch_size = 16
        while True:
            random.seed(time.time())
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                sample = random.choice(samples[label])
                sample_length = librosa.core.get_duration(filename=sample)
                offset_start = random.uniform(0, sample_length-2)
                sample_segment_data, sr = librosa.core.load(sample, sr=11025, offset=offset_start, duration=2)
                sample_segment_spectrogram = numpy.expand_dims(normalize_spectrogram(numpy.absolute(librosa.stft(sample_segment_data, n_fft=512))), axis=0)
                batch_data.append(sample_segment_spectrogram)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

    def val_data_generator():
        batch_size = 16
        index_in_val_batch = 0
        while True:
            index_in_val_batch = (index_in_val_batch + batch_size) % nb_val_samples
            random.seed(index_in_val_batch)
            batch_data = []
            batch_labels = []
            for i in xrange(batch_size):
                label = random.choice([0, 1])
                sample = random.choice(val_samples[label])
                sample_length = librosa.core.get_duration(filename=sample)
                offset_start = random.uniform(0, sample_length-2)
                sample_segment_data, sr = librosa.core.load(sample, sr=11025, offset=offset_start, duration=2)
                sample_segment_spectrogram = numpy.expand_dims(normalize_spectrogram(numpy.absolute(librosa.stft(sample_segment_data, n_fft=512))), axis=0)
                batch_data.append(sample_segment_spectrogram)
                batch_labels.append(label)
            yield (numpy.stack(batch_data), batch_labels)

#    model.load_weights('weights.hdf5')

    model = make_model()
    json_string = model.to_json()
    open('architecture.json', 'w').write(json_string)

    model.fit_generator(
        data_generator(),
        samples_per_epoch=2048,
        nb_epoch=3000,
        validation_data=val_data_generator(),
        nb_val_samples=nb_val_samples,
        callbacks=[
            ModelCheckpoint("weights.hdf5"),
            TensorBoard(log_dir='/mnt/nfs/is-speech-12-more-data',
                        histogram_freq=20,
                        write_graph=True)
        ]
    )

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()



