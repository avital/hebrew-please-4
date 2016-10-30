import pyaudio
from threading import Thread
import curses
import sys
import numpy
import scipy
import scipy.signal as signal
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import math
import uuid
import json
import os
import errno
import wave
import librosa
import subprocess
from keras.models import model_from_json
import multiprocessing
import frequency_stats
import tensorflow as tf
from keras import backend as K

# Work around version mismatch between TensorFlow and Keras.
# See https://github.com/fchollet/keras/issues/3857
import tensorflow
from tensorflow.python.ops import control_flow_ops
tensorflow.python.control_flow_ops = control_flow_ops

audio = pyaudio.PyAudio()

RATE = 11025
CHUNK = 1024
BUFFER_SECS = 2.1
FORMAT=pyaudio.paInt16

def load_model(weights_file):
    model = model_from_json(open('architecture.json').read())
    model.load_weights(weights_file)
    return model

model = load_model('weights.hdf5')

# Store TensorFlow graph. Needed to run inference in a different
# thread.  See
# https://www.tensorflow.org/versions/r0.11/api_docs/python/framework.html#get_default_graph
graph = tf.get_default_graph()

class AudioRecording():
    def start(self):
        stream = audio.open(format=FORMAT,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []
        num_frames = 0

        while True:
            num_frames = num_frames + 1
            data = stream.read(CHUNK)
            frames.append(data)
            if len(frames) > BUFFER_SECS * RATE / CHUNK:
                frames.pop(0)
                if num_frames % 12 == 0:
                    frames_array = b''.join(frames)
                    Thread(
                        target=test_segment,
                        args=(numpy.fromstring(frames_array, dtype=numpy.int16),)
                    ).start()

decayed_prediction = 0.5

def center_unit(spectrogram):
    '''
    Apply a linear transformation on the spectrogram that
    emperically gets each entry to have a distribution with
    mean 0 and standard deviation 1
    '''

    return ((spectrogram.T - frequency_stats.means[0:185]) / frequency_stats.stds[0:185]).T

def normalize_spectrogram(spectrogram):
    spectrogram2 = spectrogram + 0.00001
    return spectrogram2 / numpy.sum(spectrogram2, 0)

# Given a 2-dimensional array, return an array with smaller
# dimensions. Specifically, skip every (stride1-1) elements in
# dimension 1 and (stride2-1) elements in dimension 2
def reduce_dimensions(arr, stride1, stride2):
    return arr[0:arr.shape[0]:stride1, 0:arr.shape[1]:stride2]

def test_segment(segment):
    global graph

    with graph.as_default():
        abs_spectrogram = numpy.absolute(librosa.stft(segment, n_fft=512))
        abs_spectrogram = abs_spectrogram[0:185, 0:173]
        normalized_spectrogram = reduce_dimensions(center_unit(normalize_spectrogram(abs_spectrogram)), 3, 3)
        sample_segment_spectrogram = numpy.expand_dims(normalized_spectrogram, axis=0)
        sample_segment_spectrogram = numpy.expand_dims(sample_segment_spectrogram, axis=0)
        prediction = model.predict(sample_segment_spectrogram)[0][0]

        global decayed_prediction
        decayed_prediction = decayed_prediction * 0.6 + prediction * 0.4

        predictions_str = [' '] * 70
        num_squares = int(round(0.999 * prediction * len(predictions_str)))
        predictions_str[0:num_squares] = [unichr(9608)] * num_squares; # http://www.fileformat.info/info/unicode/char/2588/index.htm

        print u'Not Speech [{0}] Speech ({1}%)'.format(u''.join(predictions_str), int(round(prediction * 100)))

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    AudioRecording().start()

