from keras.models import Sequential
from keras.optimizers import Adam, Adadelta
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2, l1, l1l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

def make_model():
    model = Sequential()

    L2_REGULARIZATION = 0.1
    L1_REGULARIZATION = 0
    INITIAL_DROPOUT = 0
    DROPOUT = 0
    FC_DROPOUT = 0.5
    GAUSSIAN_NOISE = 0

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 185, 173)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(48, 5, 1, W_regularizer=l1(L1_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(64, 3, 1, W_regularizer=l1(L1_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 1)))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(72, 3, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Convolution2D(72, 3, 3, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(72, 1, 5, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(72, 1, 5, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Convolution2D(96, 1, 5, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Convolution2D(96, 1, 5, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(AveragePooling2D(pool_size=(1, 2)))

    model.add(Flatten())

    model.add(Dropout(FC_DROPOUT))
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(384, W_regularizer=l2(L2_REGULARIZATION)))

    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))
    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(96, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(Dropout(FC_DROPOUT))

    model.add(GaussianNoise(GAUSSIAN_NOISE))
    model.add(Dense(1, W_regularizer=l2(L2_REGULARIZATION)))
    model.add(Activation('sigmoid'))

    model.compile(optimizer=Adadelta(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
