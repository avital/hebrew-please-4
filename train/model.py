from keras.models import Sequential, Model
from keras.optimizers import Adam, Adadelta
from keras.layers import Input
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2, activity_l2, l1, l1l2
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise

def make_model():
    model = Sequential()

    L2_REGULARIZATION = 0.03
    FC_DROPOUT = 0.5

    main_input = Input(shape=(1, 62, 58))
    x = ZeroPadding2D((1, 1))(main_input)
    x = Convolution2D(64, 3, 3, W_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, W_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, W_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Convolution2D(64, 3, 3, W_regularizer=l2(L2_REGULARIZATION))(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = MaxPooling2D(pool_size=(2, 2))(x)
    # Dimensions are 15x14, which correspond to (time, frequency) xcxc

    x = Flatten()(x)

    x = Dropout(FC_DROPOUT)(x)
    x = Dense(196)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = Dropout(FC_DROPOUT)(x)
    x = Dense(196)(x)
    x = BatchNormalization()(x)
    x = ELU()(x)

    x = Dropout(FC_DROPOUT)(x)

    x = Dense(1)(x)
    predicted_label = Activation('sigmoid')(x)

    model = Model(input=[main_input], output=[predicted_label])
    model.compile(optimizer=Adadelta(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model
