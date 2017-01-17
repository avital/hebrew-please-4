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
from grl import GradientReversal

def make_model():
    model = Sequential()

    L2_REGULARIZATION = 0.03
    FC_DROPOUT = 0.5
    DOMAIN_CLASSIFIER_GRL_FACTOR = 1.0

    main_input = Input(shape=(1, 62, 58), name='input')
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

    domain_classifier = GradientReversal(DOMAIN_CLASSIFIER_GRL_FACTOR)(x)
    domain_classifier = Dropout(FC_DROPOUT)(x)
    domain_classifier = Dense(196)(domain_classifier)
    domain_classifier = BatchNormalization()(domain_classifier)
    domain_classifier = ELU()(domain_classifier)

    domain_classifier = Dropout(FC_DROPOUT)(domain_classifier)
    domain_classifier = Dense(196)(domain_classifier)
    domain_classifier = BatchNormalization()(domain_classifier)
    domain_classifier = ELU()(domain_classifier)

    domain_classifier = Dropout(FC_DROPOUT)(domain_classifier)

    domain_classifier = Dense(1)(domain_classifier)
    predicted_domain = Activation('sigmoid', name='predicted_domain')(domain_classifier)

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
    predicted_label = Activation('sigmoid', name='predicted_label')(x)

    full_model = Model(input=main_input, output=[predicted_label, predicted_domain])
    full_model.compile(optimizer=Adadelta(),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

    label_only_model = Model(input=main_input, output=predicted_label)
    label_only_model.compile(optimizer=Adadelta(),
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

    return (full_model, label_only_model)
