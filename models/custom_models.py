import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


class SoftAttention(Layer):
    # Adapted form:
    # https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    # in order to implement the attention of:
    # "Heart Sound Segmentation Using Bidirectional LSTMs With Attention", Fernando et al. 2019
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(SoftAttention, self).__init__()

    def build(self, input_shape):
        # input_shape = (None, patch_size, hidden_size)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")  # hidden_size x 1
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")  # patch_size

        self.v = self.add_weight(name="context_vector", shape=(input_shape[-2], 1),
                                 initializer="normal")  # patch_size

        super(SoftAttention, self).build(input_shape)

    def call(self, x):
        e = tf.nn.tanh(K.dot(x, self.W) + self.b)  # patch_size x 1
        s = K.dot(tf.keras.layers.Permute((2, 1))(e), self.v)  # Permute to transpose window for each time step
        a = tf.nn.softmax(s, axis=1)  # path_size x 1
        output = x * a
        if self.return_sequences:
            return output

        return tf.reduce_sum(output, axis=2)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att_weight': self.W.numpy(),  # convert to numpy to be json-serializable
            'att_bias': self.b.numpy(),
            'context_vector': self.v.numpy()
        })
        return config


def unet_pcg(nch, patch_size, dropout=0.0):
    inputs = Input(shape=(patch_size, nch))
    conv1 = Conv1D(8, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv1D(8, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = Conv1D(16, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv1D(16, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = Conv1D(32, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv1D(32, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = Conv1D(64, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv1D(64, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling1D(pool_size=2)(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = Conv1D(128, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv1D(128, 3, activation='relu', padding='same')(conv5)

    up6_prep = UpSampling1D(size=2)(conv5)

    up6 = concatenate([Conv1D(64, 2, padding='same')(up6_prep), conv4], axis=2)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(up6)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(conv6)

    up7_prep = UpSampling1D(size=2)(conv6)

    up7 = concatenate([Conv1D(64, 2, padding='same')(up7_prep), conv3], axis=2)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(up7)
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(conv7)

    up8_prep = UpSampling1D(size=2)(conv7)

    up8 = concatenate([Conv1D(32, 2, padding='same')(up8_prep), conv2], axis=2)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv1D(16, 3, activation='relu', padding='same')(up8)
    conv8 = Conv1D(16, 3, activation='relu', padding='same')(conv8)

    up9_prep = UpSampling1D(size=2)(conv8)

    up9 = concatenate([Conv1D(8, 2, padding='same')(up9_prep), conv1], axis=2)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv1D(8, 3, activation='relu', padding='same')(up9)
    conv9 = Conv1D(8, 3, activation='tanh', padding='same')(conv9)

    conv10 = Conv1D(4, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def simple_convnet(nch, patch_size):
    inputs = Input(shape=(patch_size, nch))
    conv1 = Conv1D(8, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(16, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    conv3 = Conv1D(32, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    dense_in = Flatten()(pool3)
    drop_in = Dropout(0.25)(dense_in)
    dense1 = Dense(64, activation='relu')(drop_in)
    dense3 = Dense(4, activation='softmax')(dense1)

    model = Model(inputs=[inputs], outputs=[dense3])
    return model


def simple_convnet2d(nch, patch_size):
    inputs = Input(shape=(patch_size, nch, 1))
    conv1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=2)(conv1)
    conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=2)(conv2)
    conv3 = Conv2D(32, 3, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=2)(conv3)

    dense_in = Flatten()(pool3)
    drop_in = Dropout(0.25)(dense_in)
    dense1 = Dense(64, activation='relu')(drop_in)
    dense3 = Dense(4, activation='softmax')(dense1)

    model = Model(inputs=[inputs], outputs=[dense3])
    model.summary()
    return model


def bilstm_attention_fernando19_softmax(nch, patch_size, unit_size=80):
    inputs = Input(shape=(patch_size, nch))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit_size, return_sequences=True))(inputs)
    attention = SoftAttention(return_sequences=False)(x)

    #  We change output layer to softmax.
    # Reason: the authors use passthrough layers, which do not yield valid posterior estimates
    # in order to decode sequence.
    dense = tf.keras.layers.Dense(4, activation='softmax')(  #
        attention)  # R

    model = Model(inputs=[inputs], outputs=[dense])
    model.summary()
    return model
