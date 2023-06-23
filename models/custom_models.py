import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D, Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv1D, MaxPooling1D, UpSampling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K


# TODO: check imports to avoid lambda layers
class Attention(Layer):
    # implementation form:
    # https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")

        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att_weight': self.W,
            'att_bias': self.b,
            'context_vector': self.v
        })
        return config


class SoftWindowedAttention(Layer):
    # Adapted form:
    # https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    # to implementent the mechanism attention of:
    # "Hierarchical Attention Networks for Document Classification", Yang et al. 2016
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(SoftWindowedAttention, self).__init__()

    def build(self, input_shape):
        # input_shape = (None, patch_size, hidden_size)
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="normal")  # hidden_size x 1
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros")  # patch_size

        self.v = self.add_weight(name="context_vector", shape=(input_shape[-1], 1),
                                 initializer="normal")  # patch_size

        super(SoftWindowedAttention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # patch_size x 1
        a = K.softmax(K.dot(K.transpose(e), self.v), axis=1)  # path_size x 1
        output = x * a  # (path_size, hidden) * (patch_size x 1)

        if self.return_sequences:
            return output

        return K.sum(output, axis=1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'att_weight': self.W,
            'att_bias': self.b,
            'context_vector': self.v
        })
        return config


class SoftAttention(Layer):
    # Adapted form:
    # https://stackoverflow.com/questions/62948332/how-to-add-attention-layer-to-a-bi-lstm/62949137#62949137
    # to implementent the mechanism attention of:
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


def vgg16_fine_tune(num_classes, input_shape=(224, 224, 3)):
    vgg_model = tf.keras.applications.VGG16(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=input_shape, pooling=None, classes=3,
        classifier_activation='softmax'
    )
    x = vgg_model.output
    x = Flatten()(x)  # Flatten dimensions for the FC layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)  # Dropout layer to reduce overfitting
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)  # Softmax for multiclass
    return Model(inputs=vgg_model.input, outputs=x)


def test_net(nch):
    inputs = Input(shape=(nch, 1))
    conv1 = Conv1D(32, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    conv2 = Conv1D(64, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    dense_in = Flatten()(pool2)
    drop_in = Dropout(0.5)(dense_in)
    dense3 = Dense(4, activation='softmax')(drop_in)
    model = Model(inputs=[inputs], outputs=[dense3])
    return model


def conv1d_block(input_tensor, n_filters, kernel_size=3):
    """
    Adds 2 convolutional layers with the parameters passed to it

    Args:
      input_tensor (tensor) -- the input tensor
      n_filters (int) -- number of filters
      kernel_size (int) -- kernel size for the convolution

    Returns:
      tensor of output features
    """
    x = input_tensor
    for i in range(2):
        x = tf.keras.layers.Conv1D(filters=n_filters, kernel_size=kernel_size, \
                                   kernel_initializer='he_normal', padding='same')(x)
        x = tf.keras.layers.Activation('relu')(x)

    return x


def encoder1d_block(inputs, n_filters=64, pool_size=2, dropout=0.0):
    """
  Adds two convolutional blocks and then perform down sampling on output of convolutions.

  Args:
    inputs(tensor) -- the input tensor
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size for the convolution

  Returns:
    f - the output features of the convolution block
    p - the maxpooled features with dropout
  """

    f = conv1d_block(inputs, n_filters=n_filters)
    p = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(f)
    p = tf.keras.layers.Dropout(dropout)(p)

    return f, p


def encoder1d(inputs):
    """
  This function defines the encoder or downsampling path.

  Args:
    inputs (tensor) -- batch of input images

  Returns:
    p5 - the output maxpooled features of the last encoder block
    (f1, f2, f3, f4, f5) - the output features of all the encoder blocks
  """
    f1, p1 = encoder1d_block(inputs, n_filters=8, pool_size=2, dropout=0.3)
    f2, p2 = encoder1d_block(p1, n_filters=16, pool_size=2, dropout=0.3)
    f3, p3 = encoder1d_block(p2, n_filters=32, pool_size=2, dropout=0.3)
    f4, p4 = encoder1d_block(p3, n_filters=64, pool_size=2, dropout=0.3)

    return p4, (f1, f2, f3, f4)


def bottleneck1d(inputs):
    '''
    This function defines the bottleneck convolutions to extract more features before the upsampling layers.
    '''

    return conv1d_block(inputs, n_filters=128)


def decoder_block1d(inputs, conv_output, n_filters=64, kernel_size=3, strides=2, dropout=0.0):
    """
  defines the one decoder block of the UNet

  Args:
    inputs (tensor) -- batch of input features
    conv_output (tensor) -- features from an encoder block
    n_filters (int) -- number of filters
    kernel_size (int) -- kernel size
    strides (int) -- strides for the deconvolution/upsampling
    padding (string) -- "same" or "valid", tells if shape will be preserved by zero padding

  Returns:
    c (tensor) -- output features of the decoder block
    """
    u = tf.keras.layers.Conv1DTranspose(n_filters, kernel_size=kernel_size, strides=strides, padding='same')(inputs)
    c = tf.keras.layers.concatenate([u, conv_output])
    c = tf.keras.layers.Dropout(dropout)(c)
    c = conv1d_block(c, n_filters, kernel_size=3)
    return c


def decoder1d(inputs, convs, output_channels):
    '''
    Defines the decoder of the UNet chaining together 4 decoder blocks.

    Args:
      inputs (tensor) -- batch of input features
      convs (tuple) -- features from the encoder blocks
      output_channels (int) -- number of classes in the label map

    Returns:
      outputs (tensor) -- the pixel wise label map of the image
    '''

    f1, f2, f3, f4 = convs

    c6 = decoder_block1d(inputs, f4, n_filters=64, kernel_size=2, strides=2, dropout=0.3)
    c7 = decoder_block1d(c6, f3, n_filters=32, kernel_size=2, strides=2, dropout=0.3)
    c8 = decoder_block1d(c7, f2, n_filters=16, kernel_size=2, strides=2, dropout=0.3)
    c9 = decoder_block1d(c8, f1, n_filters=8, kernel_size=2, strides=2, dropout=0.3)

    outputs = tf.keras.layers.Conv1D(output_channels, 1, activation='softmax')(c9)
    return outputs


def unet1d(*, number_channels, patch_size):
    inputs = Input(shape=(patch_size, number_channels))
    encoder_output, convs = encoder1d(inputs)
    bottle_neck = bottleneck1d(encoder_output)
    outputs = decoder1d(bottle_neck, convs, output_channels=number_channels)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


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
    # up6_prep=conv5

    up6 = concatenate([Conv1D(64, 2, padding='same')(up6_prep), conv4], axis=2)
    up6 = Dropout(dropout)(up6)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(up6)
    conv6 = Conv1D(64, 3, activation='relu', padding='same')(conv6)

    up7_prep = UpSampling1D(size=2)(conv6)
    # up7_prep=conv6

    up7 = concatenate([Conv1D(64, 2, padding='same')(up7_prep), conv3], axis=2)
    up7 = Dropout(dropout)(up7)
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(up7)
    conv7 = Conv1D(32, 3, activation='relu', padding='same')(conv7)

    up8_prep = UpSampling1D(size=2)(conv7)
    # up8_prep = conv7

    up8 = concatenate([Conv1D(32, 2, padding='same')(up8_prep), conv2], axis=2)
    up8 = Dropout(dropout)(up8)
    conv8 = Conv1D(16, 3, activation='relu', padding='same')(up8)
    conv8 = Conv1D(16, 3, activation='relu', padding='same')(conv8)

    up9_prep = UpSampling1D(size=2)(conv8)
    # up9_prep=conv8

    up9 = concatenate([Conv1D(8, 2, padding='same')(up9_prep), conv1], axis=2)
    up9 = Dropout(dropout)(up9)
    conv9 = Conv1D(8, 3, activation='relu', padding='same')(up9)
    conv9 = Conv1D(8, 3, activation='tanh', padding='same')(conv9)

    conv10 = Conv1D(4, 1, activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model


def deep_averaging_network(nch, patch_size):
    inputs = Input(shape=(patch_size, nch))
    channel_average = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, axis=0))(inputs)
    dense1 = Dense(64, activation='relu')(channel_average)
    dense2 = Dense(32, activation='relu')(dense1)
    dense3 = Dense(4, activation='softmax')(dense2)

    model = Model(inputs=[inputs], outputs=[dense3])
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
    model.summary()
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


def simple_rnn(nch, patch_size):
    inputs = Input(shape=(patch_size, nch))

    # embedding = tf.keras.layers.Embedding(nch*patch_size, embedding_size) # TODO: Investigate if Fernando19 (2) equation is an embedding layer
    x = tf.keras.layers.SimpleRNN(patch_size, return_sequences=True)(inputs)
    x = Flatten()(x)
    dense = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = Model(inputs=[inputs], outputs=[dense])
    model.summary()
    return model


def simple_bilstm(nch, patch_size, unit_size=80):
    inputs = Input(shape=(patch_size, nch))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit_size, return_sequences=True))(inputs)
    dense = tf.keras.layers.TimeDistributed(Dense(4, activation='softmax'))(x)
    model = Model(inputs=[inputs], outputs=[dense])
    model.summary()
    return model


def bilstm_attention_fernando19_softmax(nch, patch_size, unit_size=80):
    inputs = Input(shape=(patch_size, nch))

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(unit_size, return_sequences=True))(inputs)
    attention = SoftAttention(return_sequences=False)(x)
    dense = tf.keras.layers.Dense(4, activation='softmax')(
        attention)  # TODO: to reproduce the code as is use passthrough

    model = Model(inputs=[inputs], outputs=[dense])
    model.summary()
    return model
