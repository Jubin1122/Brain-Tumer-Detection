# Tensorflow
from torch import conv_transpose1d
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental import preprocessing
from keras.utils.vis_utils import plot_model

class segment:

    def __init__(self, n_filters, input_size, n_classes):
        self.n_filters = n_filters  # n_filters= 64
        self.input_size = input_size #input_size=(256, 256, 3)
        self.n_classes = n_classes  #n_classes=1

            

    ## Encoder
    def conv_block(self, inputs, filters, max_pooling):

        # Convolutional layers
        conv = layers.Conv2D(filters,
                    kernel_size = (3, 3),
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(inputs)
        conv = layers.Conv2D(filters,
                    kernel_size = (3, 3),
                    activation = None,
                    padding = 'same',
                    kernel_initializer = 'he_normal')(conv)
        skip_connection = conv
        conv = layers.BatchNormalization(axis=3)(conv)
        conv = layers.Activation('relu')(conv)
            
        # if max_pooling is True add a MaxPooling2D with 2x2 pool_size
        if max_pooling:
            next_layer = layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2))(conv)
        else:
            next_layer = conv
        
        return next_layer, skip_connection

    ## Decoder
    def upsampling_block(self, expansive_input, contractive_input, filters):
        """
            Convolutional upsampling block
            
            Arguments:
                expansive_input -- Input tensor from previous layer
                contractive_input -- Input tensor from previous skip layer
                n_filters -- Number of filters for the convolutional layers
            Returns: 
                conv -- Tensor output
        """

        # Transpose convolution
        up = layers.Conv2DTranspose(
                    filters,
                    kernel_size = (2, 2),
                    strides = (2, 2),
                    padding = 'same')(expansive_input)
        
        # Merge the previous output and the contractive_input
        merge = layers.concatenate([up, contractive_input], axis=3)
        conv = layers.Conv2D(filters,
                    kernel_size = (3, 3),
                    activation = 'relu',
                    padding = 'same',
                    kernel_initializer = 'he_normal')(merge)
        conv = layers.Conv2D(filters,
                    kernel_size = (3, 3),
                    activation = None,
                    padding = 'same',
                    kernel_initializer = 'he_normal')(conv)
        conv = layers.BatchNormalization(axis=3)(conv)
        conv = layers.Activation('relu')(conv)
        
        return conv

    def build_unet(self):

        """
        Unet model
        
        Arguments:
            input_size -- Input shape 
            n_filters -- Number of filters for the convolutional layers
            n_classes -- Number of output classes
        Returns: 
            model -- tf.keras.Model
        """
        
        # Input layer
        inputs = layers.Input(self.input_size)
        
        # Encoder (double the number of filters at each step)
        cblock1 = self.conv_block(inputs, self.n_filters, True)
        cblock2 = self.conv_block(cblock1[0], 2*self.n_filters, True)
        cblock3 = self.conv_block(cblock2[0], 4*self.n_filters, True)
        cblock4 = self.conv_block(cblock3[0], 8*self.n_filters, True)
        cblock5 = self.conv_block(cblock4[0], 16*self.n_filters, False) 

        # Decoder (halve the number of filters at each step)
        ublock6 = self.upsampling_block(cblock5[0], cblock4[1],  8*self.n_filters)
        ublock7 = self.upsampling_block(ublock6, cblock3[1],  4*self.n_filters)
        ublock8 = self.upsampling_block(ublock7, cblock2[1],  2*self.n_filters)
        ublock9 = self.upsampling_block(ublock8, cblock1[1],  self.n_filters)

        # 1x1 convolution
        conv10 = layers.Conv2D(filters = self.n_classes,
                    kernel_size = (1, 1),
                    activation = 'sigmoid',    # use softmax if n_classes>1
                    padding = 'same')(ublock9)

        model = keras.Model(inputs=inputs, outputs=conv10)

        return model










