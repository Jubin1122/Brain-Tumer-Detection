<h1 align="center">Brain Tumer Detection</h1>

### Abstract
A Brain tumor is considered as one of the aggressive diseases, among children and adults. Brain tumors account for 85 to 90 percent of all primary Central Nervous System(CNS) tumors. Every year, around 11,700 people are diagnosed with a brain tumor. The 5-year survival rate for people with a cancerous brain or CNS tumor is approximately 34 percent for men and36 percent for women. Brain Tumors are classified as: Benign Tumor, Malignant Tumor, Pituitary Tumor, etc. Proper treatment, planning, and accurate diagnostics should be implemented to improve the life expectancy of the patients. The best technique to detect brain tumors is Magnetic Resonance Imaging (MRI). A huge amount of image data is generated through the scans. These images are examined by the radiologist. A manual examination can be error-prone due to the level of complexities involved in brain tumors and their properties.
Application of automated classification techniques using Machine Learning(ML) and Artificial Intelligence(AI)has consistently shown higher accuracy than manual classification. Hence, proposing a system performing detection and classification by using Deep Learning Algorithms using Convolution-Neural Network (CNN), Artificial Neural Network (ANN), and Transfer-Learning (TL) would be helpful to doctors all around the world.

### Context
Brain Tumors are complex. There are a lot of abnormalities in the sizes and location of the brain tumor(s). This makes it really difficult for complete understanding of the nature of the tumor. Also, a professional Neurosurgeon is required for MRI analysis. Often times in developing countries the lack of skillful doctors and lack of knowledge about tumors makes it really challenging and time-consuming to generate reports from MRI’. So an automated system on Cloud can solve this problem.

### Data
In this project we are segmenting organs cells in images. The training annotations are provided as RLE-encoded masks, and the images are in 16-bit grayscale PNG format.

Each case in this competition is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. 

Note that, in this case, the test set is entirely unseen. It is roughly 50 cases, with a varying number of days and slices, as seen in the training set.

### Goal
To Detect and Classify Brain Tumor using, CNN or ANN; as an asset of Deep Learning and to examine the tumor position(segmentation). 

In this project, I will be using U-net to perform image segmentation, which has tremendous application in medical imaging. Also, finally we will be able to generalize to both partially and wholly unseen cases.

![architecture](img/UNET.png)

### Modeling Steps

-  **Data Curation**: First of all we have the MRI scanned image of the organ, and their respective RLE-encoded masks. To understand the data distribution, I created a data frame, which consists of three columns namely image_path, it's respective mask_path, and diagnosis. 
Here, if mask has a maximum pixel value greater than 0, I am flagging it as 1(Tumer), otherwise 0(no tumer).

![Data Distribution](img/data_dist.png)

- **EDA**:  For understanding the images furtherm, I randomly selected 5 samples for tumer and no-tumer condition, which I assigned it as positive(tumer) and negative(no-tumer). 
    After that, I created a grid and plot the 5 images in the RGB format for the positive and the negative case.

<h4 align="center">Tumer</h4>
<figure>
  <img src="img/tumer.png" alt=".." title="Optional title" />
</figure>

<p></p>

<h4 align="center">No Tumer</h4>
<figure>
  <img src="img/no_tumer.png" alt=".." title="Optional title" />
</figure>

- **Model**: Instead of going with general convolutional neural network, which focuses on image classification, where input is an image and output is a lable. However, in our case we are required to localize the area of abnormality. UNet is dedicated in solving this problem. 

    UNet is able to do image localisation by predicting the image pixel by pixel and the author of UNet claims in his [paper](https://arxiv.org/abs/1505.04597) that the network is strong enough to do good prediction based on even few data sets by using excessive data augmentation techniques.

    -   To optimze the model design, I create a class to divide the entire architecture into 'encoder' function, 'decoder' function, and the 'build' function. Inside the build class we can create multiple encoder and decoder blocks.

    - After reaching to the bottomost layer of U-Net architecture, we have to perform an expansion process, where the image is going to be upsized to its original size.

    ```
    conv_2d_transpose -> concatenate -> conv_layer1 -> conv_layer2
    ```
    ![Expansion Path](img/bottom_layer.png)

    - The l last layer is a convolution layer with 1 filter of size 1x1(notice that there is no dense layer in the whole network). And the rest left is the same for neural network training.


    ![Expansion Path](img/last_layer.png)

    ```python
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

    ```