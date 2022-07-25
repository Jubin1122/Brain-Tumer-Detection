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

<figure>
  <img src="img/tumer.png" alt=".." title="Optional title" />
</figure>

<h4 align="center">Tumer</h4>

