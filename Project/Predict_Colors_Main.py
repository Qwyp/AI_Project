import Autoencoder as auto
import Plot as plot
import Save_Data as savedata
import numpy as np
from keras.datasets import cifar10
from keras.layers import Input, Dense
from keras.layers import Reshape, Conv2DTranspose
from keras.layers import Conv2D, Flatten
from keras.models import Model
from keras import backend as K

# Global Parameteres and Initializations:
batch_size = 32
kernel_size = 3
latent_dim = 256
layer_filters = [64,128,256]



def handler():
    """This function handles all operations needed for the training and testing of the cifar10 dataset to predict the color of the images"""

    # 1. Load the cifar10 dataset
    (x_train,_),(x_test,_) = cifar10.load_data()

    # 2. Input Image dimensions
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]

    # input_shape is now (32,32,1)
    input_shape = (img_rows,img_cols,1)

    # 3. Create a folder for the images to be tested
    savedata.save_image()
    # 4. Display the first 100 images (input in original color)
    plot.display_images_original_color(x_test,img_rows,img_cols,channels,'saved_images')
    # 5. Display the grayscaled images of first 100 inputs
    # 6. Build autoencoder Model
    # 6.1 Build encoder model
    # 6.2 Build decoder model
    # 6.3 Build autoencoder model
    auto.build_autoencoder_model(batch_size,input_shape,kernel_size,latent_dim,layer_filters,x_train,x_test,img_rows,img_cols,channels)

handler()
