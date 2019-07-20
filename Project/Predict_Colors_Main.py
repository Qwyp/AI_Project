import Project.Autoencoder as auto
import Project.Plot as plot
import numpy as np
from keras.datasets import cifar10

# Global Parameteres and Initializations
# Network parameters:
    # input_shape
    # batch_size = 32
    # kernel_size = 3
    # latent_dim = 256
    # layer_filters = [64,128,256]
# input_shape = (img_rows,img_cols,1)
# img_cols
# img_rows
# img_gray
# img_normal
# channels
# image directory


def handler():
    'This function handles all operations needed for the training of the cifar10 dataset'

    # 1. Load the dataset
    # 2. Input Image dimensions
    # 3. Create a folder for the images to be tested (in original color) + save the image
    # 4. Create a folder for the images to tested in grayscale + save the image
    # Normalize output
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    # Normalize input
    x_train_gray = x_train_gray.astype('float32') / 255
    x_test_gray = x_test_gray.astype('float32') / 255
    # Reshape Images
    x_train_gray = x_train_gray.reshape(x_train_gray.shape[0], img_rows,img_cols, 1)
    x_test_gray = x_test_gray.reshape(x_test_gray.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows,img_cols,1)

    # 5. Train the model + save each epoche
    # 6. Show the output of the tested images (new color)
    # 7. Show accuracy

handler()
